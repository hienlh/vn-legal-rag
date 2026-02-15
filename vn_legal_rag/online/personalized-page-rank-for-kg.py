"""
Personalized PageRank for Legal Knowledge Graph

Intent-aware PageRank implementation for article scoring.
Propagates scores through KG entity relationships with
intent-weighted edge weights.

Usage:
    >>> ppr = PersonalizedPageRank(kg, embedding_gen)
    >>> seeds = {"entity_1": 0.8, "entity_2": 0.5}
    >>> result = ppr.compute(seeds, intents=[{"intent": "concept", "confidence": 0.8}])
    >>> article_scores = ppr.map_to_articles(result.scores)
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Intent-relation weight matrix
INTENT_RELATION_WEIGHTS = {
    "concept": {
        "ĐỊNH_NGHĨA_LÀ": 1.0, "DEFINED_AS": 1.0,
        "LÀ_MỘT": 0.95, "IS_A": 0.95,
        "BAO_GỒM": 0.85, "INCLUDES": 0.85,
        "LIÊN_QUAN_ĐẾN": 0.5, "RELATED_TO": 0.5,
        "THAM_CHIẾU": 0.4, "REFERENCES": 0.4,
    },
    "procedure": {
        "YÊU_CẦU": 1.0, "REQUIRES": 1.0,
        "XẢY_RA_TRƯỚC": 0.95, "PRECEDES": 0.95,
        "XẢY_RA_SAU": 0.9, "FOLLOWS": 0.9,
        "ĐIỀU_KIỆN_CHO": 0.85, "CONDITION_FOR": 0.85,
    },
    "right": {
        "CÓ_QUYỀN": 1.0, "HAS_AUTHORITY_OVER": 0.95,
        "CÓ_THẨM_QUYỀN": 0.95,
        "ỦY_QUYỀN_CHO": 0.85, "DELEGATES_TO": 0.85,
        "ÁP_DỤNG_CHO": 0.7, "APPLIES_TO": 0.7,
    },
    "obligation": {
        "CÓ_NGHĨA_VỤ": 1.0,
        "CHỊU_TRÁCH_NHIỆM": 0.95, "RESPONSIBLE_FOR": 0.95,
        "YÊU_CẦU": 0.9, "REQUIRES": 0.9,
        "BỊ_PHẠT": 0.7, "HAS_PENALTY": 0.7,
    },
    "consequence": {
        "BỊ_PHẠT": 1.0, "HAS_PENALTY": 1.0,
        "CÓ_CHẾ_TÀI": 0.95,
        "VI_PHẠM": 0.85,
        "RESULTS_IN": 0.9,
    },
    "general": {
        "RELATED_TO": 0.5, "LIÊN_QUAN_ĐẾN": 0.5,
        "REFERENCES": 0.4, "THAM_CHIẾU": 0.4,
        "REQUIRES": 0.5, "YÊU_CẦU": 0.5,
    },
}

# Symmetric relations (bidirectional)
SYMMETRIC_RELATIONS = {
    "RELATED_TO", "LIÊN_QUAN_ĐẾN",
    "SYNONYM", "ĐỒNG_NGHĨA_VỚI",
    "CONTRADICTS", "MÂU_THUẪN_VỚI",
}


@dataclass
class PPRConfig:
    """Configuration for PPR algorithm."""
    alpha: float = 0.15  # Damping factor
    max_iterations: int = 50
    tolerance: float = 1e-6
    min_query_similarity: float = 0.1
    query_similarity_gamma: float = 0.5


@dataclass
class PPRResult:
    """Result from PPR computation."""
    scores: Dict[str, float]
    iterations: int
    converged: bool
    total_nodes: int
    total_edges: int


class PersonalizedPageRank:
    """
    Intent-aware Personalized PageRank for KG-based article scoring.

    Uses relation type weights based on query intent to bias propagation
    toward more relevant entity relationships.
    """

    def __init__(
        self,
        kg: Dict[str, Any],
        embedding_gen: Optional[Any] = None,
        config: Optional[PPRConfig] = None,
    ):
        """
        Initialize PPR with knowledge graph.

        Args:
            kg: Knowledge graph dict with 'entities' and 'relationships'
            embedding_gen: Optional embedding generator for query similarity
            config: PPR configuration
        """
        self.kg = kg
        self.embedding_gen = embedding_gen
        self.config = config or PPRConfig()

        # Build graph structure
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []
        self.adjacency: Dict[str, List[Dict]] = defaultdict(list)
        self.node_articles: Dict[str, str] = {}

        # Node embeddings for query similarity
        self._node_embeddings: Dict[str, np.ndarray] = {}

        self._build_graph()

    def _build_graph(self):
        """Build adjacency graph from KG entities and relationships."""
        # Index entities
        for entity in self.kg.get("entities", []):
            eid = entity.get("id", "")
            if not eid:
                continue
            self.nodes[eid] = entity

            # Extract article ID from source_ids
            metadata = entity.get("metadata", {})
            source_ids = metadata.get("source_ids", [])
            if not source_ids and metadata.get("source_id"):
                source_ids = [metadata["source_id"]]

            for source_id in source_ids:
                match = re.search(r":d(\d+)$", source_id)
                if match:
                    self.node_articles[eid] = match.group(1)
                    break

        # Index relationships
        for rel in self.kg.get("relationships", []):
            source = rel.get("source_id", rel.get("source", ""))
            target = rel.get("target_id", rel.get("target", ""))

            if source in self.nodes and target in self.nodes:
                self.edges.append(rel)
                self.adjacency[source].append(rel)

    def build_node_embeddings(self):
        """Build embeddings for all nodes (call once after init)."""
        if not self.embedding_gen:
            return

        texts = []
        ids = []
        for eid, entity in self.nodes.items():
            text = entity.get("name", "")
            if text:
                texts.append(text)
                ids.append(eid)

        if texts:
            try:
                embs = self.embedding_gen.generate_embeddings(texts)
                for i, eid in enumerate(ids):
                    self._node_embeddings[eid] = np.array(embs[i])
            except Exception:
                pass

    def compute(
        self,
        seeds: Dict[str, float],
        intents: Optional[List[Dict[str, Any]]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> PPRResult:
        """
        Run Personalized PageRank from seed nodes.

        Args:
            seeds: Dict of seed entity IDs to initial scores
            intents: List of intent dicts with 'intent' and 'confidence' keys
            query_embedding: Optional query embedding for similarity weighting

        Returns:
            PPRResult with entity scores
        """
        if not seeds or not self.nodes:
            return PPRResult(
                scores={}, iterations=0, converged=True,
                total_nodes=0, total_edges=0
            )

        intents = intents or [{"intent": "general", "confidence": 0.5}]

        # Build node index
        node_ids = list(self.nodes.keys())
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)

        # Compute query similarities
        query_sims = {}
        if query_embedding is not None and self._node_embeddings:
            for nid, emb in self._node_embeddings.items():
                sim = self._cosine_similarity(query_embedding, emb)
                query_sims[nid] = max(self.config.min_query_similarity, sim)

        # Build transition matrix
        matrix = self._build_transition_matrix(node_index, n, intents, query_sims)

        # Build personalization vector
        personal = np.zeros(n)
        total_weight = sum(seeds.values())
        for nid, weight in seeds.items():
            if nid in node_index:
                personal[node_index[nid]] = weight / total_weight

        # Power iteration
        pr = personal.copy()
        converged = False
        iterations = 0
        alpha = self.config.alpha

        for i in range(self.config.max_iterations):
            iterations = i + 1
            new_pr = alpha * personal + (1 - alpha) * matrix.T @ pr
            diff = np.sum(np.abs(new_pr - pr))
            pr = new_pr

            if diff < self.config.tolerance:
                converged = True
                break

        # Map back to entity IDs
        scores = {node_ids[i]: float(pr[i]) for i in range(n) if pr[i] > 0}

        return PPRResult(
            scores=scores,
            iterations=iterations,
            converged=converged,
            total_nodes=n,
            total_edges=len(self.edges),
        )

    def _build_transition_matrix(
        self,
        node_index: Dict[str, int],
        n: int,
        intents: List[Dict[str, Any]],
        query_sims: Dict[str, float],
    ) -> np.ndarray:
        """Build intent-aware transition matrix."""
        matrix = np.zeros((n, n))
        gamma = self.config.query_similarity_gamma

        for edge in self.edges:
            source = edge.get("source_id", edge.get("source", ""))
            target = edge.get("target_id", edge.get("target", ""))

            if source not in node_index or target not in node_index:
                continue

            src_idx = node_index[source]
            tgt_idx = node_index[target]

            # Get intent-aware relation weight
            rel_type = edge.get("type", "RELATED_TO")
            rel_weight = self._get_multi_intent_weight(intents, rel_type)

            # Apply query similarity weighting
            tgt_sim = query_sims.get(target, 1.0)
            tgt_sim = tgt_sim ** gamma

            confidence = edge.get("confidence", 1.0)
            weight = rel_weight * confidence * tgt_sim

            matrix[src_idx, tgt_idx] += weight

            # Add symmetric edge if needed
            if rel_type in SYMMETRIC_RELATIONS:
                src_sim = query_sims.get(source, 1.0) ** gamma
                matrix[tgt_idx, src_idx] += rel_weight * confidence * src_sim

        # Row-normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums

        return matrix

    def _get_multi_intent_weight(
        self,
        intents: List[Dict[str, Any]],
        rel_type: str,
    ) -> float:
        """Get weighted relation score from multiple intents."""
        total_weight = 0.0
        total_confidence = 0.0

        for intent_info in intents:
            intent = intent_info.get("intent", "general")
            confidence = intent_info.get("confidence", 0.5)

            weights = INTENT_RELATION_WEIGHTS.get(intent, {})
            rel_weight = weights.get(rel_type, 0.3)

            total_weight += rel_weight * confidence
            total_confidence += confidence

        return total_weight / total_confidence if total_confidence > 0 else 0.3

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def map_to_articles(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Map entity PPR scores to article scores.

        Args:
            scores: Dict of entity_id -> PPR score

        Returns:
            Dict of article_id (short) -> max score
        """
        article_scores: Dict[str, float] = {}

        for entity_id, score in scores.items():
            article_id = self.node_articles.get(entity_id)
            if article_id:
                article_scores[article_id] = max(
                    article_scores.get(article_id, 0),
                    score
                )

        return article_scores
