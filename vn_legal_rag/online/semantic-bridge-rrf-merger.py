"""
Semantic Bridge - RRF-based merger for Tree + DualLevel + KG results.

Merges results from 3 retrieval tiers using Reciprocal Rank Fusion (RRF):
- Tree Traversal (structured navigation)
- DualLevel (global semantic search)
- KG Expansion (cross-chapter relations)

Features:
- RRF score fusion (rank-based, not score-based)
- Cross-chapter KG expansion
- Adjacent article expansion
- Agreement-based validation
"""

from typing import Any, Dict, List, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)


class SemanticBridge:
    """
    Merges Tree, DualLevel, and KG results using Reciprocal Rank Fusion.

    RRF handles different score scales naturally by using rank positions
    instead of raw scores for fusion.
    """

    def __init__(
        self,
        kg: Dict[str, Any],
        db: Optional[Any] = None,
        penalty_config: Optional[Dict[str, Dict[int, float]]] = None,
    ):
        """
        Initialize semantic bridge.

        Args:
            kg: Knowledge graph dict
            db: Optional database for article text retrieval
            penalty_config: Per-document article penalties.
                Format: {doc_id: {article_num: multiplier}}
                e.g., {"59-2020-QH14": {1: 0.15, 14: 0.35}}
        """
        self.kg = kg
        self.db = db
        self._penalty_config = penalty_config or {}

        # Build entity-article mapping for KG expansion
        self._entity_to_article: Dict[str, str] = {}
        self._article_to_entities: Dict[str, List[str]] = {}

        for entity in kg.get("entities", []):
            eid = entity.get("id", "")
            metadata = entity.get("metadata", {})

            sources = metadata.get("source_ids", [])
            if not sources and metadata.get("source_id"):
                sources = [metadata["source_id"]]

            for source in sources:
                self._entity_to_article[eid] = source
                if source not in self._article_to_entities:
                    self._article_to_entities[source] = []
                self._article_to_entities[source].append(eid)

        # Build KG adjacency for relation expansion
        self._adjacency: Dict[str, List[Dict]] = {}
        for rel in kg.get("relations", kg.get("relationships", [])):
            source = rel.get("source_id", rel.get("source", ""))
            if source not in self._adjacency:
                self._adjacency[source] = []
            self._adjacency[source].append(rel)

    def merge_tree_dual_results(
        self,
        tree_result: Any,
        dual_result: Optional[Any],
        kg_results: List[Dict[str, Any]],
        enable_adjacent: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Merge tree, dual, and KG results using RRF.

        Args:
            tree_result: TreeSearchResult
            dual_result: Optional DualLevelResult
            kg_results: List of KG result dicts
            enable_adjacent: Whether to expand with adjacent articles

        Returns:
            Merged list sorted by RRF score
        """
        # If no dual result, use simple tree+KG merge
        if not dual_result or not dual_result.articles:
            return self._merge_tree_and_kg(tree_result, kg_results)

        # RRF weights: tree is primary signal, KG expansion supplements
        source_weights = {
            "tree": 1.4,
            "dual": 0.7,
            "kg": 1.0,
        }

        rrf_scores: Dict[str, float] = {}
        article_data: Dict[str, Dict] = {}
        tree_article_ids: Set[str] = set()
        dual_article_ids: Set[str] = set()

        # 1. Process Tree results
        tree_weight = source_weights["tree"]
        for rank, node in enumerate(tree_result.target_nodes, start=1):
            article_id = node.node_id
            tree_article_ids.add(article_id)

            rrf_contrib = tree_weight * self.compute_rrf_score(rank)
            rrf_scores[article_id] = rrf_scores.get(article_id, 0) + rrf_contrib

            if article_id not in article_data:
                context_text = ""
                if rank - 1 < len(tree_result.contexts):
                    context_text = tree_result.contexts[rank - 1]
                article_data[article_id] = {
                    "text": context_text,
                    "metadata": {
                        "source": "tree",
                        "source_id": article_id,
                    },
                }

        # 2. Process Dual results
        dual_weight = source_weights["dual"]
        dual_articles = dual_result.articles if dual_result else []
        for rank, article in enumerate(dual_articles, start=1):
            article_id = article.get("article_id", "")
            if not article_id:
                continue

            dual_article_ids.add(article_id)
            rrf_contrib = dual_weight * self.compute_rrf_score(rank)
            rrf_scores[article_id] = rrf_scores.get(article_id, 0) + rrf_contrib

            if article_id not in article_data:
                article_text = self._get_article_text(article_id)
                if article_text:
                    article_data[article_id] = {
                        "text": article_text,
                        "metadata": {
                            "source": "dual",
                            "source_id": article_id,
                        },
                    }

        # 3. Process KG results
        kg_weight = source_weights["kg"]
        kg_article_ids: Set[str] = set()
        for rank, kg_ctx in enumerate(kg_results, start=1):
            kg_id = kg_ctx.get("id", f"kg-{rank}")
            rrf_contrib = kg_weight * self.compute_rrf_score(rank)
            rrf_scores[kg_id] = rrf_scores.get(kg_id, 0) + rrf_contrib

            # Track KG article IDs for adjacent expansion
            meta = kg_ctx.get("metadata", {})
            sid = meta.get("source_id", "") or ""
            if sid and ":" in sid and sid.split(":")[-1].startswith("d"):
                kg_article_ids.add(sid)

            if kg_id not in article_data:
                article_data[kg_id] = {
                    "text": kg_ctx.get("text", ""),
                    "metadata": {
                        "source": "kg",
                        **kg_ctx.get("metadata", {}),
                    },
                }

        # 4. Apply article penalties
        self._apply_penalties(rrf_scores)

        # 5. Build merged list sorted by RRF score
        merged = []
        for article_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if article_id in article_data:
                data = article_data[article_id]
                if data["text"]:
                    merged.append({
                        "id": article_id,
                        "text": data["text"],
                        "metadata": data["metadata"],
                        "score": rrf_score,
                    })

        # 6. ADJACENT ARTICLE EXPANSION (from semantica)
        # Close-miss pattern: failures have tree within ±3 of correct answer
        # Union all source IDs for expansion seeds
        adjacent_results = []
        if enable_adjacent:
            all_seed_ids = tree_article_ids | dual_article_ids | kg_article_ids

            # Parse doc_id from available IDs
            doc_id = None
            for aid in all_seed_ids:
                if ":" in aid:
                    doc_id = aid.rsplit(":", 1)[0]
                    break

            if doc_id and all_seed_ids:
                adjacent_results = self._expand_adjacent_articles(
                    seed_article_ids=all_seed_ids,
                    doc_id=doc_id,
                    expansion_range=2,  # ±2 articles
                )
                merged.extend(adjacent_results)

        # 7. Sort by score descending
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 8. Deduplication (after sort to keep higher-scored version)
        merged = self._deduplicate_results(merged)

        # 9. RRF threshold filtering
        # Filter out results with very low RRF scores
        MIN_RRF_THRESHOLD = 0.005
        before_filter = len(merged)
        merged = [r for r in merged if r.get("score", 0) >= MIN_RRF_THRESHOLD]

        return merged

    def _expand_adjacent_articles(
        self,
        seed_article_ids: Set[str],
        doc_id: str,
        expansion_range: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Expand with adjacent articles (±N).

        CLOSE-MISS PATTERN: Research shows failures have tree results
        within ±3 of correct answer. This recovers those cases.

        Args:
            seed_article_ids: Set of article IDs to expand from
            doc_id: Document ID for generating new article IDs
            expansion_range: Number of adjacent articles (±N)

        Returns:
            List of adjacent article dicts with decreasing scores
        """
        adjacent_results = []

        # Extract article numbers from IDs
        article_nums = []
        for aid in seed_article_ids:
            if ":" in aid and aid.split(":")[-1].startswith("d"):
                try:
                    num = int(aid.split(":")[-1][1:])
                    article_nums.append(num)
                except ValueError:
                    continue

        if not article_nums:
            return []

        # Get unique adjacent article numbers
        existing_nums = set(article_nums)
        adjacent_nums = set()

        for num in article_nums:
            for offset in range(1, expansion_range + 1):
                # Add articles before and after
                if num - offset > 0 and (num - offset) not in existing_nums:
                    adjacent_nums.add(num - offset)
                if (num + offset) not in existing_nums:
                    adjacent_nums.add(num + offset)

        # Generate adjacent article dicts with decreasing scores
        for adj_num in sorted(adjacent_nums):
            # Calculate minimum distance to any original article
            min_distance = min(abs(adj_num - orig_num) for orig_num in article_nums)

            # Score must be BELOW RRF scores (~0.008-0.016) to avoid
            # displacing actual retrieval results. Adjacent articles are
            # fallback candidates, not primary results.
            adj_score = max(0.003, 0.008 - (min_distance * 0.002))

            adj_id = f"{doc_id}:d{adj_num}"
            article_text = self._get_article_text(adj_id)

            if article_text:
                adjacent_results.append({
                    "id": f"adjacent-{adj_num}",
                    "text": article_text,
                    "metadata": {
                        "source": "adjacent_expansion",
                        "source_id": adj_id,
                        "article_id": adj_id,
                        "distance": min_distance,
                    },
                    "score": adj_score,
                })

        return adjacent_results

    def expand_with_kg(
        self,
        article_ids: List[str],
        query: str,
        max_expansion: int = 3,
    ) -> List[str]:
        """
        Expand article list using KG relations.

        Finds related articles via entity relationships for cross-chapter coverage.

        Args:
            article_ids: Initial article IDs
            query: User query (for filtering relevance)
            max_expansion: Max articles to add

        Returns:
            List of expanded article IDs
        """
        expanded_ids = set(article_ids)
        candidates: List[tuple] = []

        # Get entities from initial articles
        seed_entities = set()
        for article_id in article_ids:
            entities = self._article_to_entities.get(article_id, [])
            seed_entities.update(entities)

        # Traverse KG to find related entities
        for entity_id in seed_entities:
            if entity_id not in self._adjacency:
                continue

            for rel in self._adjacency[entity_id]:
                target_id = rel.get("target_id", rel.get("target", ""))
                if target_id in self._entity_to_article:
                    target_article = self._entity_to_article[target_id]
                    if target_article not in expanded_ids:
                        rel_type = rel.get("type", "")
                        confidence = rel.get("confidence", 0.5)
                        candidates.append((target_article, confidence, rel_type))

        # Sort by confidence and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        for article_id, _, _ in candidates[:max_expansion]:
            expanded_ids.add(article_id)

        return list(expanded_ids)

    def _get_article_penalty(self, article_id: str) -> float:
        """Return penalty multiplier for article (1.0 = no penalty).

        Extracts doc_id and article_number from article_id and looks up
        in penalty_config. Returns 1.0 if no penalty configured.

        Args:
            article_id: Full article ID (e.g., "59-2020-QH14:d206")
                        or any ID with ":dN" suffix
        """
        if not self._penalty_config:
            return 1.0

        # Extract doc_id and article_number from ID
        # Handle formats: "59-2020-QH14:d5", "adjacent-5", "kg-1", "tree-1"
        if ":d" not in article_id:
            return 1.0

        parts = article_id.rsplit(":d", 1)
        if len(parts) != 2:
            return 1.0

        doc_id = parts[0]
        try:
            article_num = int(parts[1].split(":")[0])  # strip :k1 etc.
        except ValueError:
            return 1.0

        doc_penalties = self._penalty_config.get(doc_id, {})
        return doc_penalties.get(article_num, 1.0)

    def _apply_penalties(self, rrf_scores: Dict[str, float]) -> None:
        """Apply article penalties to RRF scores in-place."""
        if not self._penalty_config:
            return
        for article_id in rrf_scores:
            penalty = self._get_article_penalty(article_id)
            if penalty != 1.0:
                rrf_scores[article_id] *= penalty

    def compute_rrf_score(self, rank: int, k: int = 60) -> float:
        """
        Compute Reciprocal Rank Fusion score.

        RRF formula: 1 / (k + rank)
        Standard k=60 (Elastic/OpenSearch default)

        Args:
            rank: 1-indexed rank position
            k: Smoothing constant

        Returns:
            RRF contribution score
        """
        return 1.0 / (k + rank)

    def merge_dual_and_kg_results(
        self,
        dual_contexts: List[Dict[str, Any]],
        kg_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge DualLevel and KG results using RRF when tree is not available.

        Used for ablation configs like no_tree, dual_only where tree is disabled
        but we still want intelligent fusion between dual and KG sources.

        Args:
            dual_contexts: Contexts from DualLevel retriever
            kg_results: Results from KG retrieval

        Returns:
            Merged and deduplicated results sorted by RRF score
        """
        # Build article_id -> RRF score map
        rrf_scores: Dict[str, float] = {}
        article_data: Dict[str, Dict] = {}

        # Source weights (dual is primary when tree unavailable)
        DUAL_WEIGHT = 1.0
        KG_WEIGHT = 0.8

        # Process DUAL results
        for rank, ctx in enumerate(dual_contexts, start=1):
            article_id = ctx.get("metadata", {}).get("article_id", ctx.get("id", f"dual-{rank}"))
            rrf_contrib = DUAL_WEIGHT * self.compute_rrf_score(rank)
            rrf_scores[article_id] = rrf_scores.get(article_id, 0) + rrf_contrib

            if article_id not in article_data:
                article_data[article_id] = {
                    "text": ctx.get("text", ""),
                    "metadata": ctx.get("metadata", {}),
                }

        # Process KG results
        for rank, kg_ctx in enumerate(kg_results, start=1):
            kg_id = kg_ctx.get("id", f"kg-{rank}")
            rrf_contrib = KG_WEIGHT * self.compute_rrf_score(rank)
            rrf_scores[kg_id] = rrf_scores.get(kg_id, 0) + rrf_contrib

            if kg_id not in article_data:
                article_data[kg_id] = {
                    "text": kg_ctx.get("text", ""),
                    "metadata": {
                        "source": "kg",
                        **kg_ctx.get("metadata", {}),
                    },
                }

        # Apply article penalties
        self._apply_penalties(rrf_scores)

        # Build merged list sorted by RRF score
        merged = []
        for article_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if article_id in article_data:
                data = article_data[article_id]
                if data["text"]:
                    merged.append({
                        "id": article_id,
                        "text": data["text"],
                        "metadata": data["metadata"],
                        "score": rrf_score,
                    })

        # Deduplication
        return self._deduplicate_results(merged)

    def _merge_tree_and_kg(
        self,
        tree_result: Any,
        kg_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Simple merge when dual is not available."""
        # Dynamic weights based on tree confidence
        tree_confidence = tree_result.confidence if tree_result else 0.0
        if tree_confidence >= 0.7:
            TREE_WEIGHT = 0.7
            KG_WEIGHT = 0.3
        elif tree_confidence >= 0.5:
            TREE_WEIGHT = 0.5
            KG_WEIGHT = 0.5
        else:
            TREE_WEIGHT = 0.3
            KG_WEIGHT = 0.7

        merged = []

        # Add tree results
        for i, context in enumerate(tree_result.contexts):
            source_id = None
            if i < len(tree_result.target_nodes):
                source_id = tree_result.target_nodes[i].node_id
            merged.append({
                "id": f"tree-{i}",
                "text": context,
                "metadata": {
                    "source": "tree",
                    "source_id": source_id,
                },
                "score": tree_result.confidence * TREE_WEIGHT,
            })

        # Add KG results
        for kg_ctx in kg_results:
            kg_ctx_copy = kg_ctx.copy()
            kg_ctx_copy["score"] = kg_ctx.get("score", 0.5) * KG_WEIGHT
            if "metadata" not in kg_ctx_copy:
                kg_ctx_copy["metadata"] = {}
            kg_ctx_copy["metadata"]["source"] = "kg"
            merged.append(kg_ctx_copy)

        # Sort by score
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Deduplicate
        return self._deduplicate_results(merged)

    def _deduplicate_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate results by text similarity."""
        seen_texts = set()
        deduplicated = []

        for item in results:
            text_key = item["text"][:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduplicated.append(item)

        return deduplicated

    def _get_article_text(self, article_id: str) -> str:
        """Get article text from database.

        Args:
            article_id: Full article ID (e.g., "59-2020-QH14:d5")

        Returns:
            Article content text or empty string
        """
        if not self.db:
            return ""

        try:
            # Use get_article_by_id which takes full ID
            article = self.db.get_article_by_id(article_id)
            if article:
                return article.content or ""
        except Exception:
            pass

        return ""


def load_penalty_config_from_domains(
    domains_dir: str = "config/domains",
) -> Dict[str, Dict[int, float]]:
    """Load article penalty configs from all domain YAML files.

    Reads each domain YAML's `article_penalties` section and builds
    a flat {doc_id: {article_num: multiplier}} lookup dict.

    Args:
        domains_dir: Path to directory containing domain YAML files

    Returns:
        Penalty config dict for SemanticBridge
    """
    import yaml
    from pathlib import Path

    penalty_config: Dict[str, Dict[int, float]] = {}
    domains_path = Path(domains_dir)

    if not domains_path.exists():
        return penalty_config

    for yaml_file in domains_path.glob("*.yaml"):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                domain = yaml.safe_load(f) or {}
        except Exception:
            continue

        article_penalties = domain.get("article_penalties", {})
        if not article_penalties:
            continue

        # doc_id from filename (e.g., "59-2020-QH14.yaml" -> "59-2020-QH14")
        doc_id = yaml_file.stem

        doc_penalties: Dict[int, float] = {}
        for category_name, category_data in article_penalties.items():
            if not isinstance(category_data, dict):
                continue
            multiplier = category_data.get("multiplier", 1.0)
            articles = category_data.get("articles", [])
            for article_num in articles:
                doc_penalties[int(article_num)] = multiplier

        if doc_penalties:
            penalty_config[doc_id] = doc_penalties
            logger.info(
                f"Loaded {len(doc_penalties)} article penalties for {doc_id}"
            )

    return penalty_config


def create_semantic_bridge(
    kg: Dict[str, Any],
    db: Optional[Any] = None,
    penalty_config: Optional[Dict[str, Dict[int, float]]] = None,
) -> SemanticBridge:
    """Factory function to create SemanticBridge."""
    return SemanticBridge(kg=kg, db=db, penalty_config=penalty_config)
