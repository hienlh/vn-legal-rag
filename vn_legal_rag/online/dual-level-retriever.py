"""
Dual-Level Retriever

Combines low-level (entity-specific) and high-level (theme/concept) retrieval
for comprehensive legal document search.

Features:
- Low-level: PPR + BFS + semantic search + keyphrase matching
- High-level: Theme matching + concept hierarchy traversal
- 6-component scoring: keyphrase, semantic, PPR, concept, theme, hierarchy
- Configurable score fusion weights
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set
import numpy as np

# Vietnamese stopwords
VN_STOPWORDS = {
    'là', 'gì', 'của', 'trong', 'và', 'các', 'có', 'được', 'để', 'cho',
    'về', 'với', 'khi', 'này', 'theo', 'hoặc', 'thì', 'như', 'nếu', 'tại',
    'một', 'những', 'bởi', 'mà', 'đã', 'đang', 'sẽ', 'phải', 'không',
}


@dataclass
class LowLevelResult:
    """Result from low-level (entity-specific) retrieval."""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    articles: Dict[str, float] = field(default_factory=dict)
    concept_scores: Dict[str, float] = field(default_factory=dict)
    semantic_scores: Dict[str, float] = field(default_factory=dict)
    ppr_scores: Dict[str, float] = field(default_factory=dict)
    keyphrase_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class HighLevelResult:
    """Result from high-level (theme/concept) retrieval."""
    themes: List[Dict[str, Any]] = field(default_factory=list)
    articles: Dict[str, float] = field(default_factory=dict)
    theme_scores: Dict[str, float] = field(default_factory=dict)
    concept_hierarchy_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class DualLevelResult:
    """Combined result from dual-level retrieval."""
    articles: List[Dict[str, Any]] = field(default_factory=list)
    final_scores: Dict[str, float] = field(default_factory=dict)
    low_level: Optional[LowLevelResult] = None
    high_level: Optional[HighLevelResult] = None
    mode: str = "dual"
    score_components: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class DualLevelConfig:
    """Configuration for dual-level retrieval."""
    # Score fusion weights (optimized - must sum to 1.0)
    concept_weight: float = 0.20
    semantic_weight: float = 0.20
    ppr_weight: float = 0.25
    keyphrase_weight: float = 0.05
    theme_weight: float = 0.15
    hierarchy_weight: float = 0.15

    # Retrieval settings
    max_low_level_results: int = 50
    max_high_level_results: int = 30
    max_final_results: int = 30
    min_score_threshold: float = 0.1

    # Ablation flags
    enable_ppr: bool = True
    enable_keyphrase: bool = True
    enable_semantic: bool = True
    enable_concept: bool = True
    enable_theme: bool = True
    enable_hierarchy: bool = True
    enable_low_level: bool = True
    enable_high_level: bool = True

    def __post_init__(self):
        total = (
            self.concept_weight + self.semantic_weight + self.ppr_weight +
            self.keyphrase_weight + self.theme_weight + self.hierarchy_weight
        )
        assert 0.99 <= total <= 1.01, f"Weights must sum to 1.0, got {total}"


class DualLevelRetriever:
    """
    Dual-level retriever combining low-level and high-level retrieval.

    Low-level: Entity matching, semantic search, PPR, keyphrases
    High-level: Theme matching, concept hierarchy traversal
    """

    def __init__(
        self,
        kg: Dict[str, Any],
        theme_index: Optional[Any] = None,
        ontology: Optional[Any] = None,
        embedding_gen: Optional[Any] = None,
        config: Optional[DualLevelConfig] = None,
        semantic_matcher: Optional[Any] = None,
        ppr: Optional[Any] = None,
        concept_matcher: Optional[Any] = None,
    ):
        """
        Initialize dual-level retriever.

        Args:
            kg: Knowledge graph dict
            theme_index: ThemeIndex for high-level retrieval
            ontology: LegalOntology for concept hierarchy
            embedding_gen: Embedding generator
            config: Retrieval configuration
            semantic_matcher: Optional SemanticMatcher
            ppr: Optional PersonalizedPageRank
            concept_matcher: Optional ConceptMatcher
        """
        self.kg = kg
        self.theme_index = theme_index
        self.ontology = ontology
        self.embedding_gen = embedding_gen
        self.config = config or DualLevelConfig()

        # External components
        self.semantic_matcher = semantic_matcher
        self.ppr = ppr
        self.concept_matcher = concept_matcher

        # Build entity index
        self._entity_index = {
            e.get("id", ""): e
            for e in kg.get("entities", [])
        }

        # Build article mappings
        self._article_entities: Dict[str, List[str]] = {}
        self._entity_to_article: Dict[str, str] = {}
        self._short_to_full_articles: Dict[str, List[str]] = {}

        for e in kg.get("entities", []):
            metadata = e.get("metadata", {})
            eid = e.get("id", "")

            sources = metadata.get("source_ids", [])
            if not sources and metadata.get("source_id"):
                sources = [metadata["source_id"]]

            for source in sources:
                if source not in self._article_entities:
                    self._article_entities[source] = []
                self._article_entities[source].append(eid)

                if eid not in self._entity_to_article:
                    self._entity_to_article[eid] = source

                # Extract short article ID
                match = re.search(r":d(\d+)$", source)
                if match:
                    short_id = match.group(1)
                    if short_id not in self._short_to_full_articles:
                        self._short_to_full_articles[short_id] = []
                    if source not in self._short_to_full_articles[short_id]:
                        self._short_to_full_articles[short_id].append(source)

    def retrieve(
        self,
        query: str,
        intents: Optional[List[Dict]] = None,
        mode: Literal["low", "high", "dual"] = "dual",
        max_results: int = 10,
    ) -> DualLevelResult:
        """
        Retrieve relevant articles using dual-level approach.

        Args:
            query: Search query
            intents: Optional intent information
            mode: Retrieval mode (low, high, dual)
            max_results: Maximum results to return

        Returns:
            DualLevelResult with ranked articles
        """
        intents = intents or []
        result = DualLevelResult(mode=mode)

        # Low-level retrieval
        if mode in ("low", "dual") and self.config.enable_low_level:
            result.low_level = self._low_level_retrieve(query, intents)

        # High-level retrieval
        if mode in ("high", "dual") and self.config.enable_high_level:
            result.high_level = self._high_level_retrieve(query)

        # Fuse scores
        result.final_scores = self._fuse_scores(
            result.low_level,
            result.high_level,
            mode,
        )

        # Build ranked article list
        result.articles = self._build_ranked_articles(
            result.final_scores,
            max_results,
        )

        return result

    def _low_level_retrieve(
        self,
        query: str,
        intents: List[Dict],
    ) -> LowLevelResult:
        """Perform low-level (entity-specific) retrieval."""
        result = LowLevelResult()

        query_lower = query.lower()
        query_terms = set(query_lower.split()) - VN_STOPWORDS

        if not query_terms:
            return result

        # 1. Keyphrase matching
        if self.config.enable_keyphrase:
            self._keyphrase_match(query_lower, query_terms, result)

        # 2. Semantic search
        if self.semantic_matcher and self.config.enable_semantic:
            try:
                semantic_scores = self.semantic_matcher.find_similar_articles(query)
                for short_id, score in semantic_scores.items():
                    full_ids = self._short_to_full_articles.get(short_id, [])
                    for full_id in full_ids:
                        current = result.semantic_scores.get(full_id, 0)
                        result.semantic_scores[full_id] = max(current, score)
            except Exception:
                pass

        # 3. PPR
        if self.ppr and self.embedding_gen and self.config.enable_ppr:
            try:
                seeds = {}
                for eid, score in result.keyphrase_scores.items():
                    if score >= 0.3:
                        seeds[eid] = score

                if seeds:
                    query_emb = np.array(
                        self.embedding_gen.generate_embeddings([query])[0]
                    )

                    ppr_result = self.ppr.compute(
                        seeds=seeds,
                        intents=intents or [{"intent": "general", "confidence": 0.5}],
                        query_embedding=query_emb,
                    )

                    ppr_article_scores = self.ppr.map_to_articles(ppr_result.scores)

                    for short_id, score in ppr_article_scores.items():
                        full_ids = self._short_to_full_articles.get(short_id, [])
                        for full_id in full_ids:
                            current = result.ppr_scores.get(full_id, 0)
                            result.ppr_scores[full_id] = max(current, score)
            except Exception:
                pass

        # 5. Combine scores
        self._combine_low_level_scores(result)

        return result

    def _keyphrase_match(
        self,
        query_lower: str,
        query_terms: Set[str],
        result: LowLevelResult,
    ) -> None:
        """Perform keyphrase matching against entities."""
        query_words = query_lower.split()
        query_2grams = {' '.join(query_words[i:i+2]) for i in range(len(query_words)-1)}
        query_3grams = {' '.join(query_words[i:i+3]) for i in range(len(query_words)-2)}

        for eid, entity in self._entity_index.items():
            name = entity.get("name", entity.get("text", "")).lower()
            name_terms = set(name.split())

            overlap = len(query_terms & name_terms)
            score = 0.0

            if overlap > 0:
                score = overlap / len(query_terms)
                if overlap >= 2:
                    score *= 1.5

            # Compound phrase match
            for phrase in query_2grams:
                if phrase in name:
                    score = max(score, 0.5)
                    score *= 1.5
                    break

            for phrase in query_3grams:
                if phrase in name:
                    score = max(score, 0.7)
                    score *= 1.5
                    break

            # Exact match
            if name in query_lower:
                score = max(score, 0.8)
                score *= 1.5
            elif query_lower in name:
                score = max(score, 0.6)
                score *= 1.2

            if score > 0:
                result.keyphrase_scores[eid] = min(1.0, score)
                result.entities.append(entity)

    def _combine_low_level_scores(self, result: LowLevelResult) -> None:
        """Combine keyphrase, semantic, PPR, concept scores into article scores."""
        all_articles: Set[str] = set()

        article_kp_counts: Dict[str, int] = {}
        article_kp_max: Dict[str, float] = {}
        article_kp_sum: Dict[str, float] = {}

        for eid, kp_score in result.keyphrase_scores.items():
            article_id = self._entity_to_article.get(eid, "")
            if article_id:
                all_articles.add(article_id)
                article_kp_counts[article_id] = article_kp_counts.get(article_id, 0) + 1
                article_kp_max[article_id] = max(article_kp_max.get(article_id, 0), kp_score)
                article_kp_sum[article_id] = article_kp_sum.get(article_id, 0) + kp_score

        all_articles.update(result.semantic_scores.keys())
        all_articles.update(result.ppr_scores.keys())
        all_articles.update(result.concept_scores.keys())

        config = self.config

        for article_id in all_articles:
            kp_max = article_kp_max.get(article_id, 0)
            kp_sum = article_kp_sum.get(article_id, 0)
            kp_count = article_kp_counts.get(article_id, 0)

            if kp_count > 0:
                import math
                keyphrase_score = kp_max + (math.log(kp_count + 1) * kp_sum / kp_count) * 0.3
            else:
                keyphrase_score = 0.0

            semantic_score = result.semantic_scores.get(article_id, 0)
            ppr_score = result.ppr_scores.get(article_id, 0)
            concept_score = result.concept_scores.get(article_id, 0)

            combined = (
                config.keyphrase_weight * keyphrase_score +
                config.semantic_weight * semantic_score +
                config.ppr_weight * ppr_score +
                config.concept_weight * concept_score
            )

            result.articles[article_id] = combined

        # Normalize
        if result.articles:
            max_score = max(result.articles.values())
            if max_score > 0:
                result.articles = {k: v/max_score for k, v in result.articles.items()}

    def _high_level_retrieve(self, query: str) -> HighLevelResult:
        """Perform high-level (theme/concept) retrieval."""
        result = HighLevelResult()

        # Theme search
        if self.theme_index and self.embedding_gen and self.config.enable_theme:
            try:
                query_emb = self.embedding_gen.embed(query)
                theme_results = self.theme_index.search(query_emb, k=10)

                for theme, score in theme_results:
                    result.themes.append(theme.to_dict())

                    for eid in theme.source_entities:
                        entity = self._entity_index.get(eid, {})
                        metadata = entity.get("metadata", {})
                        sources = metadata.get("source_ids", [])
                        if not sources and metadata.get("source_id"):
                            sources = [metadata["source_id"]]
                        for source in sources[:1]:
                            result.theme_scores[source] = max(
                                result.theme_scores.get(source, 0),
                                score
                            )
            except Exception:
                pass

        result.articles = {**result.theme_scores}

        return result

    def _fuse_scores(
        self,
        low_level: Optional[LowLevelResult],
        high_level: Optional[HighLevelResult],
        mode: str,
    ) -> Dict[str, float]:
        """Fuse scores from low and high level retrieval."""
        all_articles: Set[str] = set()
        if low_level:
            all_articles.update(low_level.articles.keys())
        if high_level:
            all_articles.update(high_level.articles.keys())

        final_scores = {}
        config = self.config

        low_weight_sum = (
            config.keyphrase_weight + config.semantic_weight +
            config.ppr_weight + config.concept_weight
        )
        high_weight_sum = config.theme_weight + config.hierarchy_weight

        for article_id in all_articles:
            score = 0.0

            if low_level and mode in ("low", "dual"):
                low_score = low_level.articles.get(article_id, 0)
                if mode == "dual":
                    score += low_score * low_weight_sum
                else:
                    score += low_score

            if high_level and mode in ("high", "dual"):
                theme_score = high_level.theme_scores.get(article_id, 0)
                hierarchy_score = high_level.concept_hierarchy_scores.get(article_id, 0)

                score += config.theme_weight * theme_score
                score += config.hierarchy_weight * hierarchy_score

            if score >= config.min_score_threshold:
                final_scores[article_id] = score

        return final_scores

    def _build_ranked_articles(
        self,
        scores: Dict[str, float],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Build ranked list of articles with metadata."""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        articles = []
        for article_id, score in sorted_items[:max_results]:
            articles.append({
                "article_id": article_id,
                "score": score,
                "entities": self._article_entities.get(article_id, []),
            })

        return articles
