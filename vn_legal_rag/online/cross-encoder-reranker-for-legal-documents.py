"""
Cross-Encoder Reranker for Vietnamese Legal Documents

Uses Vietnamese-specific phobert model for semantic reranking:
- Scores query-document pairs for relevance
- Reranks top candidates for higher precision
- Gracefully degrades if model unavailable

Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base (~400MB)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional import - graceful degradation if not installed
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None


# Default Vietnamese phobert model for legal documents
DEFAULT_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"

# Alternative models (fallback options)
FALLBACK_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


@dataclass
class RerankResult:
    """Result of cross-encoder reranking."""
    reranked_items: List[Dict[str, Any]] = field(default_factory=list)
    candidates_processed: int = 0
    top_k_returned: int = 0
    model_used: Optional[str] = None
    fallback_used: bool = False


class CrossEncoderReranker:
    """
    Reranks retrieval candidates using cross-encoder model.

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> if reranker.is_available():
        ...     result = reranker.rerank(
        ...         query="vốn điều lệ công ty TNHH",
        ...         candidates=[
        ...             {"id": "art1", "content": "Vốn điều lệ là..."},
        ...             {"id": "art2", "content": "Công ty TNHH có..."},
        ...         ],
        ...         top_k=5
        ...     )
        ...     print(result.reranked_items)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name/path
            device: Device for inference ("cuda", "cpu", or None for auto)
            max_length: Maximum sequence length
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Optional[Any] = None
        self._model_loaded = False
        self._actual_model_name: Optional[str] = None

    def is_available(self) -> bool:
        """Check if cross-encoder is available (library installed)."""
        return CROSS_ENCODER_AVAILABLE

    def _load_model(self) -> bool:
        """Lazy load model on first use."""
        if self._model_loaded:
            return self._model is not None

        if not CROSS_ENCODER_AVAILABLE:
            self._model_loaded = True
            return False

        # Try primary model first, then fallbacks
        models_to_try = [self.model_name] + FALLBACK_MODELS

        for model_name in models_to_try:
            try:
                self._model = CrossEncoder(
                    model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
                self._actual_model_name = model_name
                self._model_loaded = True
                return True
            except Exception:
                continue

        self._model_loaded = True
        return False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        content_key: str = "content",
        score_key: str = "rerank_score",
    ) -> RerankResult:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: User query
            candidates: List of candidate dicts with content
            top_k: Number of top results to return
            content_key: Key for document content in candidate dicts
            score_key: Key to store rerank score in results

        Returns:
            RerankResult with reranked items
        """
        result = RerankResult(
            candidates_processed=len(candidates),
            top_k_returned=min(top_k, len(candidates)),
        )

        if not candidates:
            return result

        # Load model if needed
        if not self._load_model():
            # Model not available - return original order
            result.reranked_items = candidates[:top_k]
            return result

        result.model_used = self._actual_model_name
        result.fallback_used = self._actual_model_name != self.model_name

        # Build query-document pairs
        pairs: List[Tuple[str, str]] = []
        for candidate in candidates:
            content = candidate.get(content_key, "")
            if isinstance(content, str) and content.strip():
                pairs.append((query, content))
            else:
                pairs.append((query, ""))

        # Score pairs
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception:
            # Scoring failed - return original order
            result.reranked_items = candidates[:top_k]
            return result

        # Combine with candidates and sort
        scored_candidates = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            candidate_copy = candidate.copy()
            candidate_copy[score_key] = float(score)
            candidate_copy["_original_rank"] = i
            scored_candidates.append(candidate_copy)

        # Sort by rerank score descending
        scored_candidates.sort(key=lambda x: x[score_key], reverse=True)

        result.reranked_items = scored_candidates[:top_k]
        result.top_k_returned = len(result.reranked_items)

        return result

    def rerank_with_scores(
        self,
        query: str,
        article_scores: Dict[str, float],
        article_contents: Dict[str, str],
        top_k: int = 10,
    ) -> Tuple[Dict[str, float], RerankResult]:
        """
        Rerank articles and return updated scores.

        Args:
            query: User query
            article_scores: Article ID -> original score
            article_contents: Article ID -> content text
            top_k: Number of top results to return

        Returns:
            Tuple of (reranked_scores, RerankResult)
        """
        # Build candidates list
        candidates = []
        for article_id, score in article_scores.items():
            content = article_contents.get(article_id, "")
            candidates.append({
                "id": article_id,
                "content": content,
                "original_score": score,
            })

        # Sort by original score to get top candidates
        candidates.sort(key=lambda x: x["original_score"], reverse=True)

        # Rerank top candidates (limit for efficiency)
        max_candidates = min(len(candidates), top_k * 3)  # 3x for headroom
        result = self.rerank(
            query=query,
            candidates=candidates[:max_candidates],
            top_k=top_k,
            content_key="content",
        )

        # Build new score dict
        reranked_scores: Dict[str, float] = {}
        for item in result.reranked_items:
            article_id = item["id"]
            # Combine original score with rerank score
            original = item.get("original_score", 0.0)
            rerank = item.get("rerank_score", 0.0)
            # Weighted combination: 60% rerank + 40% original
            reranked_scores[article_id] = 0.6 * rerank + 0.4 * original

        return reranked_scores, result
