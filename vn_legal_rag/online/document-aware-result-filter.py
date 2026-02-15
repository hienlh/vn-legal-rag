"""
Document-Aware Result Filter for Vietnamese Legal RAG

Filters and re-ranks retrieval results based on document references detected in query:
- Detects specific legal documents mentioned (Luật, Nghị định, Thông tư, etc.)
- Supports 3 filter modes: boost, filter, rerank
- Extracts document identifiers (numbers, years, names)

Example: "theo Nghị định 01/2021/NĐ-CP" → boost articles from that decree
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Document Detection Patterns
# ============================================================================

# Pattern: Document Type + Number + Year + Issuer
# Examples: "Luật 59/2020/QH14", "Nghị định 01/2021/NĐ-CP", "Thông tư 78/2021/TT-BTC"
DOC_REFERENCE_PATTERN = re.compile(
    r"(?P<type>luật|nghị\s*định|thông\s*tư|quyết\s*định|nghị\s*quyết|"
    r"nđ|tt|qđ|nq|luat|nghi\s*dinh|thong\s*tu|quyet\s*dinh|nghi\s*quyet)"
    r"[\s\-]*"
    r"(?P<number>\d+)"
    r"(?:[\/\-](?P<year>\d{4}))?"
    r"(?:[\/\-](?P<issuer>[A-ZĐa-zđ\-]+))?",
    re.IGNORECASE | re.UNICODE
)

# Document type normalization
DOC_TYPE_NORMALIZE: Dict[str, str] = {
    "luật": "luat",
    "luat": "luat",
    "nghị định": "nghi_dinh",
    "nghi dinh": "nghi_dinh",
    "nghịđịnh": "nghi_dinh",
    "nđ": "nghi_dinh",
    "nd": "nghi_dinh",
    "thông tư": "thong_tu",
    "thong tu": "thong_tu",
    "thôngtư": "thong_tu",
    "tt": "thong_tu",
    "quyết định": "quyet_dinh",
    "quyet dinh": "quyet_dinh",
    "quyếtđịnh": "quyet_dinh",
    "qđ": "quyet_dinh",
    "qd": "quyet_dinh",
    "nghị quyết": "nghi_quyet",
    "nghi quyet": "nghi_quyet",
    "nghịquyết": "nghi_quyet",
    "nq": "nghi_quyet",
}

# Vietnamese document keywords for general matching
DOC_KEYWORDS: Dict[str, List[str]] = {
    "luat": ["luật", "luat", "bộ luật", "bo luat"],
    "nghi_dinh": ["nghị định", "nghi dinh", "nđ", "nd"],
    "thong_tu": ["thông tư", "thong tu", "tt"],
    "quyet_dinh": ["quyết định", "quyet dinh", "qđ", "qd"],
    "nghi_quyet": ["nghị quyết", "nghi quyet", "nq"],
}


@dataclass
class DocumentHint:
    """Detected document reference from query."""
    doc_id: str  # Normalized ID like "nghi_dinh_01_2021"
    doc_type: str  # luat, nghi_dinh, thong_tu, quyet_dinh, nghi_quyet
    doc_number: Optional[str] = None
    doc_year: Optional[str] = None
    doc_issuer: Optional[str] = None
    confidence: float = 1.0
    matched_text: str = ""

    @property
    def full_reference(self) -> str:
        """Generate full document reference string."""
        parts = [self.doc_type]
        if self.doc_number:
            parts.append(self.doc_number)
        if self.doc_year:
            parts.append(self.doc_year)
        if self.doc_issuer:
            parts.append(self.doc_issuer)
        return "_".join(parts)


@dataclass
class FilteredResult:
    """Result of document-aware filtering."""
    filtered_scores: Dict[str, float]
    document_hints: List[DocumentHint] = field(default_factory=list)
    boosted_count: int = 0
    filtered_count: int = 0
    original_count: int = 0


class DocumentFilter:
    """
    Filters retrieval results based on document references in query.

    Example:
        >>> filter = DocumentFilter()
        >>> hints = filter.detect_documents("theo Nghị định 01/2021/NĐ-CP")
        >>> print(hints[0].doc_type)
        'nghi_dinh'
    """

    def __init__(
        self,
        boost_factor: float = 2.0,
        penalty_factor: float = 0.5,
        min_confidence: float = 0.5,
    ):
        """
        Initialize document filter.

        Args:
            boost_factor: Score multiplier for matching documents (mode: boost, rerank)
            penalty_factor: Score multiplier for non-matching documents (mode: rerank)
            min_confidence: Minimum confidence for document hints
        """
        self.boost_factor = boost_factor
        self.penalty_factor = penalty_factor
        self.min_confidence = min_confidence

    def detect_documents(self, query: str) -> List[DocumentHint]:
        """
        Detect document references in query.

        Args:
            query: User query string

        Returns:
            List of detected document hints
        """
        hints: List[DocumentHint] = []
        query_lower = query.lower()

        # Pattern-based detection (specific references)
        for match in DOC_REFERENCE_PATTERN.finditer(query_lower):
            doc_type_raw = match.group("type").strip()
            doc_type = self._normalize_doc_type(doc_type_raw)

            if not doc_type:
                continue

            doc_number = match.group("number")
            doc_year = match.group("year")
            doc_issuer = match.group("issuer")

            # Build doc_id
            doc_id_parts = [doc_type]
            if doc_number:
                doc_id_parts.append(doc_number)
            if doc_year:
                doc_id_parts.append(doc_year)

            hint = DocumentHint(
                doc_id="_".join(doc_id_parts),
                doc_type=doc_type,
                doc_number=doc_number,
                doc_year=doc_year,
                doc_issuer=doc_issuer.upper() if doc_issuer else None,
                confidence=1.0,  # Exact pattern match
                matched_text=match.group(0),
            )
            hints.append(hint)

        # Keyword-based detection (general references) - lower confidence
        if not hints:
            for doc_type, keywords in DOC_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        hint = DocumentHint(
                            doc_id=doc_type,
                            doc_type=doc_type,
                            confidence=0.6,  # Keyword match only
                            matched_text=keyword,
                        )
                        hints.append(hint)
                        break  # One hint per doc_type

        # Filter by min confidence
        hints = [h for h in hints if h.confidence >= self.min_confidence]

        # Deduplicate by doc_id
        seen_ids: set = set()
        unique_hints: List[DocumentHint] = []
        for hint in hints:
            if hint.doc_id not in seen_ids:
                seen_ids.add(hint.doc_id)
                unique_hints.append(hint)

        return unique_hints

    def _normalize_doc_type(self, doc_type_raw: str) -> Optional[str]:
        """Normalize document type to standard form."""
        doc_type_clean = doc_type_raw.lower().strip()
        doc_type_clean = re.sub(r"\s+", " ", doc_type_clean)
        return DOC_TYPE_NORMALIZE.get(doc_type_clean)

    def filter_articles(
        self,
        scores: Dict[str, float],
        hints: List[DocumentHint],
        mode: str = "boost",
        article_metadata: Optional[Dict[str, Dict]] = None,
    ) -> FilteredResult:
        """
        Filter/re-rank article scores based on document hints.

        Args:
            scores: Article ID -> score mapping
            hints: Detected document hints from query
            mode: Filter mode - "boost", "filter", "rerank"
            article_metadata: Optional article_id -> metadata mapping
                              Expected keys: "doc_type", "doc_number", "doc_year"

        Returns:
            FilteredResult with adjusted scores
        """
        result = FilteredResult(
            filtered_scores={},
            document_hints=hints,
            original_count=len(scores),
        )

        if not hints:
            # No document hints - return original scores
            result.filtered_scores = scores.copy()
            return result

        for article_id, score in scores.items():
            matched = self._article_matches_hints(article_id, hints, article_metadata)

            if mode == "filter":
                # Only keep matching articles
                if matched:
                    result.filtered_scores[article_id] = score
                    result.filtered_count += 1
            elif mode == "boost":
                # Boost matching articles
                if matched:
                    result.filtered_scores[article_id] = score * self.boost_factor
                    result.boosted_count += 1
                else:
                    result.filtered_scores[article_id] = score
            elif mode == "rerank":
                # Boost matching, penalize non-matching
                if matched:
                    result.filtered_scores[article_id] = score * self.boost_factor
                    result.boosted_count += 1
                else:
                    result.filtered_scores[article_id] = score * self.penalty_factor
            else:
                # Unknown mode - keep original
                result.filtered_scores[article_id] = score

        return result

    def _article_matches_hints(
        self,
        article_id: str,
        hints: List[DocumentHint],
        article_metadata: Optional[Dict[str, Dict]] = None,
    ) -> bool:
        """Check if article matches any document hint."""
        article_id_lower = article_id.lower()

        for hint in hints:
            # Method 1: Check article_id contains hint patterns
            if hint.doc_type in article_id_lower:
                if hint.doc_number:
                    if hint.doc_number in article_id_lower:
                        if hint.doc_year:
                            if hint.doc_year in article_id_lower:
                                return True
                        else:
                            return True
                else:
                    return True

            # Method 2: Use metadata if available
            if article_metadata and article_id in article_metadata:
                meta = article_metadata[article_id]
                meta_type = meta.get("doc_type", "").lower()
                meta_number = str(meta.get("doc_number", ""))
                meta_year = str(meta.get("doc_year", ""))

                if hint.doc_type == meta_type:
                    if hint.doc_number:
                        if hint.doc_number == meta_number:
                            if hint.doc_year:
                                if hint.doc_year == meta_year:
                                    return True
                            else:
                                return True
                    else:
                        return True

        return False

    def apply_to_retrieval(
        self,
        query: str,
        scores: Dict[str, float],
        mode: str = "boost",
        article_metadata: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[Dict[str, float], List[DocumentHint]]:
        """
        Convenience method: detect documents and filter in one call.

        Args:
            query: User query
            scores: Article scores from retrieval
            mode: Filter mode
            article_metadata: Optional article metadata

        Returns:
            Tuple of (filtered_scores, document_hints)
        """
        hints = self.detect_documents(query)
        result = self.filter_articles(scores, hints, mode, article_metadata)
        return result.filtered_scores, hints
