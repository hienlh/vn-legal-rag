"""
Document-Level Filtering for Legal Retrieval

Detects which legal documents a query refers to and filters/boosts results accordingly.
This helps resolve ambiguity when short article IDs (like "114") appear in multiple documents.

Vietnamese Legal Document Types:
- Luật (QH): Laws passed by National Assembly (e.g., 59-2020-QH14 = Luật Doanh nghiệp 2020)
- Nghị định (ND): Government decrees (e.g., 01-2021-ND)
- Thông tư (TT): Ministry circulars
- Quyết định (QD): Decisions

Usage:
    >>> from vn_legal_rag.offline.document_filter import DocumentFilter
    >>> filter = DocumentFilter(kg)
    >>> hints = filter.detect_documents("Điều 114 Luật Doanh nghiệp 2020")
    >>> # hints = [DocumentHint(doc_id="59-2020-QH14", confidence=0.9, ...)]
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..utils.simple_logger import get_logger


@dataclass
class DocumentHint:
    """Hint about which document a query refers to."""

    doc_id: str
    doc_type: str  # "luat", "nghi_dinh", "thong_tu", "quyet_dinh"
    confidence: float  # 0.0 to 1.0
    matched_keywords: List[str] = field(default_factory=list)
    matched_pattern: Optional[str] = None


@dataclass
class FilteredResult:
    """Result of document filtering."""

    original_scores: Dict[str, float]
    filtered_scores: Dict[str, float]
    document_hints: List[DocumentHint]
    boost_applied: bool


# Document keyword mappings
DOCUMENT_KEYWORDS = {
    # Luật Doanh nghiệp 2020
    "59-2020-QH14": {
        "keywords": [
            "luật doanh nghiệp",
            "luật doanh nghiệp 2020",
            "doanh nghiệp",
            "công ty",
            "cổ đông",
            "cổ phần",
            "tnhh",
            "trách nhiệm hữu hạn",
            "phá sản",
            "vốn điều lệ",
            "thành viên hội đồng",
            "giám đốc",
            "đại hội đồng cổ đông",
            "ban kiểm soát",
            "doanh nghiệp tư nhân",
            "thành lập doanh nghiệp",
            "điều kiện thành lập",
            "công ty cổ phần",
            "công ty tnhh",
            "hội đồng quản trị",
            "tổng giám đốc",
            "chủ tịch hội đồng",
            "quyền cổ đông",
            "nghĩa vụ cổ đông",
            "giải thể doanh nghiệp",
            "trường hợp giải thể",
        ],
        "patterns": [
            r"luật\s+doanh\s+nghiệp",
            r"59[/-]2020[/-]qh14",
            r"luật\s+số\s+59",
            r"doanh\s+nghiệp\s+tư\s+nhân",
            r"công\s+ty\s+cổ\s+phần",
            r"cổ\s+đông",
            r"giám\s+đốc",
        ],
        "type": "luat",
        "base_weight": 1.0,
    },
    # Nghị định 01/2021 - Đăng ký doanh nghiệp
    "01-2021-ND": {
        "keywords": [
            "nghị định 01",
            "nghị định đăng ký",
            "đăng ký doanh nghiệp",
            "đăng ký kinh doanh",
            "hồ sơ đăng ký",
            "thủ tục đăng ký",
            "giấy chứng nhận đăng ký",
            "mã số doanh nghiệp",
            "cơ quan đăng ký kinh doanh",
            "hộ kinh doanh",
        ],
        "patterns": [
            r"nghị\s+định\s+01[/-]2021",
            r"01[/-]2021[/-]n[đd]",
            r"đăng\s+ký\s+doanh\s+nghiệp",
        ],
        "type": "nghi_dinh",
        "base_weight": 0.9,
    },
    # Nghị định 168/2025
    "168-2025-ND": {
        "keywords": [
            "nghị định 168",
            "168/2025",
        ],
        "patterns": [
            r"168[/-]2025[/-]n[đd]",
            r"nghị\s+định\s+168",
        ],
        "type": "nghi_dinh",
        "base_weight": 0.8,
    },
    # Nghị định 23/2022 - Giải thể
    "23-2022-ND": {
        "keywords": [
            "nghị định 23",
            "giải thể",
            "23/2022",
        ],
        "patterns": [
            r"23[/-]2022[/-]n[đd]",
            r"nghị\s+định\s+23",
        ],
        "type": "nghi_dinh",
        "base_weight": 0.8,
    },
    # Nghị định 47/2021 - Doanh nghiệp xã hội
    "47-2021-ND": {
        "keywords": [
            "doanh nghiệp xã hội",
            "nghị định 47",
        ],
        "patterns": [
            r"47[/-]2021[/-]n[đd]",
            r"doanh\s+nghiệp\s+xã\s+hội",
        ],
        "type": "nghi_dinh",
        "base_weight": 0.8,
    },
}

# Generic document type patterns
DOC_TYPE_PATTERNS = {
    "luat": [
        r"luật\s+(\w+)",
        r"luật\s+số\s+\d+",
    ],
    "nghi_dinh": [
        r"nghị\s+định\s+(\d+)",
        r"nd[/-]\d+",
    ],
    "thong_tu": [
        r"thông\s+tư\s+(\d+)",
        r"tt[/-]\d+",
    ],
}


class DocumentFilter:
    """
    Filter and boost articles based on detected document references in query.
    """

    def __init__(
        self,
        kg: Dict[str, Any],
        default_boost: float = 2.0,
        penalty_factor: float = 0.5,
    ):
        """
        Initialize document filter.

        Args:
            kg: Knowledge graph dict
            default_boost: Boost multiplier for matching documents
            penalty_factor: Penalty multiplier for non-matching documents
        """
        self.kg = kg
        self.default_boost = default_boost
        self.penalty_factor = penalty_factor

        self.logger = get_logger("document_filter")

        # Build document -> articles mapping
        self._doc_articles: Dict[str, Set[str]] = {}
        for e in kg.get("entities", []):
            doc_id = e.get("metadata", {}).get("document_id", "")
            source_id = e.get("metadata", {}).get("source_id", "")
            if doc_id and source_id:
                if doc_id not in self._doc_articles:
                    self._doc_articles[doc_id] = set()
                self._doc_articles[doc_id].add(source_id)

        # Compile patterns
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for doc_id, config in DOCUMENT_KEYWORDS.items():
            self._compiled_patterns[doc_id] = [
                re.compile(p, re.IGNORECASE) for p in config.get("patterns", [])
            ]

    def detect_documents(self, query: str) -> List[DocumentHint]:
        """
        Detect which documents a query likely refers to.

        Args:
            query: Search query

        Returns:
            List of DocumentHint with confidence scores
        """
        query_lower = query.lower()
        hints = []

        for doc_id, config in DOCUMENT_KEYWORDS.items():
            confidence = 0.0
            matched_keywords = []
            matched_pattern = None

            # Check patterns first (highest confidence)
            for pattern in self._compiled_patterns.get(doc_id, []):
                if pattern.search(query_lower):
                    confidence = max(confidence, 0.9)
                    matched_pattern = pattern.pattern
                    break

            # Check keywords
            for keyword in config.get("keywords", []):
                if keyword in query_lower:
                    matched_keywords.append(keyword)
                    kw_conf = min(0.8, 0.3 + len(keyword.split()) * 0.15)
                    confidence = max(confidence, kw_conf)

            if confidence > 0.2:
                hints.append(
                    DocumentHint(
                        doc_id=doc_id,
                        doc_type=config.get("type", "unknown"),
                        confidence=confidence * config.get("base_weight", 1.0),
                        matched_keywords=matched_keywords,
                        matched_pattern=matched_pattern,
                    )
                )

        hints.sort(key=lambda h: h.confidence, reverse=True)
        return hints

    def filter_articles(
        self,
        article_scores: Dict[str, float],
        hints: List[DocumentHint],
        mode: str = "boost",  # "boost", "filter", "rerank"
    ) -> FilteredResult:
        """
        Filter/boost article scores based on document hints.

        Args:
            article_scores: Original article scores (article_id -> score)
            hints: Document hints from detect_documents()
            mode:
                - "boost": Multiply matching docs by boost factor
                - "filter": Only keep articles from matching docs
                - "rerank": Boost matching + penalize non-matching

        Returns:
            FilteredResult with adjusted scores
        """
        if not hints:
            return FilteredResult(
                original_scores=article_scores,
                filtered_scores=article_scores.copy(),
                document_hints=[],
                boost_applied=False,
            )

        # Get articles from hinted documents
        hinted_articles: Set[str] = set()
        doc_confidence: Dict[str, float] = {}

        for hint in hints:
            doc_articles = self._doc_articles.get(hint.doc_id, set())
            hinted_articles.update(doc_articles)
            for article in doc_articles:
                doc_confidence[article] = max(
                    doc_confidence.get(article, 0), hint.confidence
                )

        filtered_scores = {}

        for article_id, score in article_scores.items():
            if mode == "filter":
                if article_id in hinted_articles:
                    filtered_scores[article_id] = score

            elif mode == "boost":
                if article_id in hinted_articles:
                    boost = self.default_boost * doc_confidence.get(article_id, 1.0)
                    filtered_scores[article_id] = score * boost
                else:
                    filtered_scores[article_id] = score

            elif mode == "rerank":
                if article_id in hinted_articles:
                    boost = self.default_boost * doc_confidence.get(article_id, 1.0)
                    filtered_scores[article_id] = score * boost
                else:
                    filtered_scores[article_id] = score * self.penalty_factor

        return FilteredResult(
            original_scores=article_scores,
            filtered_scores=filtered_scores,
            document_hints=hints,
            boost_applied=True,
        )

    def get_document_articles(self, doc_id: str) -> Set[str]:
        """Get all article IDs from a document."""
        return self._doc_articles.get(doc_id, set())

    def get_article_document(self, article_id: str) -> Optional[str]:
        """Get document ID for an article."""
        if ":" in article_id:
            return article_id.split(":")[0]
        return None
