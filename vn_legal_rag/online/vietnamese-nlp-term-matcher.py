"""
Vietnamese NLP Term Matcher for Ontology-Based Query Expansion.

Uses underthesea for Vietnamese word segmentation and text normalization
to match query terms with ontology classes.
"""

from typing import Dict, List, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)

# Graceful import for underthesea
try:
    from underthesea import word_tokenize, text_normalize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    logger.warning("underthesea not available, using basic Vietnamese matching")


class VietnameseNLPMatcher:
    """
    Vietnamese NLP-based ontology term matcher.

    Uses underthesea for word segmentation and matching Vietnamese
    queries to ontology class labels.
    """

    # Compound words specific to Vietnamese legal domain
    COMPOUND_MAPPINGS = {
        "công ty cổ phần": "JointStockCompany",
        "công_ty cổ_phần": "JointStockCompany",
        "công ty tnhh": "LimitedLiabilityCompany",
        "công_ty tnhh": "LimitedLiabilityCompany",
        "công ty trách nhiệm hữu hạn": "LimitedLiabilityCompany",
        "tnhh một thành viên": "SingleMemberLLC",
        "tnhh 1 thành viên": "SingleMemberLLC",
        "tnhh hai thành viên": "MultiMemberLLC",
        "tnhh 2 thành viên": "MultiMemberLLC",
        "doanh nghiệp tư nhân": "PrivateEnterprise",
        "công ty hợp danh": "Partnership",
        "hộ kinh doanh": "HouseholdBusiness",
        "nghị định": "Decree",
        "nghị_định": "Decree",
        "thông tư": "Circular",
        "thông_tư": "Circular",
        "quyết định": "Decision",
        "quyết_định": "Decision",
        "nghị quyết": "Resolution",
        "nghị_quyết": "Resolution",
        "vốn điều lệ": "Monetary",
        "vốn_điều_lệ": "Monetary",
        "người đại diện theo pháp luật": "LegalRepresentative",
        "người_đại_diện": "LegalRepresentative",
        "cổ đông": "Shareholder",
        "cổ_đông": "Shareholder",
        "thành viên": "Member",
        "thành_viên": "Member",
        "giám đốc": "Director",
        "giám_đốc": "Director",
        "hội đồng quản trị": "BoardMember",
        "hội_đồng_quản_trị": "BoardMember",
        "hđqt": "BoardMember",
    }

    def __init__(
        self,
        ontology_labels: Dict[str, str],
        use_compound_matching: bool = True,
    ):
        """
        Initialize the Vietnamese NLP matcher.

        Args:
            ontology_labels: Dict mapping class_name -> Vietnamese label
            use_compound_matching: Enable compound word detection
        """
        self.ontology_labels = ontology_labels
        self.use_compound_matching = use_compound_matching
        self._vi_to_class: Dict[str, str] = {}
        self._normalized_index: Dict[str, str] = {}

        self._build_lookup_index()

    def _build_lookup_index(self) -> None:
        """Build normalized lookup index for fast matching."""
        # Index from Vietnamese labels to class names
        for cls_name, vi_label in self.ontology_labels.items():
            normalized = self._normalize(vi_label)
            self._vi_to_class[normalized] = cls_name
            # Also index without underscores
            self._vi_to_class[normalized.replace("_", " ")] = cls_name
            # Index original lowercase
            self._vi_to_class[vi_label.lower()] = cls_name

        # Add compound mappings
        if self.use_compound_matching:
            for compound, cls_name in self.COMPOUND_MAPPINGS.items():
                self._vi_to_class[compound] = cls_name

        logger.debug(f"Built Vietnamese lookup index with {len(self._vi_to_class)} entries")

    def _normalize(self, text: str) -> str:
        """
        Normalize Vietnamese text for matching.

        Uses underthesea if available, otherwise basic normalization.
        """
        if not text:
            return ""

        text = text.lower().strip()

        if UNDERTHESEA_AVAILABLE:
            try:
                # Normalize unicode
                text = text_normalize(text)
                # Tokenize into words (returns list)
                tokens = word_tokenize(text)
                return " ".join(tokens).lower()
            except Exception as e:
                logger.debug(f"underthesea normalization failed: {e}")
                return text

        # Basic fallback normalization
        # Replace multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    def find_matches(self, query: str) -> List[str]:
        """
        Find ontology classes matching query terms.

        Uses N-gram matching (3, 2, 1) to find longest matches first.

        Args:
            query: Vietnamese query string

        Returns:
            List of matched ontology class names
        """
        if not query:
            return []

        normalized = self._normalize(query)
        matches: List[str] = []
        matched_spans: Set[tuple] = set()  # Track matched positions to avoid overlap

        # First check compound mappings (highest priority)
        if self.use_compound_matching:
            for compound, cls_name in self.COMPOUND_MAPPINGS.items():
                if compound in normalized:
                    if cls_name not in matches:
                        matches.append(cls_name)

        # N-gram matching (longest first: 4, 3, 2, 1)
        words = normalized.split()
        for n in [4, 3, 2, 1]:
            for i in range(len(words) - n + 1):
                # Skip if this span overlaps with already matched
                span = (i, i + n)
                if any(self._spans_overlap(span, m) for m in matched_spans):
                    continue

                ngram = " ".join(words[i : i + n])

                # Check direct match
                if ngram in self._vi_to_class:
                    cls_name = self._vi_to_class[ngram]
                    if cls_name not in matches:
                        matches.append(cls_name)
                        matched_spans.add(span)

                # Check without underscores
                ngram_no_underscore = ngram.replace("_", " ")
                if ngram_no_underscore in self._vi_to_class:
                    cls_name = self._vi_to_class[ngram_no_underscore]
                    if cls_name not in matches:
                        matches.append(cls_name)
                        matched_spans.add(span)

        return matches

    def _spans_overlap(self, span1: tuple, span2: tuple) -> bool:
        """Check if two spans overlap."""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def find_best_match(self, term: str) -> Optional[str]:
        """
        Find the single best matching ontology class for a term.

        Args:
            term: Vietnamese term to match

        Returns:
            Best matching class name or None
        """
        matches = self.find_matches(term)
        return matches[0] if matches else None

    def update_labels(self, new_labels: Dict[str, str]) -> None:
        """
        Update ontology labels and rebuild index.

        Args:
            new_labels: Additional class_name -> Vietnamese label mappings
        """
        self.ontology_labels.update(new_labels)
        self._build_lookup_index()

    @staticmethod
    def is_available() -> bool:
        """Check if Vietnamese NLP (underthesea) is available."""
        return UNDERTHESEA_AVAILABLE
