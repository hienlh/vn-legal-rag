"""
Ontology-Based Query Expander for Vietnamese Legal RAG

Expands queries using ontology class hierarchy for better retrieval coverage:
- Company types: CTCP, TNHH, DNTN, etc.
- Document types: Luat, Nghi dinh, Thong tu, etc.
- Legal concepts hierarchy traversal

Supports expansion modes: children, parents, siblings, all
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# Vietnamese Legal Ontology - Class Hierarchy
# ============================================================================

# Class hierarchy: child -> parent mapping
CLASS_HIERARCHY: Dict[str, Optional[str]] = {
    # Root classes
    "Thing": None,
    "Organization": "Thing",
    "Document": "Thing",
    "LegalConcept": "Thing",

    # Company types
    "Company": "Organization",
    "JointStockCompany": "Company",          # CTCP
    "LimitedLiabilityCompany": "Company",    # TNHH
    "SingleMemberLLC": "LimitedLiabilityCompany",  # TNHH 1TV
    "MultiMemberLLC": "LimitedLiabilityCompany",   # TNHH 2TV+
    "PrivateEnterprise": "Company",          # DNTN
    "Partnership": "Company",                # Hop danh
    "HouseholdBusiness": "Organization",     # HKD

    # Document types
    "Law": "Document",                       # Luat
    "Decree": "Document",                    # Nghi dinh
    "Circular": "Document",                  # Thong tu
    "Decision": "Document",                  # Quyet dinh
    "Resolution": "Document",                # Nghi quyet

    # Legal concepts
    "CharterCapital": "LegalConcept",        # Von dieu le
    "LegalRepresentative": "LegalConcept",   # Nguoi DDPL
    "Shareholder": "LegalConcept",           # Co dong
    "Member": "LegalConcept",                # Thanh vien
    "BoardOfDirectors": "LegalConcept",      # HDQT
    "MembersCouncil": "LegalConcept",        # HDTV
}

# Vietnamese labels for classes
CLASS_LABELS_VI: Dict[str, str] = {
    "Company": "công ty",
    "JointStockCompany": "công ty cổ phần",
    "LimitedLiabilityCompany": "công ty trách nhiệm hữu hạn",
    "SingleMemberLLC": "công ty TNHH một thành viên",
    "MultiMemberLLC": "công ty TNHH hai thành viên",
    "PrivateEnterprise": "doanh nghiệp tư nhân",
    "Partnership": "công ty hợp danh",
    "HouseholdBusiness": "hộ kinh doanh",
    "Organization": "tổ chức",
    "Law": "luật",
    "Decree": "nghị định",
    "Circular": "thông tư",
    "Decision": "quyết định",
    "Resolution": "nghị quyết",
    "Document": "văn bản",
    "CharterCapital": "vốn điều lệ",
    "LegalRepresentative": "người đại diện theo pháp luật",
    "Shareholder": "cổ đông",
    "Member": "thành viên",
    "BoardOfDirectors": "hội đồng quản trị",
    "MembersCouncil": "hội đồng thành viên",
}

# Vietnamese term -> class mapping (for lookup)
VI_TO_CLASS: Dict[str, str] = {v.lower(): k for k, v in CLASS_LABELS_VI.items()}

# Additional Vietnamese abbreviations and variants
VI_ABBREVIATIONS: Dict[str, str] = {
    "ctcp": "JointStockCompany",
    "tnhh": "LimitedLiabilityCompany",
    "tnhh 1tv": "SingleMemberLLC",
    "tnhh mtv": "SingleMemberLLC",
    "tnhh 2tv": "MultiMemberLLC",
    "dntn": "PrivateEnterprise",
    "hkd": "HouseholdBusiness",
    "nđ": "Decree",
    "nd": "Decree",
    "tt": "Circular",
    "vđl": "CharterCapital",
    "nddpl": "LegalRepresentative",
    "hđqt": "BoardOfDirectors",
    "hdqt": "BoardOfDirectors",
    "hđtv": "MembersCouncil",
}


@dataclass
class ExpansionResult:
    """Result of ontology-based term expansion."""
    original_term: str
    matched_class: Optional[str] = None
    parent_classes: List[str] = field(default_factory=list)
    child_classes: List[str] = field(default_factory=list)
    sibling_classes: List[str] = field(default_factory=list)
    expanded_terms_vi: List[str] = field(default_factory=list)

    @property
    def all_expanded_terms(self) -> List[str]:
        """Get all expanded Vietnamese terms including original."""
        terms = [self.original_term]
        terms.extend(self.expanded_terms_vi)
        return list(dict.fromkeys(terms))  # Dedupe preserving order


class OntologyExpander:
    """
    Expands queries using ontology class hierarchy.

    Example:
        >>> expander = OntologyExpander()
        >>> result = expander.expand_term("công ty", include_children=True)
        >>> print(result.expanded_terms_vi)
        ['công ty cổ phần', 'công ty trách nhiệm hữu hạn', ...]
    """

    def __init__(self):
        """Initialize with built-in Vietnamese legal ontology."""
        self._hierarchy = CLASS_HIERARCHY
        self._labels_vi = CLASS_LABELS_VI
        self._vi_to_class = {**VI_TO_CLASS, **VI_ABBREVIATIONS}
        self._children = self._build_children_map()

    def _build_children_map(self) -> Dict[str, List[str]]:
        """Build parent -> children mapping for downward traversal."""
        children: Dict[str, List[str]] = {}
        for child, parent in self._hierarchy.items():
            if parent:
                if parent not in children:
                    children[parent] = []
                children[parent].append(child)
        return children

    def find_class(self, term: str) -> Optional[str]:
        """Find ontology class for Vietnamese term."""
        term_lower = term.lower().strip()

        # Direct lookup
        if term_lower in self._vi_to_class:
            return self._vi_to_class[term_lower]

        # Partial match (term contains class label or vice versa)
        for vi_label, class_name in self._vi_to_class.items():
            if vi_label in term_lower or term_lower in vi_label:
                return class_name

        return None

    def get_parent_classes(self, class_name: str, max_depth: int = 3) -> List[str]:
        """Get parent classes up the hierarchy."""
        parents = []
        current = class_name
        depth = 0

        while depth < max_depth:
            parent = self._hierarchy.get(current)
            if not parent or parent == "Thing":
                break
            parents.append(parent)
            current = parent
            depth += 1

        return parents

    def get_child_classes(self, class_name: str, max_depth: int = 2) -> List[str]:
        """Get child classes down the hierarchy (DFS)."""
        children = []
        stack = [(class_name, 0)]

        while stack:
            current, depth = stack.pop()
            if depth >= max_depth:
                continue
            for child in self._children.get(current, []):
                children.append(child)
                stack.append((child, depth + 1))

        return children

    def get_sibling_classes(self, class_name: str) -> List[str]:
        """Get sibling classes (same parent)."""
        parent = self._hierarchy.get(class_name)
        if not parent:
            return []

        siblings = []
        for child in self._children.get(parent, []):
            if child != class_name:
                siblings.append(child)
        return siblings

    def expand_term(
        self,
        term: str,
        include_parents: bool = True,
        include_children: bool = True,
        include_siblings: bool = False,
        max_depth: int = 2,
    ) -> ExpansionResult:
        """
        Expand a single Vietnamese term using ontology.

        Args:
            term: Vietnamese term to expand
            include_parents: Include parent classes
            include_children: Include child classes (subclasses)
            include_siblings: Include sibling classes
            max_depth: Maximum traversal depth

        Returns:
            ExpansionResult with matched class and expanded terms
        """
        result = ExpansionResult(original_term=term)

        class_name = self.find_class(term)
        if not class_name:
            return result

        result.matched_class = class_name

        # Collect related classes
        if include_parents:
            result.parent_classes = self.get_parent_classes(class_name, max_depth)
        if include_children:
            result.child_classes = self.get_child_classes(class_name, max_depth)
        if include_siblings:
            result.sibling_classes = self.get_sibling_classes(class_name)

        # Convert to Vietnamese labels
        all_classes = result.parent_classes + result.child_classes + result.sibling_classes
        for cls in all_classes:
            vi_label = self._labels_vi.get(cls)
            if vi_label and vi_label not in result.expanded_terms_vi:
                result.expanded_terms_vi.append(vi_label)

        return result

    def expand_query(
        self,
        query: str,
        mode: str = "children",
        max_terms: int = 10,
    ) -> Tuple[str, List[ExpansionResult]]:
        """
        Expand a query with ontology-based terms.

        Args:
            query: Original query string
            mode: Expansion mode - "children", "parents", "siblings", "all"
            max_terms: Maximum expanded terms to add

        Returns:
            Tuple of (expanded_query, list_of_expansion_results)
        """
        # Set expansion flags based on mode
        include_parents = mode in ("parents", "all")
        include_children = mode in ("children", "all")
        include_siblings = mode in ("siblings", "all")

        expansions: List[ExpansionResult] = []
        expanded_terms: Set[str] = set()

        # Try n-gram matching (longest first: 3-gram, 2-gram, 1-gram)
        words = query.lower().split()
        matched_spans: Set[int] = set()  # Track matched word indices

        for n in [3, 2, 1]:
            for i in range(len(words) - n + 1):
                if any(j in matched_spans for j in range(i, i + n)):
                    continue  # Skip if any word already matched

                ngram = " ".join(words[i:i + n])
                result = self.expand_term(
                    ngram,
                    include_parents=include_parents,
                    include_children=include_children,
                    include_siblings=include_siblings,
                )

                if result.matched_class:
                    expansions.append(result)
                    expanded_terms.update(result.expanded_terms_vi)
                    matched_spans.update(range(i, i + n))

        # Build expanded query
        if expanded_terms:
            # Limit to max_terms
            terms_to_add = list(expanded_terms)[:max_terms]
            expanded_query = f"{query} ({' '.join(terms_to_add)})"
        else:
            expanded_query = query

        return expanded_query, expansions
