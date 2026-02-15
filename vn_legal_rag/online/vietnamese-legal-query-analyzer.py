"""
Vietnamese Legal Query Analyzer

Merges query analysis, expansion, and classification into single module:
- Query intent detection (penalty, definition, procedure, etc.)
- Query type classification (article_lookup, guidance_document, etc.)
- Query expansion (abbreviations, synonyms)
- Entity extraction (article refs, law refs, keywords)

Provides optimized retrieval strategy based on query characteristics.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# Query Intent & Type Classification
# ============================================================================

class QueryIntent(Enum):
    """Legal query intent types."""
    PENALTY = "penalty"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    REQUIREMENT = "requirement"
    REFERENCE = "reference"
    GENERAL = "general"


class LegalQueryType(Enum):
    """Legal query types with retrieval strategy mapping."""
    ARTICLE_LOOKUP = "article_lookup"
    GUIDANCE_DOCUMENT = "guidance_document"
    SITUATION_ANALYSIS = "situation_analysis"
    COMPARE_REGULATIONS = "compare_regulations"
    CASE_LAW_LOOKUP = "case_law_lookup"
    TIMELINE_HISTORY = "timeline_history"
    GENERAL = "general"


@dataclass
class QueryTypeConfig:
    """Configuration for retrieval strategy based on query type."""
    query_type: LegalQueryType
    retrieval_method: str
    hybrid_alpha: float
    max_hops: int
    use_reasoning: bool = False
    use_temporal: bool = False


# Query type to retrieval strategy mapping
QUERY_TYPE_CONFIGS: Dict[LegalQueryType, QueryTypeConfig] = {
    LegalQueryType.ARTICLE_LOOKUP: QueryTypeConfig(
        query_type=LegalQueryType.ARTICLE_LOOKUP,
        retrieval_method="Vector + Light Graph",
        hybrid_alpha=0.3,
        max_hops=1,
    ),
    LegalQueryType.GUIDANCE_DOCUMENT: QueryTypeConfig(
        query_type=LegalQueryType.GUIDANCE_DOCUMENT,
        retrieval_method="Graph Traversal Multi-hop",
        hybrid_alpha=0.8,
        max_hops=3,
    ),
    LegalQueryType.SITUATION_ANALYSIS: QueryTypeConfig(
        query_type=LegalQueryType.SITUATION_ANALYSIS,
        retrieval_method="Hybrid + Reasoning",
        hybrid_alpha=0.7,
        max_hops=3,
        use_reasoning=True,
    ),
    LegalQueryType.COMPARE_REGULATIONS: QueryTypeConfig(
        query_type=LegalQueryType.COMPARE_REGULATIONS,
        retrieval_method="Multi-doc Comparison",
        hybrid_alpha=0.6,
        max_hops=2,
    ),
    LegalQueryType.CASE_LAW_LOOKUP: QueryTypeConfig(
        query_type=LegalQueryType.CASE_LAW_LOOKUP,
        retrieval_method="Case + Article Reference",
        hybrid_alpha=0.5,
        max_hops=2,
    ),
    LegalQueryType.TIMELINE_HISTORY: QueryTypeConfig(
        query_type=LegalQueryType.TIMELINE_HISTORY,
        retrieval_method="Temporal Retrieval",
        hybrid_alpha=0.4,
        max_hops=2,
        use_temporal=True,
    ),
    LegalQueryType.GENERAL: QueryTypeConfig(
        query_type=LegalQueryType.GENERAL,
        retrieval_method="Hybrid Search",
        hybrid_alpha=0.5,
        max_hops=2,
    ),
}


# ============================================================================
# Vietnamese Regex Patterns for Query Type Classification
# ============================================================================

# Rule-based patterns for fast query type classification (before LLM fallback)
QUERY_TYPE_PATTERNS: Dict[LegalQueryType, List[str]] = {
    LegalQueryType.ARTICLE_LOOKUP: [
        r"[Đđ]iều\s+\d+\s*(quy\s+định|nói|là)",
        r"theo\s+[Đđ]iều\s+\d+",
        r"nội\s+dung\s+[Đđ]iều\s+\d+",
        r"[Đđ]iều\s+\d+\s+[Ll]uật",
        r"khoản\s+\d+\s+[Đđ]iều\s+\d+",
    ],
    LegalQueryType.GUIDANCE_DOCUMENT: [
        r"nghị\s+định\s+.*hướng\s+dẫn",
        r"thông\s+tư\s+.*hướng\s+dẫn",
        r"văn\s+bản\s+.*hướng\s+dẫn",
        r"hướng\s+dẫn\s+thi\s+hành",
        r"quy\s+định\s+chi\s+tiết",
    ],
    LegalQueryType.SITUATION_ANALYSIS: [
        r"vi\s+phạm\s+.*điều",
        r"hành\s+vi\s+.*vi\s+phạm",
        r"trường\s+hợp\s+.*xử\s+lý",
        r"có\s+.*vi\s+phạm",
        r"bị\s+(xử\s+)?phạt",
        r"hậu\s+quả\s+pháp\s+lý",
    ],
    LegalQueryType.COMPARE_REGULATIONS: [
        r"so\s+sánh",
        r"khác\s+(biệt|nhau)",
        r"giống\s+nhau",
        r"sự\s+khác\s+biệt",
        r"điểm\s+mới",
        r"thay\s+đổi\s+gì",
    ],
    LegalQueryType.CASE_LAW_LOOKUP: [
        r"án\s+lệ",
        r"bản\s+án",
        r"quyết\s+định\s+.*tòa",
        r"tiền\s+lệ\s+pháp",
        r"vụ\s+án\s+.*liên\s+quan",
    ],
    LegalQueryType.TIMELINE_HISTORY: [
        r"lịch\s+sử\s+.*sửa\s+đổi",
        r"quá\s+trình\s+.*thay\s+đổi",
        r"từ\s+.*đến\s+nay",
        r"trước\s+đây",
        r"sau\s+khi\s+sửa\s+đổi",
        r"hiệu\s+lực\s+từ",
    ],
}

# Compile patterns for efficiency
_COMPILED_PATTERNS: Dict[LegalQueryType, List[re.Pattern]] = {
    qtype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for qtype, patterns in QUERY_TYPE_PATTERNS.items()
}


# ============================================================================
# Query Expansion
# ============================================================================

# Vietnamese abbreviations → full form
ABBREVIATIONS: Dict[str, str] = {
    "CTCP": "Công ty cổ phần",
    "TNHH": "Trách nhiệm hữu hạn",
    "TNHH 1TV": "Trách nhiệm hữu hạn một thành viên",
    "TNHH 2TV": "Trách nhiệm hữu hạn hai thành viên",
    "1TV": "một thành viên",
    "2TV": "hai thành viên trở lên",
    "DNTN": "Doanh nghiệp tư nhân",
    "HKD": "Hộ kinh doanh",
    "HĐQT": "Hội đồng quản trị",
    "HĐTV": "Hội đồng thành viên",
    "TGĐ": "Tổng giám đốc",
    "GĐ": "Giám đốc",
    "ĐKKD": "Đăng ký kinh doanh",
    "VĐL": "Vốn điều lệ",
    "MST": "Mã số thuế",
    "DN": "Doanh nghiệp",
    "cty": "công ty",
    "Cty": "Công ty",
}

# Synonym expansions: informal → formal legal terms
SYNONYMS: Dict[str, List[str]] = {
    "lập công ty": ["thành lập doanh nghiệp", "đăng ký doanh nghiệp"],
    "mở công ty": ["thành lập doanh nghiệp", "đăng ký doanh nghiệp"],
    "đóng cửa": ["giải thể doanh nghiệp", "chấm dứt hoạt động"],
    "chuyển đổi loại hình": ["chuyển đổi doanh nghiệp", "tổ chức lại doanh nghiệp"],
    "tăng vốn": ["tăng vốn điều lệ"],
    "giảm vốn": ["giảm vốn điều lệ"],
    "góp vốn": ["góp vốn điều lệ"],
    "góp thêm vốn": ["góp vốn bổ sung", "tăng vốn góp"],
    "thay đổi vốn": ["thay đổi vốn điều lệ", "tăng vốn điều lệ", "giảm vốn điều lệ"],
    "bán cổ phần": ["chuyển nhượng cổ phần"],
    "đổi đại diện": ["thay đổi người đại diện theo pháp luật"],
    "tăng vốn tnhh": ["tăng vốn điều lệ công ty TNHH"],
}

# Topic hints for chapter selection guidance
TOPIC_HINTS: Dict[str, List[str]] = {
    "chuyển_đổi": ["chuyển đổi loại hình", "tổ chức lại doanh nghiệp"],
    "giải_thể": ["giải thể doanh nghiệp", "chấm dứt hoạt động"],
    "tạm_ngừng": ["tạm ngừng kinh doanh", "ngừng hoạt động tạm thời"],
    "vốn_điều_lệ": ["tăng vốn điều lệ", "giảm vốn điều lệ"],
}


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    expanded: str
    abbreviations_found: List[Tuple[str, str]] = field(default_factory=list)
    synonyms_applied: List[Tuple[str, str]] = field(default_factory=list)
    topic_hints: List[str] = field(default_factory=list)


@dataclass
class AnalyzedQuery:
    """Analyzed query with extracted information."""
    original_query: str
    intent: QueryIntent
    query_type: LegalQueryType
    type_config: QueryTypeConfig
    article_refs: List[int] = field(default_factory=list)
    law_refs: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    expanded: Optional[ExpandedQuery] = None


# ============================================================================
# Main Query Analyzer Class
# ============================================================================

class VietnameseLegalQueryAnalyzer:
    """
    All-in-one query analyzer for Vietnamese legal queries.

    Combines:
    - Query expansion (abbreviations, synonyms)
    - Intent detection
    - Query type classification
    - Entity extraction
    """

    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            llm_provider: Optional LLM for advanced classification
        """
        self.llm_provider = llm_provider

    def analyze(self, query: str) -> AnalyzedQuery:
        """
        Analyze query and return comprehensive analysis.

        Args:
            query: Natural language query in Vietnamese

        Returns:
            AnalyzedQuery with all extracted information
        """
        # 1. Expand query
        expanded = self._expand_query(query)

        # 2. Detect intent
        intent = self._detect_intent(expanded.expanded)

        # 3. Classify query type
        query_type = self._classify_query_type(expanded.expanded)
        type_config = QUERY_TYPE_CONFIGS.get(
            query_type,
            QUERY_TYPE_CONFIGS[LegalQueryType.GENERAL]
        )

        # 4. Extract entities
        article_refs = self._extract_article_refs(query)
        law_refs = self._extract_law_refs(query)
        keywords = self._extract_keywords(query)

        return AnalyzedQuery(
            original_query=query,
            intent=intent,
            query_type=query_type,
            type_config=type_config,
            article_refs=article_refs,
            law_refs=law_refs,
            keywords=keywords,
            expanded=expanded,
        )

    def _expand_query(self, query: str) -> ExpandedQuery:
        """Expand query with abbreviations and synonyms."""
        result = ExpandedQuery(original=query, expanded=query)
        expanded = query

        # Build word-boundary regex pattern (sorted by length, longest first)
        sorted_abbrs = sorted(ABBREVIATIONS.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(abbr) for abbr in sorted_abbrs)
        abbr_pattern = re.compile(f"\\b({pattern})\\b", re.IGNORECASE)

        # Replacement function that preserves original case
        def replace_abbr(match):
            abbr = match.group(1)
            # Find canonical form (case-insensitive)
            for key, value in ABBREVIATIONS.items():
                if abbr.upper() == key.upper():
                    result.abbreviations_found.append((abbr, value))
                    return f"{abbr} ({value})"
            return abbr

        expanded = abbr_pattern.sub(replace_abbr, expanded)

        # Add synonym expansions
        query_lower = query.lower()
        for informal, formal_list in SYNONYMS.items():
            if informal in query_lower:
                for formal in formal_list:
                    if formal.lower() not in expanded.lower():
                        result.synonyms_applied.append((informal, formal))
                        expanded = f"{expanded}, {formal}"

        # Detect topic hints
        result.topic_hints = self._detect_topic_hints(query_lower)
        result.expanded = expanded

        return result

    def _detect_topic_hints(self, query_lower: str) -> List[str]:
        """Detect topic hints for guiding chapter selection."""
        hints = []

        # Conversion patterns
        if any(kw in query_lower for kw in ["chuyển đổi", "chuyển từ", "sang tnhh", "sang ctcp"]):
            hints.extend(TOPIC_HINTS.get("chuyển_đổi", []))

        # Dissolution
        if any(kw in query_lower for kw in ["giải thể", "đóng cửa", "chấm dứt"]):
            hints.extend(TOPIC_HINTS.get("giải_thể", []))

        # Suspension patterns (tạm ngừng kinh doanh)
        if any(kw in query_lower for kw in ["tạm ngừng", "tạm ngưng", "tạm nghỉ"]):
            hints.extend(TOPIC_HINTS.get("tạm_ngừng", []))

        # Capital changes
        if any(kw in query_lower for kw in ["tăng vốn", "giảm vốn"]):
            hints.extend(TOPIC_HINTS.get("vốn_điều_lệ", []))

        return list(set(hints))

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent from patterns."""
        query_lower = query.lower()

        # Penalty patterns
        if any(p in query_lower for p in ["phạt", "hình phạt", "mức phạt", "xử phạt"]):
            return QueryIntent.PENALTY

        # Definition patterns
        if any(p in query_lower for p in ["là gì", "định nghĩa", "nghĩa là"]):
            return QueryIntent.DEFINITION

        # Procedure patterns
        if any(p in query_lower for p in ["thủ tục", "quy trình", "hồ sơ"]):
            return QueryIntent.PROCEDURE

        # Requirement patterns
        if any(p in query_lower for p in ["điều kiện", "yêu cầu", "cần phải"]):
            return QueryIntent.REQUIREMENT

        # Reference patterns
        if re.search(r"[Đđ]iều\s+\d+", query):
            return QueryIntent.REFERENCE

        return QueryIntent.GENERAL

    def _classify_query_type(self, query: str) -> LegalQueryType:
        """
        Classify query type using hybrid approach: rules first, LLM fallback.

        Priority order:
        1. Rule-based classification (fast, regex patterns)
        2. LLM-based classification (if rules return GENERAL and LLM available)
        3. Default to GENERAL
        """
        # Step 1: Try rule-based classification (fast path)
        rule_result = self._classify_by_rules(query)
        if rule_result != LegalQueryType.GENERAL:
            return rule_result

        # Step 2: LLM fallback if available (slow path)
        # Note: LLM classification not implemented yet, return GENERAL
        return LegalQueryType.GENERAL

    def _classify_by_rules(self, query: str) -> LegalQueryType:
        """
        Fast rule-based classification using Vietnamese regex patterns.

        Returns:
            LegalQueryType - matched type or GENERAL if no patterns match
        """
        # Check each query type's patterns in priority order
        priority_order = [
            LegalQueryType.ARTICLE_LOOKUP,      # Most specific - direct article ref
            LegalQueryType.COMPARE_REGULATIONS, # Comparison keywords
            LegalQueryType.TIMELINE_HISTORY,    # Temporal keywords
            LegalQueryType.CASE_LAW_LOOKUP,     # Case law keywords
            LegalQueryType.GUIDANCE_DOCUMENT,   # Document reference
            LegalQueryType.SITUATION_ANALYSIS,  # Situation/violation analysis
        ]

        for qtype in priority_order:
            patterns = _COMPILED_PATTERNS.get(qtype, [])
            for pattern in patterns:
                if pattern.search(query):
                    return qtype

        return LegalQueryType.GENERAL

    def _extract_article_refs(self, query: str) -> List[int]:
        """Extract article number references."""
        refs = []
        for pattern in [r"[Đđ]iều\s+(\d+)", r"[Đđ]\.\s*(\d+)"]:
            matches = re.findall(pattern, query)
            refs.extend(int(m) for m in matches)
        return sorted(set(refs))

    def _extract_law_refs(self, query: str) -> List[str]:
        """Extract law number references."""
        refs = []
        patterns = [
            r"[Ll]uật\s+(?:số\s+)?(\d+/\d{4}/[A-Z0-9\-]+)",
            r"[Nn]ghị\s+định\s+(?:số\s+)?(\d+/\d{4}/[A-Z0-9\-]+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, query)
            refs.extend(matches)
        return list(set(refs))

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords for search."""
        # Simple word tokenization
        words = re.findall(r"\b[\w]+\b", query, re.UNICODE)

        stopwords = {
            'là', 'gì', 'của', 'trong', 'và', 'các', 'có', 'được',
            'để', 'cho', 'về', 'với', 'khi', 'này', 'theo',
        }

        keywords = []
        for word in words:
            if word.lower() not in stopwords and len(word) >= 2 and not word.isdigit():
                keywords.append(word)

        return keywords


# ============================================================================
# Convenience Functions
# ============================================================================

def expand_query(query: str) -> ExpandedQuery:
    """Convenience function to expand a query."""
    analyzer = VietnameseLegalQueryAnalyzer()
    return analyzer._expand_query(query)


def analyze_query(query: str, llm_provider: Optional[str] = None) -> AnalyzedQuery:
    """Convenience function to analyze a query."""
    analyzer = VietnameseLegalQueryAnalyzer(llm_provider=llm_provider)
    return analyzer.analyze(query)
