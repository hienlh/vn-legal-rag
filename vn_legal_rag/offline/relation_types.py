"""
Legal domain relation types for Vietnamese legal document relation extraction.

Relation types designed for:
- Vietnamese legal documents (Luật, Nghị định, Thông tư)
- Cross-reference relationships between articles
- Semantic relationships in corporate law

Ported from semantica/legal/relation_types.py
"""

from enum import Enum
from typing import Dict, List, FrozenSet


class LegalRelationType(Enum):
    """Relation types for Vietnamese legal documents."""

    # Prerequisite/Dependency Relations
    REQUIRES = "REQUIRES"  # X requires Y (điều kiện tiên quyết)
    DEPENDS_ON = "DEPENDS_ON"  # X depends on Y
    REQUIRED_BY = "REQUIRED_BY"  # X được yêu cầu bởi Y (inverse of REQUIRES)

    # Consequence Relations
    HAS_PENALTY = "HAS_PENALTY"  # violation has penalty
    RESULTS_IN = "RESULTS_IN"  # action results in consequence

    # Scope Relations
    APPLIES_TO = "APPLIES_TO"  # rule applies to subject
    EXCLUDES = "EXCLUDES"  # rule excludes subject

    # Conditional Relations
    CONDITION_FOR = "CONDITION_FOR"  # condition for action

    # Definition Relations
    DEFINED_AS = "DEFINED_AS"  # term defined as
    INCLUDES = "INCLUDES"  # definition includes

    # Cross-Reference Relations
    REFERENCES = "REFERENCES"  # article references another
    AMENDS = "AMENDS"  # article amends another
    SUPERSEDES = "SUPERSEDES"  # article supersedes another
    IMPLEMENTS = "IMPLEMENTS"  # article implements (huong dan thi hanh)
    REPLACES = "REPLACES"  # X thay thế hoàn toàn Y
    REPEALS = "REPEALS"  # X bãi bỏ Y

    # Structural Relations
    CONTAINS = "CONTAINS"  # document contains chapter/article
    PART_OF = "PART_OF"  # clause is part of article

    # Authority Relations
    AUTHORIZED_BY = "AUTHORIZED_BY"  # action authorized by entity
    PERFORMED_BY = "PERFORMED_BY"  # action performed by role
    HAS_AUTHORITY_OVER = "HAS_AUTHORITY_OVER"  # X có thẩm quyền đối với Y
    RESPONSIBLE_FOR = "RESPONSIBLE_FOR"  # X chịu trách nhiệm về Y
    DELEGATES_TO = "DELEGATES_TO"  # X ủy quyền cho Y

    # Ontology Relations
    IS_A = "IS_A"  # X là một loại Y (classification)
    RELATED_TO = "RELATED_TO"  # X liên quan đến Y (general)
    REGULATES = "REGULATES"  # X quy định về Y
    PRECEDES = "PRECEDES"  # X xảy ra trước Y (temporal/procedural)
    FOLLOWS = "FOLLOWS"  # X xảy ra sau Y (temporal/procedural)
    CONTRADICTS = "CONTRADICTS"  # X mâu thuẫn với Y
    SYNONYM = "SYNONYM"  # X đồng nghĩa với Y

    # Database Relations
    DEFINED_IN = "DEFINED_IN"  # X được định nghĩa tại Y
    MENTIONED_IN = "MENTIONED_IN"  # X được đề cập tại Y


# Generic relation types (Vietnamese)
LEGAL_RELATION_TYPES_GENERIC = [
    "YÊU_CẦU",
    "BỊ_PHẠT",
    "ÁP_DỤNG_CHO",
    "ĐIỀU_KIỆN_CHO",
    "ĐỊNH_NGHĨA_LÀ",
    "THAM_CHIẾU",
    "SỬA_ĐỔI",
    "BAO_GỒM",
    "QUY_ĐỊNH_VỀ",
    "BAN_HÀNH",
    "LÀ_MỘT",
    "LIÊN_QUAN_ĐẾN",
    "ĐƯỢC_YÊU_CẦU_BỞI",
    "XẢY_RA_TRƯỚC",
    "XẢY_RA_SAU",
    "MÂU_THUẪN_VỚI",
    "ĐỒNG_NGHĨA_VỚI",
    "THAY_THẾ",
    "BÃI_BỎ",
    "HƯỚNG_DẪN",
    "ĐỊNH_NGHĨA_TẠI",
    "ĐỀ_CẬP_TẠI",
]

# Domain-specific relation types
LEGAL_RELATION_TYPES_DOMAIN = [
    "CÓ_QUYỀN",
    "CÓ_NGHĨA_VỤ",
    "CÓ_THẨM_QUYỀN",
    "CÓ_THẨM_QUYỀN_ĐỐI_VỚI",
    "ỦY_QUYỀN_CHO",
    "CHỊU_TRÁCH_NHIỆM",
    "VI_PHẠM",
    "CÓ_CHẾ_TÀI",
    "CÓ_THỂ_BỊ_CHẾ_TÀI",
    "LOẠI_TRỪ",
    "NGOẠI_LỆ_CỦA",
    "BẢO_HỘ",
    "NGHIÊM_CẤM",
]

# Combined relation types
LEGAL_RELATION_TYPES = LEGAL_RELATION_TYPES_GENERIC + LEGAL_RELATION_TYPES_DOMAIN

# Set for O(1) lookup
LEGAL_RELATION_TYPES_SET: FrozenSet[str] = frozenset(t.upper() for t in LEGAL_RELATION_TYPES)


# Trigger words for each relation type
LEGAL_RELATION_TRIGGERS: Dict[str, List[str]] = {
    # Generic relations
    "YÊU_CẦU": ["phải có", "cần có", "yêu cầu", "đòi hỏi", "bắt buộc phải"],
    "BỊ_PHẠT": ["bị phạt", "bị xử phạt", "chịu phạt", "bị xử lý"],
    "ÁP_DỤNG_CHO": ["áp dụng cho", "áp dụng đối với", "được áp dụng"],
    "ĐIỀU_KIỆN_CHO": ["là điều kiện", "với điều kiện", "điều kiện để"],
    "ĐỊNH_NGHĨA_LÀ": ["là", "được hiểu là", "được định nghĩa là", "có nghĩa là"],
    "THAM_CHIẾU": ["theo", "căn cứ", "quy định tại", "nêu tại", "theo quy định"],
    "SỬA_ĐỔI": ["sửa đổi", "bổ sung"],
    "BAO_GỒM": ["bao gồm", "gồm có", "chứa", "bao hàm"],
    "QUY_ĐỊNH_VỀ": ["quy định về", "quy định chi tiết về", "hướng dẫn về"],
    "BAN_HÀNH": ["ban hành", "công bố", "phát hành"],
    # Ontology relations
    "LÀ_MỘT": ["là một", "là loại", "thuộc loại", "là một dạng"],
    "LIÊN_QUAN_ĐẾN": ["liên quan đến", "liên quan tới", "có liên quan", "gắn với"],
    "XẢY_RA_TRƯỚC": ["trước khi", "xảy ra trước", "thực hiện trước"],
    "XẢY_RA_SAU": ["sau khi", "xảy ra sau", "tiếp theo", "sau đó"],
    "MÂU_THUẪN_VỚI": ["mâu thuẫn với", "trái với", "không phù hợp với"],
    "ĐỒNG_NGHĨA_VỚI": ["đồng nghĩa với", "còn gọi là", "hay còn gọi", "viết tắt là"],
    "THAY_THẾ": ["thay thế", "thay cho", "thế chỗ"],
    "BÃI_BỎ": ["bãi bỏ", "hủy bỏ", "chấm dứt hiệu lực"],
    "HƯỚNG_DẪN": ["hướng dẫn thi hành", "hướng dẫn thực hiện", "quy định chi tiết thi hành"],
    # Domain-specific relations
    "CÓ_QUYỀN": ["có quyền", "được quyền", "quyền của", "được phép", "có thể"],
    "CÓ_NGHĨA_VỤ": ["có nghĩa vụ", "phải", "bắt buộc", "có trách nhiệm"],
    "CÓ_THẨM_QUYỀN": ["có thẩm quyền", "thuộc thẩm quyền", "quyết định về"],
    "CHỊU_TRÁCH_NHIỆM": ["chịu trách nhiệm", "trách nhiệm của", "phải chịu"],
    "VI_PHẠM": ["vi phạm", "không tuân thủ", "trái với", "không thực hiện"],
    "LOẠI_TRỪ": ["loại trừ", "không bao gồm", "trừ", "ngoại trừ"],
    "NGOẠI_LỆ_CỦA": ["trừ trường hợp", "không áp dụng", "ngoại lệ", "trừ khi"],
    "BẢO_HỘ": ["bảo hộ", "bảo vệ", "được bảo hộ"],
    "NGHIÊM_CẤM": ["nghiêm cấm", "cấm", "không được", "không được phép"],
}


# Relation patterns (enum-based)
RELATION_PATTERNS: Dict[LegalRelationType, List[str]] = {
    LegalRelationType.REQUIRES: [
        r"để\s+(.+?)\s+phải\s+(.+)",
        r"điều kiện\s+để\s+(.+?)\s+là\s+(.+)",
        r"yêu cầu\s+(.+?)\s+phải\s+(.+)",
    ],
    LegalRelationType.HAS_PENALTY: [
        r"vi phạm\s+(.+?)\s+(?:bị\s+)?(?:phạt|xử phạt)\s+(.+)",
        r"hành vi\s+(.+?)\s+(?:bị\s+)?(?:phạt|xử lý)\s+(.+)",
    ],
    LegalRelationType.APPLIES_TO: [
        r"(?:quy định\s+)?(?:này\s+)?áp dụng\s+(?:cho|đối với)\s+(.+)",
        r"(.+?)\s+(?:được\s+)?áp dụng\s+(?:cho|đối với)\s+(.+)",
    ],
    LegalRelationType.EXCLUDES: [
        r"(?:quy định\s+)?(?:này\s+)?không\s+áp dụng\s+(?:cho|đối với)\s+(.+)",
        r"trừ\s+(?:trường hợp\s+)?(.+)",
        r"ngoại trừ\s+(.+)",
    ],
    LegalRelationType.DEFINED_AS: [
        r"(.+?)\s+là\s+(.+)",
        r"(.+?)\s+được\s+(?:hiểu|định nghĩa)\s+là\s+(.+)",
    ],
    LegalRelationType.INCLUDES: [
        r"(.+?)\s+bao\s+gồm\s+(.+)",
        r"(.+?)\s+gồm\s+(?:có\s+)?(.+)",
    ],
    LegalRelationType.REFERENCES: [
        r"theo\s+(?:quy\s+định\s+)?(?:tại\s+)?Điều\s+(\d+)",
        r"căn\s+cứ\s+(?:vào\s+)?(?:Điều|Khoản)\s+(.+)",
    ],
    LegalRelationType.AMENDS: [
        r"sửa\s+đổi\s+(?:bổ\s+sung\s+)?(.+)",
        r"bổ\s+sung\s+(.+)",
    ],
    LegalRelationType.SUPERSEDES: [
        r"thay\s+thế\s+(.+)",
        r"bãi\s+bỏ\s+(.+)",
    ],
}


# Inverse relations for bidirectional extraction
INVERSE_RELATIONS: Dict[LegalRelationType, LegalRelationType] = {
    LegalRelationType.REQUIRES: LegalRelationType.CONDITION_FOR,
    LegalRelationType.APPLIES_TO: LegalRelationType.PART_OF,
    LegalRelationType.CONTAINS: LegalRelationType.PART_OF,
    LegalRelationType.INCLUDES: LegalRelationType.PART_OF,
    LegalRelationType.IS_A: LegalRelationType.CONTAINS,
    LegalRelationType.PRECEDES: LegalRelationType.FOLLOWS,
    LegalRelationType.REFERENCES: LegalRelationType.MENTIONED_IN,
    LegalRelationType.DELEGATES_TO: LegalRelationType.AUTHORIZED_BY,
    LegalRelationType.REQUIRES: LegalRelationType.REQUIRED_BY,
}

# Symmetric relations (A rel B ⟺ B rel A)
SYMMETRIC_RELATIONS: FrozenSet[LegalRelationType] = frozenset([
    LegalRelationType.RELATED_TO,
    LegalRelationType.CONTRADICTS,
    LegalRelationType.SYNONYM,
    LegalRelationType.EXCLUDES,
])

# Transitive relations (A rel B ∧ B rel C → A rel C)
TRANSITIVE_RELATIONS: FrozenSet[LegalRelationType] = frozenset([
    LegalRelationType.IS_A,
    LegalRelationType.PART_OF,
    LegalRelationType.CONTAINS,
    LegalRelationType.PRECEDES,
    LegalRelationType.FOLLOWS,
    LegalRelationType.REQUIRES,
])


# Cross-reference relation mapping
CROSSREF_RELATION_MAPPING: Dict[str, str] = {
    "THAM_CHIẾU": "references",
    "SỬA_ĐỔI": "amends",
    "THAY_THẾ": "supersedes",
    "BÃI_BỎ": "repeals",
    "HƯỚNG_DẪN": "implements",
    "ĐỊNH_NGHĨA_TẠI": "defined_in",
    "ĐỀ_CẬP_TẠI": "mentioned_in",
    "QUY_ĐỊNH_VỀ": "regulates",
}

CROSSREF_TYPE_TO_PREDICATE: Dict[str, str] = {
    v: k for k, v in CROSSREF_RELATION_MAPPING.items()
}

CROSSREF_PREDICATES: FrozenSet[str] = frozenset(CROSSREF_RELATION_MAPPING.keys())


# Vietnamese prompts for LLM-based extraction
LEGAL_RELATION_PROMPT_VI = """Bạn là chuyên gia trích xuất quan hệ ngữ nghĩa từ văn bản pháp luật Việt Nam.

Cho các thực thể đã được trích xuất, hãy xác định quan hệ giữa chúng.

## Loại quan hệ chính

| Quan hệ | Ý nghĩa | Trigger words |
|---------|---------|---------------|
| YÊU_CẦU | X yêu cầu/cần có Y | "phải có", "cần có", "yêu cầu" |
| BỊ_PHẠT | Vi phạm X bị xử phạt Y | "bị phạt", "bị xử phạt" |
| ÁP_DỤNG_CHO | Quy định X áp dụng cho Y | "áp dụng cho", "áp dụng đối với" |
| ĐỊNH_NGHĨA_LÀ | X được định nghĩa là Y | "là", "được hiểu là" |
| THAM_CHIẾU | X tham chiếu đến Y | "theo", "căn cứ", "quy định tại" |
| CÓ_QUYỀN | Chủ thể có quyền thực hiện X | "có quyền", "được quyền" |
| CÓ_NGHĨA_VỤ | Chủ thể có nghĩa vụ thực hiện X | "có nghĩa vụ", "phải" |
| NGHIÊM_CẤM | Cấm hành vi X | "nghiêm cấm", "cấm", "không được" |

Thực thể đã trích xuất:
{entities}

Văn bản:
{text}

Trả về JSON array:
[{{"subject": "entity_text", "predicate": "LOẠI_QUAN_HỆ", "object": "entity_text", "confidence": 0.0-1.0}}]
"""


# Pattern-based extraction fallback
LEGAL_RELATION_PATTERNS: Dict[str, List[str]] = {
    "YÊU_CẦU": [
        r"(?P<subject>.+?)\s+(?:phải có|cần có|yêu cầu|đòi hỏi)\s+(?P<object>.+)",
        r"(?:Để|Muốn)\s+(?P<subject>.+?)\s+(?:phải|cần)\s+(?P<object>.+)",
    ],
    "BỊ_PHẠT": [
        r"(?:Vi phạm|Không tuân thủ)\s+(?P<subject>.+?)\s+(?:bị phạt|bị xử phạt)\s+(?P<object>.+)",
    ],
    "ÁP_DỤNG_CHO": [
        r"(?P<subject>.+?)\s+(?:áp dụng cho|áp dụng đối với)\s+(?P<object>.+)",
    ],
    "ĐỊNH_NGHĨA_LÀ": [
        r"(?P<subject>.+?)\s+(?:là|được hiểu là|được định nghĩa là)\s+(?P<object>.+)",
    ],
    "THAM_CHIẾU": [
        r"(?:theo|căn cứ|quy định tại)\s+(?P<object>(?:Điều|Khoản|Điểm)\s+\d+)",
    ],
    "BAO_GỒM": [
        r"(?P<subject>.+?)\s+(?:bao gồm|gồm có|chứa)\s+(?P<object>.+)",
    ],
    "CÓ_QUYỀN": [
        r"(?P<subject>.+?)\s+(?:có quyền|được quyền|được phép)\s+(?P<object>.+)",
    ],
    "CÓ_NGHĨA_VỤ": [
        r"(?P<subject>.+?)\s+(?:có nghĩa vụ|có trách nhiệm)\s+(?P<object>.+)",
        r"(?P<subject>.+?)\s+(?:phải|bắt buộc)\s+(?P<object>.+)",
    ],
    "NGHIÊM_CẤM": [
        r"(?:Nghiêm cấm|Cấm)\s+(?P<subject>.+?)\s+(?P<object>.+)",
        r"(?P<subject>.+?)\s+(?:không được|không được phép)\s+(?P<object>.+)",
    ],
    "CHỊU_TRÁCH_NHIỆM": [
        r"(?P<subject>.+?)\s+(?:chịu trách nhiệm về|chịu trách nhiệm)\s+(?P<object>.+)",
    ],
}
