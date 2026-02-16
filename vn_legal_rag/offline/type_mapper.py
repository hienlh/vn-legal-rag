"""
Type Mapper for Vietnamese Legal Domain

Maps between unified extractor string types and enum types for ontology generation.
Ensures compatibility between LightRAG-style unified extraction and ontology generator.

Ported from semantica/legal/type_mapper.py
"""

from typing import Any, Dict, List, Optional, Tuple

from .entity_types import LegalEntityType
from .relation_types import LegalRelationType


# Entity Type Mapping: string -> enum
STRING_TO_ENTITY_TYPE: Dict[str, LegalEntityType] = {
    # Vietnamese types
    "TỔ_CHỨC": LegalEntityType.ORGANIZATION,
    "TO_CHUC": LegalEntityType.ORGANIZATION,
    "ORGANIZATION": LegalEntityType.ORGANIZATION,

    "VAI_TRÒ": LegalEntityType.PERSON_ROLE,
    "VAI_TRO": LegalEntityType.PERSON_ROLE,
    "PERSON_ROLE": LegalEntityType.PERSON_ROLE,

    "THUẬT_NGỮ": LegalEntityType.LEGAL_TERM,
    "THUAT_NGU": LegalEntityType.LEGAL_TERM,
    "LEGAL_TERM": LegalEntityType.LEGAL_TERM,

    "THAM_CHIẾU": LegalEntityType.LEGAL_REFERENCE,
    "THAM_CHIEU": LegalEntityType.LEGAL_REFERENCE,
    "LEGAL_REFERENCE": LegalEntityType.LEGAL_REFERENCE,

    "TIỀN_TỆ": LegalEntityType.MONETARY,
    "TIEN_TE": LegalEntityType.MONETARY,
    "MONETARY": LegalEntityType.MONETARY,

    "TỶ_LỆ": LegalEntityType.PERCENTAGE,
    "TY_LE": LegalEntityType.PERCENTAGE,
    "PERCENTAGE": LegalEntityType.PERCENTAGE,

    "THỜI_HẠN": LegalEntityType.DURATION,
    "THOI_HAN": LegalEntityType.DURATION,
    "DURATION": LegalEntityType.DURATION,

    "ĐỊA_ĐIỂM": LegalEntityType.LOCATION,
    "DIA_DIEM": LegalEntityType.LOCATION,
    "LOCATION": LegalEntityType.LOCATION,

    "ĐIỀU_KIỆN": LegalEntityType.CONDITION,
    "DIEU_KIEN": LegalEntityType.CONDITION,
    "CONDITION": LegalEntityType.CONDITION,

    "HÀNH_VI": LegalEntityType.ACTION,
    "HANH_VI": LegalEntityType.ACTION,
    "ACTION": LegalEntityType.ACTION,

    "CHẾ_TÀI": LegalEntityType.PENALTY,
    "CHE_TAI": LegalEntityType.PENALTY,
    "PENALTY": LegalEntityType.PENALTY,
}

# Reverse mapping: enum -> Vietnamese string
ENTITY_TYPE_TO_STRING: Dict[LegalEntityType, str] = {
    LegalEntityType.ORGANIZATION: "TỔ_CHỨC",
    LegalEntityType.PERSON_ROLE: "VAI_TRÒ",
    LegalEntityType.LEGAL_TERM: "THUẬT_NGỮ",
    LegalEntityType.LEGAL_REFERENCE: "THAM_CHIẾU",
    LegalEntityType.MONETARY: "TIỀN_TỆ",
    LegalEntityType.PERCENTAGE: "TỶ_LỆ",
    LegalEntityType.DURATION: "THỜI_HẠN",
    LegalEntityType.LOCATION: "ĐỊA_ĐIỂM",
    LegalEntityType.CONDITION: "ĐIỀU_KIỆN",
    LegalEntityType.ACTION: "HÀNH_VI",
    LegalEntityType.PENALTY: "CHẾ_TÀI",
}


# Relation Type Mapping: string -> enum
STRING_TO_RELATION_TYPE: Dict[str, LegalRelationType] = {
    "YÊU_CẦU": LegalRelationType.REQUIRES,
    "YEU_CAU": LegalRelationType.REQUIRES,
    "REQUIRES": LegalRelationType.REQUIRES,

    "THUỘC_VỀ": LegalRelationType.PART_OF,
    "THUOC_VE": LegalRelationType.PART_OF,
    "PART_OF": LegalRelationType.PART_OF,

    "CÓ_THẨM_QUYỀN": LegalRelationType.HAS_AUTHORITY_OVER,
    "CO_THAM_QUYEN": LegalRelationType.HAS_AUTHORITY_OVER,
    "HAS_AUTHORITY_OVER": LegalRelationType.HAS_AUTHORITY_OVER,

    "THAM_CHIẾU": LegalRelationType.REFERENCES,
    "THAM_CHIEU": LegalRelationType.REFERENCES,
    "REFERENCES": LegalRelationType.REFERENCES,

    "ĐỊNH_NGHĨA": LegalRelationType.DEFINED_AS,
    "DINH_NGHIA": LegalRelationType.DEFINED_AS,
    "ĐỊNH_NGHĨA_LÀ": LegalRelationType.DEFINED_AS,
    "DINH_NGHIA_LA": LegalRelationType.DEFINED_AS,
    "DEFINED_AS": LegalRelationType.DEFINED_AS,

    "ÁP_DỤNG": LegalRelationType.APPLIES_TO,
    "AP_DUNG": LegalRelationType.APPLIES_TO,
    "ÁP_DỤNG_CHO": LegalRelationType.APPLIES_TO,
    "AP_DUNG_CHO": LegalRelationType.APPLIES_TO,
    "APPLIES_TO": LegalRelationType.APPLIES_TO,

    "XỬ_PHẠT": LegalRelationType.HAS_PENALTY,
    "XU_PHAT": LegalRelationType.HAS_PENALTY,
    "BỊ_PHẠT": LegalRelationType.HAS_PENALTY,
    "BI_PHAT": LegalRelationType.HAS_PENALTY,
    "HAS_PENALTY": LegalRelationType.HAS_PENALTY,

    "THỜI_HẠN": LegalRelationType.CONDITION_FOR,
    "THOI_HAN": LegalRelationType.CONDITION_FOR,

    "TỶ_LỆ": LegalRelationType.INCLUDES,
    "TY_LE": LegalRelationType.INCLUDES,

    "NGOẠI_TRỪ": LegalRelationType.EXCLUDES,
    "NGOAI_TRU": LegalRelationType.EXCLUDES,
    "EXCLUDES": LegalRelationType.EXCLUDES,

    "BAO_GỒM": LegalRelationType.INCLUDES,
    "BAO_GOM": LegalRelationType.INCLUDES,
    "INCLUDES": LegalRelationType.INCLUDES,

    "SỬA_ĐỔI": LegalRelationType.AMENDS,
    "SUA_DOI": LegalRelationType.AMENDS,
    "AMENDS": LegalRelationType.AMENDS,

    "THAY_THẾ": LegalRelationType.SUPERSEDES,
    "THAY_THE": LegalRelationType.SUPERSEDES,
    "SUPERSEDES": LegalRelationType.SUPERSEDES,

    "BÃI_BỎ": LegalRelationType.REPEALS,
    "BAI_BO": LegalRelationType.REPEALS,
    "REPEALS": LegalRelationType.REPEALS,

    "HƯỚNG_DẪN": LegalRelationType.IMPLEMENTS,
    "HUONG_DAN": LegalRelationType.IMPLEMENTS,
    "IMPLEMENTS": LegalRelationType.IMPLEMENTS,

    "CÓ_QUYỀN": LegalRelationType.AUTHORIZED_BY,
    "CO_QUYEN": LegalRelationType.AUTHORIZED_BY,

    "CÓ_NGHĨA_VỤ": LegalRelationType.RESPONSIBLE_FOR,
    "CO_NGHIA_VU": LegalRelationType.RESPONSIBLE_FOR,

    "CHỊU_TRÁCH_NHIỆM": LegalRelationType.RESPONSIBLE_FOR,
    "CHIU_TRACH_NHIEM": LegalRelationType.RESPONSIBLE_FOR,
    "RESPONSIBLE_FOR": LegalRelationType.RESPONSIBLE_FOR,

    "ỦY_QUYỀN_CHO": LegalRelationType.DELEGATES_TO,
    "UY_QUYEN_CHO": LegalRelationType.DELEGATES_TO,
    "DELEGATES_TO": LegalRelationType.DELEGATES_TO,

    "LÀ_MỘT": LegalRelationType.IS_A,
    "LA_MOT": LegalRelationType.IS_A,
    "IS_A": LegalRelationType.IS_A,

    "LIÊN_QUAN_ĐẾN": LegalRelationType.RELATED_TO,
    "LIEN_QUAN_DEN": LegalRelationType.RELATED_TO,
    "LIÊN_QUAN": LegalRelationType.RELATED_TO,
    "RELATED_TO": LegalRelationType.RELATED_TO,

    "QUY_ĐỊNH_VỀ": LegalRelationType.REGULATES,
    "QUY_DINH_VE": LegalRelationType.REGULATES,
    "REGULATES": LegalRelationType.REGULATES,

    "XẢY_RA_TRƯỚC": LegalRelationType.PRECEDES,
    "XAY_RA_TRUOC": LegalRelationType.PRECEDES,
    "PRECEDES": LegalRelationType.PRECEDES,

    "XẢY_RA_SAU": LegalRelationType.FOLLOWS,
    "XAY_RA_SAU": LegalRelationType.FOLLOWS,
    "FOLLOWS": LegalRelationType.FOLLOWS,

    "MÂU_THUẪN_VỚI": LegalRelationType.CONTRADICTS,
    "MAU_THUAN_VOI": LegalRelationType.CONTRADICTS,
    "CONTRADICTS": LegalRelationType.CONTRADICTS,

    "ĐỒNG_NGHĨA_VỚI": LegalRelationType.SYNONYM,
    "DONG_NGHIA_VOI": LegalRelationType.SYNONYM,
    "SYNONYM": LegalRelationType.SYNONYM,

    "NGHIÊM_CẤM": LegalRelationType.EXCLUDES,
    "NGHIEM_CAM": LegalRelationType.EXCLUDES,

    "LOẠI_TRỪ": LegalRelationType.EXCLUDES,
    "LOAI_TRU": LegalRelationType.EXCLUDES,

    "VI_PHẠM": LegalRelationType.CONTRADICTS,
    "VI_PHAM": LegalRelationType.CONTRADICTS,

    "BẢO_HỘ": LegalRelationType.AUTHORIZED_BY,
    "BAO_HO": LegalRelationType.AUTHORIZED_BY,

    "ĐIỀU_KIỆN_CHO": LegalRelationType.CONDITION_FOR,
    "DIEU_KIEN_CHO": LegalRelationType.CONDITION_FOR,
    "CONDITION_FOR": LegalRelationType.CONDITION_FOR,

    "ĐƯỢC_YÊU_CẦU_BỞI": LegalRelationType.REQUIRED_BY,
    "DUOC_YEU_CAU_BOI": LegalRelationType.REQUIRED_BY,
    "REQUIRED_BY": LegalRelationType.REQUIRED_BY,

    "KẾT_QUẢ": LegalRelationType.RESULTS_IN,
    "KET_QUA": LegalRelationType.RESULTS_IN,
    "RESULTS_IN": LegalRelationType.RESULTS_IN,

    "THỰC_HIỆN_BỞI": LegalRelationType.PERFORMED_BY,
    "THUC_HIEN_BOI": LegalRelationType.PERFORMED_BY,
    "PERFORMED_BY": LegalRelationType.PERFORMED_BY,

    "CHỨA": LegalRelationType.CONTAINS,
    "CHUA": LegalRelationType.CONTAINS,
    "CONTAINS": LegalRelationType.CONTAINS,

    "ĐỊNH_NGHĨA_TẠI": LegalRelationType.DEFINED_IN,
    "DINH_NGHIA_TAI": LegalRelationType.DEFINED_IN,
    "DEFINED_IN": LegalRelationType.DEFINED_IN,

    "ĐỀ_CẬP_TẠI": LegalRelationType.MENTIONED_IN,
    "DE_CAP_TAI": LegalRelationType.MENTIONED_IN,
    "MENTIONED_IN": LegalRelationType.MENTIONED_IN,
}

# Reverse mapping: enum -> Vietnamese string
RELATION_TYPE_TO_STRING: Dict[LegalRelationType, str] = {
    LegalRelationType.REQUIRES: "YÊU_CẦU",
    LegalRelationType.DEPENDS_ON: "PHỤ_THUỘC",
    LegalRelationType.REQUIRED_BY: "ĐƯỢC_YÊU_CẦU_BỞI",
    LegalRelationType.HAS_PENALTY: "BỊ_PHẠT",
    LegalRelationType.RESULTS_IN: "KẾT_QUẢ",
    LegalRelationType.APPLIES_TO: "ÁP_DỤNG_CHO",
    LegalRelationType.EXCLUDES: "LOẠI_TRỪ",
    LegalRelationType.CONDITION_FOR: "ĐIỀU_KIỆN_CHO",
    LegalRelationType.DEFINED_AS: "ĐỊNH_NGHĨA_LÀ",
    LegalRelationType.INCLUDES: "BAO_GỒM",
    LegalRelationType.CONTAINS: "CHỨA",
    LegalRelationType.REFERENCES: "THAM_CHIẾU",
    LegalRelationType.AMENDS: "SỬA_ĐỔI",
    LegalRelationType.SUPERSEDES: "THAY_THẾ",
    LegalRelationType.IMPLEMENTS: "HƯỚNG_DẪN",
    LegalRelationType.REPLACES: "THAY_THẾ",
    LegalRelationType.REPEALS: "BÃI_BỎ",
    LegalRelationType.PART_OF: "THUỘC_VỀ",
    LegalRelationType.PRECEDES: "XẢY_RA_TRƯỚC",
    LegalRelationType.FOLLOWS: "XẢY_RA_SAU",
    LegalRelationType.AUTHORIZED_BY: "ĐƯỢC_ỦY_QUYỀN",
    LegalRelationType.PERFORMED_BY: "THỰC_HIỆN_BỞI",
    LegalRelationType.HAS_AUTHORITY_OVER: "CÓ_THẨM_QUYỀN",
    LegalRelationType.RESPONSIBLE_FOR: "CHỊU_TRÁCH_NHIỆM",
    LegalRelationType.DELEGATES_TO: "ỦY_QUYỀN_CHO",
    LegalRelationType.IS_A: "LÀ_MỘT",
    LegalRelationType.RELATED_TO: "LIÊN_QUAN_ĐẾN",
    LegalRelationType.REGULATES: "QUY_ĐỊNH_VỀ",
    LegalRelationType.CONTRADICTS: "MÂU_THUẪN_VỚI",
    LegalRelationType.SYNONYM: "ĐỒNG_NGHĨA_VỚI",
    LegalRelationType.DEFINED_IN: "ĐỊNH_NGHĨA_TẠI",
    LegalRelationType.MENTIONED_IN: "ĐỀ_CẬP_TẠI",
}


def map_entity_type(
    type_str: str,
    default: LegalEntityType = LegalEntityType.LEGAL_TERM,
) -> LegalEntityType:
    """
    Map string type to LegalEntityType enum.

    Args:
        type_str: Entity type as string (e.g., "TỔ_CHỨC", "ORGANIZATION")
        default: Default type if mapping not found

    Returns:
        LegalEntityType enum value
    """
    if not type_str:
        return default

    normalized = type_str.upper().strip()
    return STRING_TO_ENTITY_TYPE.get(normalized, default)


def map_relation_type(
    predicate: str,
    default: LegalRelationType = LegalRelationType.RELATED_TO,
) -> LegalRelationType:
    """
    Map string predicate to LegalRelationType enum.

    Args:
        predicate: Relation predicate as string
        default: Default type if mapping not found

    Returns:
        LegalRelationType enum value
    """
    if not predicate:
        return default

    normalized = predicate.upper().strip()
    return STRING_TO_RELATION_TYPE.get(normalized, default)


def entity_type_to_string(
    entity_type: LegalEntityType,
    use_vietnamese: bool = True,
) -> str:
    """Convert LegalEntityType enum to string."""
    if use_vietnamese:
        return ENTITY_TYPE_TO_STRING.get(entity_type, entity_type.value)
    return entity_type.value


def relation_type_to_string(
    relation_type: LegalRelationType,
    use_vietnamese: bool = True,
) -> str:
    """Convert LegalRelationType enum to string."""
    if use_vietnamese:
        return RELATION_TYPE_TO_STRING.get(relation_type, relation_type.value)
    return relation_type.value


def map_entity(
    entity: Dict[str, Any],
    add_enum: bool = True,
) -> Dict[str, Any]:
    """Map entity dict to include enum type."""
    result = entity.copy()
    type_str = entity.get("type", "")
    entity_type = map_entity_type(type_str)

    if add_enum:
        result["entity_type"] = entity_type

    result["type_mapped"] = entity_type.value
    return result


def map_relation(
    relation: Dict[str, Any],
    add_enum: bool = True,
) -> Dict[str, Any]:
    """Map relation dict to include enum type."""
    result = relation.copy()
    predicate = relation.get("predicate", "")
    relation_type = map_relation_type(predicate)

    if add_enum:
        result["relation_type"] = relation_type

    result["predicate_mapped"] = relation_type.value
    return result


def map_extraction_result(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    add_enums: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Map all entities and relations from unified extraction result.

    Args:
        entities: List of entity dicts
        relations: List of relation dicts
        add_enums: If True, add enum fields

    Returns:
        Tuple of (mapped_entities, mapped_relations)
    """
    mapped_entities = [map_entity(e, add_enum=add_enums) for e in entities]
    mapped_relations = [map_relation(r, add_enum=add_enums) for r in relations]
    return mapped_entities, mapped_relations


def is_valid_entity_type(type_str: str) -> bool:
    """Check if string is a valid entity type."""
    return type_str.upper().strip() in STRING_TO_ENTITY_TYPE


def is_valid_relation_type(predicate: str) -> bool:
    """Check if string is a valid relation type."""
    return predicate.upper().strip() in STRING_TO_RELATION_TYPE


def get_all_entity_types() -> List[str]:
    """Get all valid entity type strings."""
    return list(STRING_TO_ENTITY_TYPE.keys())


def get_all_relation_types() -> List[str]:
    """Get all valid relation type strings."""
    return list(STRING_TO_RELATION_TYPE.keys())
