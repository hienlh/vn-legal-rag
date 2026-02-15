"""
Type definitions for Vietnamese legal RAG system.

Exports:
- Entity types (LegalEntityType, LEGAL_ENTITY_TYPES)
- Relation types (LegalRelationType)
- Tree models (TreeNode, TreeIndex, UnifiedForest, NodeType, CrossRefEdge)
- Type mappers (map_entity_type, map_relation_type, map_extraction_result)
"""

from typing import Any, Dict, List, Optional, Tuple

from .entity_types import LegalEntityType, LEGAL_ENTITY_TYPES
from .relation_types import LegalRelationType
from .tree_models import (
    TreeNode,
    TreeIndex,
    UnifiedForest,
    NodeType,
    CrossRefEdge,
    ForestStats,
)

# Ablation config for testing (import from kebab-case filename)
from importlib import import_module as _import_module
_ablation_module = _import_module(".ablation-config-for-rag-component-testing", "vn_legal_rag.types")
AblationConfig = _ablation_module.AblationConfig
get_ablation_configs = _ablation_module.get_ablation_configs
get_paper_ablation_configs = _ablation_module.get_paper_ablation_configs


# =============================================================================
# Type Mapping (merged from type_mapper.py)
# =============================================================================

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

STRING_TO_RELATION_TYPE: Dict[str, LegalRelationType] = {
    # Core relations
    "YÊU_CẦU": LegalRelationType.REQUIRES,
    "YEU_CAU": LegalRelationType.REQUIRES,
    "REQUIRES": LegalRelationType.REQUIRES,

    "THAM_CHIẾU": LegalRelationType.REFERENCES,
    "THAM_CHIEU": LegalRelationType.REFERENCES,
    "REFERENCES": LegalRelationType.REFERENCES,

    "ÁP_DỤNG_CHO": LegalRelationType.APPLIES_TO,
    "AP_DUNG_CHO": LegalRelationType.APPLIES_TO,
    "APPLIES_TO": LegalRelationType.APPLIES_TO,

    "ĐỊNH_NGHĨA_LÀ": LegalRelationType.DEFINED_AS,
    "DINH_NGHIA_LA": LegalRelationType.DEFINED_AS,
    "DEFINED_AS": LegalRelationType.DEFINED_AS,

    "BAO_GỒM": LegalRelationType.INCLUDES,
    "BAO_GOM": LegalRelationType.INCLUDES,
    "INCLUDES": LegalRelationType.INCLUDES,

    "THUỘC_VỀ": LegalRelationType.PART_OF,
    "THUOC_VE": LegalRelationType.PART_OF,
    "PART_OF": LegalRelationType.PART_OF,

    # Cross-reference relations
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
}


def map_entity_type(
    type_str: str,
    default: LegalEntityType = LegalEntityType.LEGAL_TERM,
) -> LegalEntityType:
    """
    Map string type to LegalEntityType enum.

    Args:
        type_str: Entity type as string
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


def map_extraction_result(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Map all entities and relations from extraction result.

    Args:
        entities: List of entity dicts
        relations: List of relation dicts

    Returns:
        Tuple of (mapped_entities, mapped_relations)
    """
    mapped_entities = []
    for e in entities:
        entity = e.copy()
        entity["entity_type"] = map_entity_type(e.get("type", ""))
        mapped_entities.append(entity)

    mapped_relations = []
    for r in relations:
        relation = r.copy()
        relation["relation_type"] = map_relation_type(r.get("predicate", ""))
        mapped_relations.append(relation)

    return mapped_entities, mapped_relations


__all__ = [
    # Entity types
    "LegalEntityType",
    "LEGAL_ENTITY_TYPES",

    # Relation types
    "LegalRelationType",

    # Tree models
    "TreeNode",
    "TreeIndex",
    "UnifiedForest",
    "NodeType",
    "CrossRefEdge",
    "ForestStats",

    # Ablation config
    "AblationConfig",
    "get_ablation_configs",
    "get_paper_ablation_configs",

    # Type mappers
    "map_entity_type",
    "map_relation_type",
    "map_extraction_result",
    "STRING_TO_ENTITY_TYPE",
    "STRING_TO_RELATION_TYPE",
]
