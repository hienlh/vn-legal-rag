"""
Legal domain relation types for Vietnamese legal document relation extraction.

Relation types designed for:
- Vietnamese legal documents (Luật, Nghị định, Thông tư)
- Cross-reference relationships between articles
- Semantic relationships in corporate law
"""

from enum import Enum


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
    REPLACES = "REPLACES"  # X thay thế hoàn toàn Y (alias for SUPERSEDES)
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

    # Database Relations (R_database - for cross-ref resolution)
    DEFINED_IN = "DEFINED_IN"  # X được định nghĩa tại Y
    MENTIONED_IN = "MENTIONED_IN"  # X được đề cập tại Y
