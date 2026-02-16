"""
Offline phase modules for Vietnamese Legal RAG.

This package contains modules for offline processing:
- Database models and management
- Entity/relation extraction
- Knowledge graph building
- Entity deduplication
- Type systems and mappings
- Abbreviation extraction
- Cross-reference detection
- Entity profiling
- Theme extraction
- Document filtering
"""

# Database models and manager
from .models import (
    Base,
    LegalAbbreviationModel,
    LegalArticleModel,
    LegalChapterModel,
    LegalClauseModel,
    LegalCrossReferenceModel,
    LegalDocumentModel,
    LegalPointModel,
    LegalSectionModel,
    make_article_id,
    make_chapter_id,
    make_clause_id,
    make_crossref_id,
    make_document_id,
    make_point_id,
    make_section_id,
    normalize_so_hieu,
)

from .database_manager import LegalDocumentDB

# Entity/relation extraction
from .unified_entity_relation_extractor import (
    ExtractionResult,
    UnifiedLegalExtractor,
)

# Knowledge graph building
from .incremental_knowledge_graph_builder import (
    IncrementalKGBuilder,
    IncrementalKGResult,
    MergedEntity,
    MergedRelation,
    slugify_vietnamese,
)

# Ontology generation
from importlib import import_module

_ontology_gen = import_module(".legal-ontology-generator", "vn_legal_rag.offline")
LegalOntologyGenerator = _ontology_gen.LegalOntologyGenerator
LegalOntology = _ontology_gen.LegalOntology
OntologyClass = _ontology_gen.OntologyClass
OntologyProperty = _ontology_gen.OntologyProperty

# Type systems (enum-based)
from .entity_types import LegalEntityType
from .relation_types import LegalRelationType
from .type_mapper import (
    map_entity_type,
    map_relation_type,
    entity_type_to_string,
    relation_type_to_string,
    map_entity,
    map_relation,
    map_extraction_result,
    is_valid_entity_type,
    is_valid_relation_type,
)

# Entity deduplication
from .deduplicator import (
    DeduplicationResult,
    EntityMatch,
    LegalEntityDeduplicator,
    deduplicate_entities,
    merge_entities_by_slug,
)

# Abbreviation extraction
from .abbreviation_extractor import (
    AbbreviationExtractor,
    AbbreviationMatch,
    KNOWN_LEGAL_ABBREVIATIONS,
    expand_search_terms,
    get_full_form,
)

# Cross-reference detection
from .crossref_detector import (
    CrossReference,
    CrossReferenceDetector,
    extract_context_sentence,
)

# Entity profiling
from .entity_profiler import (
    EntityProfile,
    EntityProfiler,
    ProfileCache,
)

# Theme extraction
from .theme_extractor import (
    Subgraph,
    Theme,
    ThemeExtractor,
    ThemeIndex,
)

# Document filtering
from .document_filter import (
    DOCUMENT_KEYWORDS,
    DOC_TYPE_PATTERNS,
    DocumentFilter,
    DocumentHint,
    FilteredResult,
)

__all__ = [
    # Models
    "Base",
    "LegalDocumentModel",
    "LegalChapterModel",
    "LegalSectionModel",
    "LegalArticleModel",
    "LegalClauseModel",
    "LegalPointModel",
    "LegalCrossReferenceModel",
    "LegalAbbreviationModel",
    # ID helpers
    "make_document_id",
    "make_chapter_id",
    "make_section_id",
    "make_article_id",
    "make_clause_id",
    "make_point_id",
    "make_crossref_id",
    "normalize_so_hieu",
    # Database
    "LegalDocumentDB",
    # Extraction
    "UnifiedLegalExtractor",
    "ExtractionResult",
    # KG Building
    "IncrementalKGBuilder",
    "IncrementalKGResult",
    "MergedEntity",
    "MergedRelation",
    "slugify_vietnamese",
    # Ontology
    "LegalOntologyGenerator",
    "LegalOntology",
    "OntologyClass",
    "OntologyProperty",
    # Type systems
    "LegalEntityType",
    "LegalRelationType",
    "map_entity_type",
    "map_relation_type",
    "entity_type_to_string",
    "relation_type_to_string",
    "map_entity",
    "map_relation",
    "map_extraction_result",
    "is_valid_entity_type",
    "is_valid_relation_type",
    # Deduplication
    "LegalEntityDeduplicator",
    "DeduplicationResult",
    "EntityMatch",
    "deduplicate_entities",
    "merge_entities_by_slug",
    # Abbreviation extraction
    "AbbreviationExtractor",
    "AbbreviationMatch",
    "get_full_form",
    "expand_search_terms",
    "KNOWN_LEGAL_ABBREVIATIONS",
    # Cross-reference detection
    "CrossReferenceDetector",
    "CrossReference",
    "extract_context_sentence",
    # Entity profiling
    "EntityProfiler",
    "EntityProfile",
    "ProfileCache",
    # Theme extraction
    "ThemeExtractor",
    "Theme",
    "Subgraph",
    "ThemeIndex",
    # Document filtering
    "DocumentFilter",
    "DocumentHint",
    "FilteredResult",
    "DOCUMENT_KEYWORDS",
    "DOC_TYPE_PATTERNS",
]
