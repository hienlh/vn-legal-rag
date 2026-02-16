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
- Summary generators (chapter, article, document)
- Web scraping (TVPL legal document scraper)
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

# Cross-reference post-processor (kebab-case filename)
_crossref_post = import_module(
    ".crossref-postprocessor-for-llm-extracted-relations", "vn_legal_rag.offline"
)
CrossRefPostProcessor = _crossref_post.CrossRefPostProcessor
CrossRefCandidate = _crossref_post.CrossRefCandidate
ParsedReference = _crossref_post.ParsedReference
extract_crossrefs_from_relations = _crossref_post.extract_crossrefs_from_relations

# Incremental KG updater (kebab-case filename)
_incr_updater = import_module(
    ".incremental-kg-updater-for-new-documents", "vn_legal_rag.offline"
)
IncrementalUpdater = _incr_updater.IncrementalUpdater
UpdateResult = _incr_updater.UpdateResult

# Terminology extractor (kebab-case filename)
_term_extractor = import_module(
    ".terminology-extractor-from-legal-articles", "vn_legal_rag.offline"
)
TerminologyExtractor = _term_extractor.TerminologyExtractor
ExtractedTerm = _term_extractor.ExtractedTerm
TerminologyResult = _term_extractor.TerminologyResult
extract_terminology_from_db = _term_extractor.extract_terminology_from_db

# Synonym miner (kebab-case filename)
_synonym_miner = import_module(
    ".synonym-miner-from-qa-corpus", "vn_legal_rag.offline"
)
SynonymMiner = _synonym_miner.SynonymMiner
SynonymPair = _synonym_miner.SynonymPair
SynonymMiningResult = _synonym_miner.SynonymMiningResult
mine_synonyms_for_domain = _synonym_miner.mine_synonyms_for_domain

# KG-SQLite bidirectional linker (kebab-case filename)
_kg_linker = import_module(
    ".kg-sqlite-bidirectional-linker", "vn_legal_rag.offline"
)
KGSQLiteLinker = _kg_linker.KGSQLiteLinker
EntityContext = _kg_linker.EntityContext
LinkResult = _kg_linker.LinkResult
create_linker = _kg_linker.create_linker

# Relation validator (kebab-case filename)
_rel_validator = import_module(
    ".relation-validator-for-kg-pipeline", "vn_legal_rag.offline"
)
RelationValidator = _rel_validator.RelationValidator
ValidationStats = _rel_validator.ValidationStats
SEMANTIC_TYPE_RULES = _rel_validator.SEMANTIC_TYPE_RULES
validate_relations = _rel_validator.validate_relations

# Entity resolver (kebab-case filename)
_entity_resolver = import_module(
    ".entity-resolver-for-deduplication", "vn_legal_rag.offline"
)
EntityResolver = _entity_resolver.EntityResolver
EntityMetadata = _entity_resolver.EntityMetadata
ResolverStats = _entity_resolver.ResolverStats
resolve_entity = _entity_resolver.resolve_entity
batch_resolve_entities = _entity_resolver.batch_resolve_entities

# Type systems (enum-based)
from .entity_types import LegalEntityType, LEGAL_ABBREVIATIONS, LEGAL_SYNONYMS
from .relation_types import LegalRelationType, LEGAL_RELATION_TYPES, CROSSREF_PREDICATES
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

# Summary generators
from .summary import (
    ArticleSummary,
    ArticleSummaryCheckpoint,
    ArticleSummaryGenerator,
    ChapterSummary,
    ChapterSummaryCheckpoint,
    ChapterSummaryGenerator,
    DocumentSummary,
    DocumentSummaryCheckpoint,
    DocumentSummaryGenerator,
)

# Scraper (for web scraping new legal documents)
from .scraper import (
    BaseLegalScraper,
    DocumentStatus,
    DocumentType,
    HierarchyExtractor,
    LegalAppendix,
    LegalAppendixItem,
    LegalArticle,
    LegalChapter,
    LegalClause,
    LegalDocument,
    LegalPoint,
    LegalSection,
    TVPLScraper,
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
    # Cross-reference post-processor
    "CrossRefPostProcessor",
    "CrossRefCandidate",
    "ParsedReference",
    "extract_crossrefs_from_relations",
    # Incremental KG updater
    "IncrementalUpdater",
    "UpdateResult",
    # Terminology extractor
    "TerminologyExtractor",
    "ExtractedTerm",
    "TerminologyResult",
    "extract_terminology_from_db",
    # Synonym miner
    "SynonymMiner",
    "SynonymPair",
    "SynonymMiningResult",
    "mine_synonyms_for_domain",
    # KG-SQLite linker
    "KGSQLiteLinker",
    "EntityContext",
    "LinkResult",
    "create_linker",
    # Relation validator
    "RelationValidator",
    "ValidationStats",
    "SEMANTIC_TYPE_RULES",
    "validate_relations",
    # Entity resolver
    "EntityResolver",
    "EntityMetadata",
    "ResolverStats",
    "resolve_entity",
    "batch_resolve_entities",
    # Type systems
    "LegalEntityType",
    "LegalRelationType",
    "LEGAL_RELATION_TYPES",
    "LEGAL_ABBREVIATIONS",
    "LEGAL_SYNONYMS",
    "CROSSREF_PREDICATES",
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
    # Summary generators
    "ChapterSummary",
    "ChapterSummaryCheckpoint",
    "ChapterSummaryGenerator",
    "ArticleSummary",
    "ArticleSummaryCheckpoint",
    "ArticleSummaryGenerator",
    "DocumentSummary",
    "DocumentSummaryCheckpoint",
    "DocumentSummaryGenerator",
    # Scraper (data models for scraped documents)
    "LegalDocument",
    "LegalChapter",
    "LegalSection",
    "LegalArticle",
    "LegalClause",
    "LegalPoint",
    "LegalAppendix",
    "LegalAppendixItem",
    "DocumentStatus",
    "DocumentType",
    "BaseLegalScraper",
    "TVPLScraper",
    "HierarchyExtractor",
]
