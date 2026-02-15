"""
Offline phase modules for Vietnamese Legal RAG.

This package contains modules for offline processing:
- Database models and management
- Entity/relation extraction
- Knowledge graph building
- Entity deduplication
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
]
