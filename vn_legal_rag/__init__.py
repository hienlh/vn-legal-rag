"""
Vietnamese Legal RAG - 3-Tier Retrieval System

A state-of-the-art Retrieval-Augmented Generation (RAG) system for Vietnamese legal
document question answering, featuring a novel 3-tier retrieval architecture.

Tier 1: Tree Traversal - LLM-guided navigation through document hierarchy
Tier 2: DualLevel Retrieval - 6-component semantic search
Tier 3: Semantic Bridge - RRF-based fusion with KG expansion

Performance (379 Q&A pairs):
- Hit@10: 76.53%
- Recall@5: 58.12%
- MRR: 0.5422

Basic Usage:
    >>> from vn_legal_rag import LegalGraphRAG
    >>> import json
    >>>
    >>> # Load knowledge graph
    >>> with open("data/kg_enhanced/legal_kg.json") as f:
    ...     kg = json.load(f)
    >>>
    >>> # Initialize RAG system
    >>> graphrag = LegalGraphRAG(
    ...     kg=kg,
    ...     db_path="data/legal_docs.db",
    ...     llm_provider="openai",
    ...     llm_model="gpt-4o-mini",
    ... )
    >>>
    >>> # Query
    >>> result = graphrag.query("Điều kiện thành lập công ty cổ phần?")
    >>> print(result.response)
"""

__version__ = "1.0.0"
__author__ = "Vietnamese Legal RAG Team"
__license__ = "MIT"
__all__ = [
    # Version
    "__version__",

    # Main RAG system (online)
    "LegalGraphRAG",
    "GraphRAGResponse",
    "create_legal_graphrag",

    # Retrieval components
    "TreeTraversalRetriever",
    "TreeSearchResult",
    "DualLevelRetriever",
    "DualLevelResult",
    "DualLevelConfig",
    "SemanticBridge",

    # Query analysis
    "VietnameseLegalQueryAnalyzer",
    "AnalyzedQuery",
    "ExpandedQuery",
    "QueryIntent",
    "LegalQueryType",
    "QueryTypeConfig",
    "analyze_query",
    "expand_query",

    # Knowledge graph (offline)
    "UnifiedLegalExtractor",
    "ExtractionResult",
    "IncrementalKGBuilder",
    "IncrementalKGResult",
    "MergedEntity",
    "MergedRelation",

    # Database
    "LegalDocumentDB",
    "LegalDocumentModel",
    "LegalArticleModel",
    "LegalChapterModel",

    # Types
    "LegalEntityType",
    "LegalRelationType",
    "TreeNode",
    "TreeIndex",
    "UnifiedForest",
    "NodeType",

    # Utilities
    "LLMProvider",
    "create_llm_provider",
    "EmbeddingProvider",
    "create_embedding_provider",
    "expand_abbreviations",
    "format_citation",
    "cosine_similarity",

    # PPR
    "PersonalizedPageRank",
    "PPRResult",
    "PPRConfig",

    # Ablation
    "AblationConfig",
    "get_ablation_configs",
    "get_paper_ablation_configs",
]

# =============================================================================
# Main RAG System (online phase)
# =============================================================================

from vn_legal_rag.online import (
    # Main entry point
    LegalGraphRAG,
    GraphRAGResponse,
    create_legal_graphrag,

    # Tier 1: Tree Traversal
    TreeTraversalRetriever,
    TreeSearchResult,
    build_tree_retriever,

    # Tier 2: DualLevel
    DualLevelRetriever,
    DualLevelResult,
    DualLevelConfig,
    LowLevelResult,
    HighLevelResult,

    # Tier 3: Semantic Bridge
    SemanticBridge,
    create_semantic_bridge,

    # Query Analysis
    VietnameseLegalQueryAnalyzer,
    AnalyzedQuery,
    ExpandedQuery,
    QueryIntent,
    LegalQueryType,
    QueryTypeConfig,
    analyze_query,
    expand_query,

    # PPR
    PersonalizedPageRank,
    PPRResult,
    PPRConfig,
)

# =============================================================================
# Knowledge Graph Building (offline phase)
# =============================================================================

from vn_legal_rag.offline import (
    # Database models
    Base,
    LegalDocumentModel,
    LegalChapterModel,
    LegalSectionModel,
    LegalArticleModel,
    LegalClauseModel,
    LegalPointModel,
    LegalCrossReferenceModel,
    LegalAbbreviationModel,

    # Database manager
    LegalDocumentDB,

    # Entity/relation extraction
    UnifiedLegalExtractor,
    ExtractionResult,

    # KG building
    IncrementalKGBuilder,
    IncrementalKGResult,
    MergedEntity,
    MergedRelation,
    slugify_vietnamese,

    # ID helpers
    make_document_id,
    make_chapter_id,
    make_section_id,
    make_article_id,
    make_clause_id,
    make_point_id,
    make_crossref_id,
    normalize_so_hieu,
)

# =============================================================================
# Types
# =============================================================================

from vn_legal_rag.types import (
    # Entity types
    LegalEntityType,
    LEGAL_ENTITY_TYPES,

    # Relation types
    LegalRelationType,

    # Tree models
    TreeNode,
    TreeIndex,
    UnifiedForest,
    NodeType,
    CrossRefEdge,

    # Type mappers
    map_entity_type,
    map_relation_type,
    map_extraction_result,
    STRING_TO_ENTITY_TYPE,
    STRING_TO_RELATION_TYPE,

    # Ablation config
    AblationConfig,
    get_ablation_configs,
    get_paper_ablation_configs,
)

# =============================================================================
# Utilities
# =============================================================================

from vn_legal_rag.utils import (
    # LLM provider
    LLMProvider,
    create_llm_provider,

    # Abbreviations
    expand_abbreviations,
    get_abbreviation_variants,
    expand_search_query,
    LEGAL_ABBREVIATIONS,

    # Citations
    format_citation,
    parse_citation,
    format_article_citation,

    # Embeddings
    EmbeddingProvider,
    create_embedding_provider,
    cosine_similarity,
    cosine_similarity_matrix,
)


# =============================================================================
# Package Metadata
# =============================================================================

def get_version() -> str:
    """Get package version."""
    return __version__


def get_info() -> dict:
    """Get package information."""
    return {
        "name": "vn-legal-rag",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "3-Tier RAG System for Vietnamese Legal Documents",
        "repository": "https://github.com/yourusername/vn_legal_rag",
    }
