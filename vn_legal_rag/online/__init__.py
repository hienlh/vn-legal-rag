"""
Online Phase - 3-Tier Retrieval for Vietnamese Legal RAG

This module provides the online query processing components for Vietnamese legal RAG:
- Tree Traversal Retriever (Tier 1)
- DualLevel Retriever (Tier 2)
- Semantic Bridge (Tier 3)
- Query Analysis & Expansion
- Personalized PageRank for KG

Architecture:
    Query → Analyzer → Tree + DualLevel (parallel) → Semantic Bridge → Response

Performance (current):
    Hit Rate: 78.93%, Recall@5: 58.12%, MRR: 0.5422
"""

# Main entry point
from importlib import import_module

_legal_graphrag = import_module(".legal-graphrag-3tier-query-engine", "vn_legal_rag.online")
LegalGraphRAG = _legal_graphrag.LegalGraphRAG
GraphRAGResponse = _legal_graphrag.GraphRAGResponse
create_legal_graphrag = _legal_graphrag.create_legal_graphrag

# Tree Traversal (Tier 1)
_tree_retriever = import_module(".tree-traversal-retriever", "vn_legal_rag.online")
TreeTraversalRetriever = _tree_retriever.TreeTraversalRetriever
TreeSearchResult = _tree_retriever.TreeSearchResult
build_tree_retriever = _tree_retriever.build_tree_retriever

# DualLevel Retriever (Tier 2)
_dual_retriever = import_module(".dual-level-retriever", "vn_legal_rag.online")
DualLevelRetriever = _dual_retriever.DualLevelRetriever
DualLevelResult = _dual_retriever.DualLevelResult
DualLevelConfig = _dual_retriever.DualLevelConfig
LowLevelResult = _dual_retriever.LowLevelResult
HighLevelResult = _dual_retriever.HighLevelResult

# Semantic Bridge (Tier 3)
_semantic_bridge = import_module(".semantic-bridge-rrf-merger", "vn_legal_rag.online")
SemanticBridge = _semantic_bridge.SemanticBridge
create_semantic_bridge = _semantic_bridge.create_semantic_bridge

# Query Analysis
_query_analyzer = import_module(".vietnamese-legal-query-analyzer", "vn_legal_rag.online")
VietnameseLegalQueryAnalyzer = _query_analyzer.VietnameseLegalQueryAnalyzer
AnalyzedQuery = _query_analyzer.AnalyzedQuery
ExpandedQuery = _query_analyzer.ExpandedQuery
QueryIntent = _query_analyzer.QueryIntent
LegalQueryType = _query_analyzer.LegalQueryType
QueryTypeConfig = _query_analyzer.QueryTypeConfig
expand_query = _query_analyzer.expand_query
analyze_query = _query_analyzer.analyze_query

# Personalized PageRank
_ppr = import_module(".personalized-page-rank-for-kg", "vn_legal_rag.online")
PersonalizedPageRank = _ppr.PersonalizedPageRank
PPRResult = _ppr.PPRResult
PPRConfig = _ppr.PPRConfig

# Ontology Expander (Query Expansion)
_ontology = import_module(".ontology-based-query-expander", "vn_legal_rag.online")
OntologyExpander = _ontology.OntologyExpander
ExpansionResult = _ontology.ExpansionResult

# Document Filter (Result Filtering)
_doc_filter = import_module(".document-aware-result-filter", "vn_legal_rag.online")
DocumentFilter = _doc_filter.DocumentFilter
DocumentHint = _doc_filter.DocumentHint
FilteredResult = _doc_filter.FilteredResult

# Cross-Encoder Reranker
_reranker = import_module(".cross-encoder-reranker-for-legal-documents", "vn_legal_rag.online")
CrossEncoderReranker = _reranker.CrossEncoderReranker
RerankResult = _reranker.RerankResult


__all__ = [
    # Main
    "LegalGraphRAG",
    "GraphRAGResponse",
    "create_legal_graphrag",

    # Tree Retrieval
    "TreeTraversalRetriever",
    "TreeSearchResult",
    "build_tree_retriever",

    # DualLevel Retrieval
    "DualLevelRetriever",
    "DualLevelResult",
    "DualLevelConfig",
    "LowLevelResult",
    "HighLevelResult",

    # Semantic Bridge
    "SemanticBridge",
    "create_semantic_bridge",

    # Query Analysis
    "VietnameseLegalQueryAnalyzer",
    "AnalyzedQuery",
    "ExpandedQuery",
    "QueryIntent",
    "LegalQueryType",
    "QueryTypeConfig",
    "expand_query",
    "analyze_query",

    # PPR
    "PersonalizedPageRank",
    "PPRResult",
    "PPRConfig",

    # Ontology Expander
    "OntologyExpander",
    "ExpansionResult",

    # Document Filter
    "DocumentFilter",
    "DocumentHint",
    "FilteredResult",

    # Cross-Encoder Reranker
    "CrossEncoderReranker",
    "RerankResult",
]
