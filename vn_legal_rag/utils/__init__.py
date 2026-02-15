"""
Utility modules for Vietnamese legal RAG system.

Exports:
- LLM provider with caching (LLMProvider, create_llm_provider)
- Abbreviation expansion (expand_abbreviations, LEGAL_ABBREVIATIONS)
- Citation formatting (format_citation, parse_citation)
- Text embeddings (EmbeddingProvider, cosine_similarity)
- Config loading (load_config, validate_config)
- Data loaders (load_kg, load_forest, load_summaries)
- Logging setup (setup_logging, get_logger)
"""

from .llm_provider_with_caching import LLMProvider, create_llm_provider
from .vietnamese_abbreviation_expander import (
    expand_abbreviations,
    get_abbreviation_variants,
    expand_search_query,
    LEGAL_ABBREVIATIONS,
)
from .legal_citation_formatter import (
    format_citation,
    parse_citation,
    format_article_citation,
)
from .text_embeddings_provider import (
    EmbeddingProvider,
    create_embedding_provider,
    cosine_similarity,
    cosine_similarity_matrix,
)

# Config and data loading
from importlib import import_module

_config_loader = import_module(".config-loader-with-yaml-support", "vn_legal_rag.utils")
load_config = _config_loader.load_config
validate_config = _config_loader.validate_config
get_config_value = _config_loader.get_config_value
merge_configs = _config_loader.merge_configs
save_config = _config_loader.save_config

_data_loaders = import_module(".data-loaders-for-kg-and-summaries", "vn_legal_rag.utils")
load_kg = _data_loaders.load_kg
load_forest = _data_loaders.load_forest
load_summaries = _data_loaders.load_summaries
load_training_data = _data_loaders.load_training_data
save_json = _data_loaders.save_json
load_json = _data_loaders.load_json
build_forest_from_db = _data_loaders.build_forest_from_db

from .simple_logger import get_logger, setup_logging

# Progress tracker (import from kebab-case filename)
_progress_module = import_module(".simple-progress-tracker-for-testing", "vn_legal_rag.utils")
ProgressTracker = _progress_module.ProgressTracker
get_progress_tracker = _progress_module.get_progress_tracker


__all__ = [
    # LLM provider
    "LLMProvider",
    "create_llm_provider",

    # Abbreviations
    "expand_abbreviations",
    "get_abbreviation_variants",
    "expand_search_query",
    "LEGAL_ABBREVIATIONS",

    # Citations
    "format_citation",
    "parse_citation",
    "format_article_citation",

    # Embeddings
    "EmbeddingProvider",
    "create_embedding_provider",
    "cosine_similarity",
    "cosine_similarity_matrix",

    # Config
    "load_config",
    "validate_config",
    "get_config_value",
    "merge_configs",
    "save_config",

    # Data loaders
    "load_kg",
    "load_forest",
    "load_summaries",
    "load_training_data",
    "save_json",
    "load_json",
    "build_forest_from_db",

    # Logging
    "setup_logging",
    "get_logger",

    # Progress tracker
    "ProgressTracker",
    "get_progress_tracker",
]
