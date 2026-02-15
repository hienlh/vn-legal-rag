# Code Standards & Codebase Structure Guide

**Project**: Vietnamese Legal RAG (vn_legal_rag)
**Last Updated**: 2026-02-14
**Python Version**: 3.10+

## Directory Structure

```
vn_legal_rag/
├── offline/                          # Knowledge Graph Extraction Pipeline
│   ├── __init__.py                   # Public API exports
│   ├── models.py                     # SQLAlchemy ORM models (~250 LOC)
│   ├── database_manager.py           # SQLite CRUD interface (~300 LOC)
│   ├── unified_entity_relation_extractor.py    # LLM-based NER/RE (~400 LOC)
│   └── incremental_knowledge_graph_builder.py  # KG builder with checkpoints (~350 LOC)
│
├── online/                           # 3-Tier Online Retrieval Engine
│   ├── __init__.py                   # Public API exports
│   ├── legal-graphrag-3tier-query-engine.py         # Main entry (950 LOC)
│   ├── tree-traversal-retriever.py                  # Tier 1 (600 LOC)
│   ├── dual-level-retriever.py                      # Tier 2 (800 LOC)
│   ├── semantic-bridge-rrf-merger.py                # Tier 3 (500 LOC)
│   ├── vietnamese-legal-query-analyzer.py           # Query processing (400 LOC)
│   ├── personalized-page-rank-for-kg.py             # PPR scoring (200 LOC)
│   ├── cross-encoder-reranker-for-legal-documents.py  # Reranking (150 LOC)
│   ├── document-aware-result-filter.py              # Post-filtering (100 LOC)
│   └── ontology-based-query-expander.py             # Query expansion (150 LOC)
│
├── types/                            # Data Models & Type Definitions
│   ├── __init__.py                   # Public API exports
│   ├── entity_types.py               # LegalEntityType enum (~50 LOC)
│   ├── relation_types.py             # LegalRelationType enum (~70 LOC)
│   ├── tree_models.py                # TreeNode, CrossRefEdge, UnifiedForest (~200 LOC)
│   └── ablation-config-for-rag-component-testing.py  # AblationConfig (~80 LOC)
│
├── utils/                            # Utility Functions & Providers
│   ├── __init__.py                   # Public API exports + factory functions
│   ├── basic_llm_provider.py         # BaseLLMProvider abstract class (~250 LOC)
│   ├── llm_provider_with_caching.py  # LLM caching wrapper (~180 LOC)
│   ├── text_embeddings_provider.py   # Embedding generation (~180 LOC)
│   ├── vietnamese_abbreviation_expander.py  # Abbreviation expansion (~130 LOC)
│   ├── legal_citation_formatter.py   # Citation formatting (~100 LOC)
│   ├── simple_logger.py              # Structured logging (~60 LOC)
│   ├── config-loader-with-yaml-support.py  # YAML config loading (~150 LOC)
│   ├── data-loaders-for-kg-and-summaries.py  # Data loading utilities (~200 LOC)
│   └── simple-progress-tracker-for-testing.py  # Progress tracking (~60 LOC)
│
├── config/                           # Domain-Specific Configurations
│   ├── default.yaml                  # Template configuration
│   └── domains/                      # One YAML file per legal decree
│       ├── 59-2020-QH14.yaml         # Luật Doanh nghiệp 2020
│       ├── 01-2021-ND.yaml           # Nghị định 01/2021/NĐ-CP
│       ├── 16-2023-ND.yaml, etc.
│
├── __init__.py                       # Package root; main public API
└── py.typed                          # PEP 561 marker for type hints
```

## File Naming Conventions

### Module Names

**Rule 1: Kebab-Case for Multi-Word Modules**

Use kebab-case (hyphens) when a module name contains multiple words that form a semantic unit describing tier-specific or complex functionality.

```python
# ✓ Correct: kebab-case for tier/feature modules
tree-traversal-retriever.py              # Tier 1 retrieval
dual-level-retriever.py                  # Tier 2 retrieval
semantic-bridge-rrf-merger.py            # Tier 3 fusion
vietnamese-legal-query-analyzer.py       # Query processing
personalized-page-rank-for-kg.py         # PPR algorithm
cross-encoder-reranker-for-legal-documents.py  # Reranking
document-aware-result-filter.py          # Post-filtering
ontology-based-query-expander.py         # Query expansion
```

**Rule 2: Snake_Case for Utilities**

Use snake_case (underscores) for smaller utility modules providing general-purpose functionality.

```python
# ✓ Correct: snake_case for utilities
basic_llm_provider.py                    # LLM client interface
llm_provider_with_caching.py             # Caching wrapper
text_embeddings_provider.py              # Embeddings
simple_logger.py                         # Logging
entity_types.py                          # Type enumerations
relation_types.py                        # Type enumerations
tree_models.py                           # Data structures
models.py                                # ORM models
```

**Rule 3: Long, Self-Documenting Names**

Names should be long enough to be self-documenting. Prefer clarity over brevity.

```python
# ✓ Correct: Clear purpose
cross-encoder-reranker-for-legal-documents.py

# ✗ Incorrect: Ambiguous
reranker.py
ce_ranker.py
```

**Import Pattern for Kebab-Case Modules**

```python
from importlib import import_module

# Import from kebab-case filename
_module = import_module(".tree-traversal-retriever", "vn_legal_rag.online")
TreeTraversalRetriever = _module.TreeTraversalRetriever
TreeSearchResult = _module.TreeSearchResult

# Alternative: Direct import if kebab-case doesn't contain hyphen
from vn_legal_rag.offline import LegalDocumentDB
```

## Code Style Guide

### Style Standards

**Formatter**: Black (line length: 88 characters)
**Linter**: Ruff (with configuration in pyproject.toml)
**Type Checker**: MyPy (strict mode for new code)

```bash
# Format code
black vn_legal_rag/ tests/

# Lint
ruff check vn_legal_rag/ tests/ --fix

# Type check
mypy vn_legal_rag/ --strict
```

### Python Conventions

**Rule 1: Type Hints (Mandatory for Public APIs)**

All public functions must have complete type hints.

```python
# ✓ Correct
def query(
    self,
    query_text: str,
    adaptive_retrieval: bool = True,
    top_k: int = 10
) -> GraphRAGResponse:
    """
    Execute a legal query through the 3-tier retrieval system.

    Args:
        query_text: Vietnamese legal question
        adaptive_retrieval: Enable adaptive tier selection by query type
        top_k: Number of final results to return

    Returns:
        GraphRAGResponse with answer, citations, and reasoning path
    """
    ...

# ✗ Incorrect (no type hints)
def query(self, query_text, adaptive_retrieval=True, top_k=10):
    ...
```

**Rule 2: Dataclasses for Data Models**

Use dataclasses (Python 3.10+ feature) for structured data.

```python
# ✓ Correct
from dataclasses import dataclass, field

@dataclass
class GraphRAGResponse:
    response: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    confidence: float = 0.0
    query_type: LegalQueryType = LegalQueryType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)

# ✗ Incorrect (legacy class with __init__)
class GraphRAGResponse:
    def __init__(self, response, citations=None, ...):
        ...
```

**Rule 3: Enums for Constants**

Use Enum for fixed sets of string/int constants.

```python
# ✓ Correct
from enum import Enum

class LegalQueryType(str, Enum):
    ARTICLE_LOOKUP = "article_lookup"
    GUIDANCE_DOCUMENT = "guidance_document"
    SITUATION_ANALYSIS = "situation_analysis"
    COMPARE_REGULATIONS = "compare_regulations"
    CASE_LAW_LOOKUP = "case_law_lookup"
    GENERAL = "general"

# Usage: LegalQueryType.ARTICLE_LOOKUP.value

# ✗ Incorrect (magic strings)
QUERY_TYPE_ARTICLE = "article_lookup"
QUERY_TYPE_GUIDANCE = "guidance_document"
```

**Rule 4: Docstrings (Google Style)**

All public functions and classes require docstrings in Google style.

```python
# ✓ Correct
def get_article(self, article_id: str) -> Optional[Article]:
    """
    Retrieve article text and metadata from database.

    Queries the legal_docs.db SQLite database by article ID.
    Returns None if article not found.

    Args:
        article_id: Article identifier in format "doc:d5" (article 5 of doc)

    Returns:
        Article object with text, metadata, and nested clauses.
        None if article not found in database.

    Raises:
        DatabaseError: If database connection fails.
        ValueError: If article_id format is invalid.

    Example:
        >>> db = LegalDocumentDB("data/legal_docs.db")
        >>> article = db.get_article("59-2020-QH14:d5")
        >>> print(article.content)
    """
    ...

# ✗ Incorrect (no docstring)
def get_article(self, article_id: str) -> Optional[Article]:
    ...
```

**Rule 5: Error Handling**

Use specific exception types. Always include context.

```python
# ✓ Correct
try:
    article = db.get_article(article_id)
except DatabaseError as e:
    logger.error(f"Database error fetching article {article_id}: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid article ID format: {article_id}: {e}")
    return None

# ✗ Incorrect (bare except)
try:
    article = db.get_article(article_id)
except:
    return None
```

**Rule 6: Logging**

Use SimpleLogger for structured logging.

```python
# ✓ Correct
from vn_legal_rag.utils import SimpleLogger

logger = SimpleLogger(__name__)
logger.info(f"Processing query: {query_text}")
logger.debug(f"Tier 1 results: {tier1_results}")
logger.warning(f"Low confidence: {confidence}")
logger.error(f"Failed to retrieve: {error}")

# ✗ Incorrect (print statements)
print(f"Processing query: {query_text}")
```

**Rule 7: Constants**

Define module-level constants in UPPERCASE.

```python
# ✓ Correct
DEFAULT_TOP_K = 10
MAX_QUERY_LENGTH = 1000
CACHE_EXPIRATION_HOURS = 24
RRF_K = 60  # RRF fusion parameter

# ✗ Incorrect (magic numbers in code)
if len(query) > 1000:
    ...
```

## Module-Specific Guidelines

### Offline Module (`offline/`)

**Purpose**: Knowledge graph extraction from legal documents

**Responsibilities**:
- Load legal documents (HTML/text)
- Extract entities and relations via LLM
- Build knowledge graph with checkpoints
- Store in SQLite and JSON formats

**Key Guidelines**:
1. All database operations go through LegalDocumentDB class
2. ORM models in models.py use SQLAlchemy declarative syntax
3. Entity/relation extraction via UnifiedEntityRelationExtractor
4. Support checkpoint resume in IncrementalKGBuilder

**Example: Adding a New Extraction Type**

```python
# In offline/__init__.py
from .incremental_knowledge_graph_builder import (
    IncrementalKGBuilder,
)

# Add to __all__
__all__ = [
    "LegalDocumentDB",
    "IncrementalKGBuilder",
    "UnifiedEntityRelationExtractor",
]
```

### Online Module (`online/`)

**Purpose**: 3-tier retrieval engine for query processing

**Responsibilities**:
- Analyze query intent (Tier 0)
- Tree traversal retrieval (Tier 1)
- Semantic search (Tier 2)
- RRF fusion (Tier 3)
- Generate citations

**Key Guidelines**:
1. Each tier independently functional
2. All tiers work with same underlying data (KG, summaries)
3. Pluggable LLM providers via dependency injection
4. Support ablation configuration (disable tiers)

**Example: Adding a New Tier**

```python
# New file: vn_legal_rag/online/new-tier-retriever.py

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class NewTierResult:
    """Result from new tier."""
    article_ids: List[str]
    scores: List[float]
    reasoning: List[str]

class NewTierRetriever:
    """New retrieval tier implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def retrieve(self, query: str, kg: Dict) -> NewTierResult:
        """Execute retrieval."""
        ...

# In legal-graphrag-3tier-query-engine.py
from importlib import import_module
_new_tier = import_module(".new-tier-retriever", "vn_legal_rag.online")
NewTierRetriever = _new_tier.NewTierRetriever
```

### Types Module (`types/`)

**Purpose**: Shared data models and enumerations

**Responsibilities**:
- Define entity types (11 legal entity types)
- Define relation types (28+ legal relation types)
- Define document tree structures
- Define configuration objects

**Key Guidelines**:
1. Enums for all type classifications
2. Dataclasses for all data structures
3. Immutable when possible
4. Version enums when adding new types

**Example: Adding a New Entity Type**

```python
# In types/entity_types.py

class LegalEntityType(Enum):
    """Entity types for Vietnamese legal documents."""

    # ... existing types ...

    # NEW: Add new type at end
    SANCTION = "SANCTION"  # Biện pháp xử lý (new)

# Update list export
LEGAL_ENTITY_TYPES: List[str] = [e.value for e in LegalEntityType]
```

### Utils Module (`utils/`)

**Purpose**: Reusable utility functions and providers

**Responsibilities**:
- LLM provider abstraction
- Embedding generation
- Configuration loading
- Citation formatting
- Logging

**Key Guidelines**:
1. Provider classes use abstract base class pattern
2. Factory functions for object creation
3. Wrap external dependencies (openai, anthropic)
4. Cache expensive operations

**Example: Adding a New Provider**

```python
# In utils/basic_llm_provider.py

from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        pass

# In utils/__init__.py

def create_llm_provider(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    cache_db: Optional[str] = None
) -> BaseLLMProvider:
    """Factory function for LLM providers."""
    if provider == "openai":
        return OpenAIProvider(model)
    elif provider == "anthropic":
        return AnthropicProvider(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

## Configuration Management

### Domain YAML Configuration

Each legal decree has a config file in `config/domains/`:

```yaml
# config/domains/59-2020-QH14.yaml

name: 59-2020-QH14
description: Luật Doanh nghiệp 2020
language: vi

abbreviations:
  TGĐ: Tổng Giám đốc
  TNHH: Trách nhiệm hữu hạn
  HĐQT: Hội đồng quản trị
  ĐHĐCĐ: Đại hội đồng cổ đông

synonyms:
  hợp đồng: khế ước
  tài sản: tài sản riêng

topic_hints:
  registration: "đăng ký doanh nghiệp"
  structure: "cơ cấu quản lý"

article_selection_examples:
  - query: "Hành vi kinh doanh không đăng ký vi phạm điều nào?"
    article: "59-2020-QH14:d5"
```

**Usage in Code**:

```python
from vn_legal_rag.utils import ConfigLoader

loader = ConfigLoader()
config = loader.load("config/domains/59-2020-QH14.yaml")
abbreviations = config.get("abbreviations", {})
```

## Summary

**Key Takeaways**:

1. **Naming**: Kebab-case for complex multi-word modules, snake_case for utilities
2. **Types**: Mandatory type hints on public APIs; use dataclasses for data models
3. **Modularity**: Each tier independent; pluggable LLM providers
4. **Configuration**: YAML domain configs for customization
5. **Code Quality**: Black + Ruff + MyPy; strict type checking

**Additional Topics**: See [Testing, Dependencies & Security Guide](./code-standards-testing-deps-security-guide.md) for:
- Testing standards (pytest conventions, naming)
- Import organization & documentation standards
- Dependencies (core, LLM, dev)
- Performance & security guidelines

---

**Last Updated**: 2026-02-14
**Version**: 1.0
