# Code Standards: Testing, Dependencies & Security

**Project**: Vietnamese Legal RAG (vn_legal_rag)
**Last Updated**: 2026-02-14
**Related**: [Code Standards (Main)](./code-standards.md)

## Testing Standards

### Test Location

```
tests/
├── test-online-module-imports.py       # Module import verification
└── integration/                         # Integration tests
    └── test-end-to-end-legal-qa.py
```

### Test Naming

```python
# ✓ Correct
def test_tier1_retrieves_articles_for_article_lookup_query():
    ...

def test_dual_level_scorer_weights_six_components():
    ...

def test_rrf_fusion_handles_empty_tier1_results():
    ...

# ✗ Incorrect
def test_tier1():
    ...

def test_dual():
    ...
```

### Pytest Conventions

```python
# ✓ Correct
import pytest
from vn_legal_rag.online import TreeTraversalRetriever

@pytest.fixture
def mock_forest():
    """Load test forest."""
    ...

def test_tree_traversal_selects_chapters(mock_forest):
    """Test chapter selection in Loop 1."""
    retriever = TreeTraversalRetriever(mock_forest)
    result = retriever.retrieve("query")

    assert len(result.chapters) > 0
    assert all(c.type == "chapter" for c in result.chapters)

@pytest.mark.parametrize("query_type,expected_tier", [
    ("article_lookup", "tier1"),
    ("guidance", "tier2"),
    ("comparison", "tier3"),
])
def test_adaptive_routing(query_type, expected_tier):
    """Test adaptive tier selection."""
    ...
```

## Import Organization

**Rule 1: Import Order**

```python
# Standard library imports
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
from sqlalchemy import Column, Integer, String

# Local imports
from vn_legal_rag.types import LegalEntityType, LegalQueryType
from vn_legal_rag.utils import SimpleLogger, create_llm_provider

# Conditional imports (if needed)
try:
    import openai
except ImportError:
    openai = None  # type: ignore
```

**Rule 2: Relative Imports Within Package**

```python
# ✓ Correct (relative imports within package)
from .models import Article, Chapter
from ..types import LegalEntityType
from ..utils import SimpleLogger

# ✗ Incorrect (absolute imports within same package)
from vn_legal_rag.offline.models import Article
```

## Documentation Standards

### Module Docstring

Every module should have a top-level docstring:

```python
"""
Tree Traversal Retriever for Tier 1 Legal Document Navigation.

This module implements LLM-guided navigation through Vietnamese legal
document hierarchy using chapter and article summaries.

Structure:
- TreeTraversalRetriever: Main class for tree navigation
- TreeSearchResult: Result structure from tree search
- _loop_1_select_chapters: Helper for chapter selection
- _loop_2_select_articles: Helper for article selection

Example:
    >>> retriever = TreeTraversalRetriever(forest, llm_provider)
    >>> result = retriever.retrieve("Điều 5 quy định gì?")
    >>> print(result.articles)
"""
```

### Function Docstring

Google style with all sections:

```python
def retrieve(
    self,
    query: str,
    top_k: int = 10,
    multi_path: bool = True
) -> TreeSearchResult:
    """
    Retrieve articles via tree traversal.

    Executes Loop 1 (chapter selection) and Loop 2 (article selection)
    guided by LLM using document summaries.

    Args:
        query: Vietnamese legal question
        top_k: Number of candidate chapters to consider in Loop 1
        multi_path: Whether to explore multiple paths

    Returns:
        TreeSearchResult with selected chapters and articles

    Raises:
        ValueError: If query is empty
        LLMError: If LLM API call fails

    Example:
        >>> result = retriever.retrieve("Phạt bao nhiêu nếu vi phạm?")
        >>> for article in result.articles:
        ...     print(f"{article.id}: {article.name}")
    """
    ...
```

### Inline Comments

Only use inline comments for non-obvious logic:

```python
# ✓ Correct: Explains why, not what
# Boost articles appearing in both tier1 and tier2 results
# to increase confidence in agreement
agreement_score = tier1_rank + tier2_rank + AGREEMENT_BOOST

# ✗ Incorrect: Obvious from code
rank = tier1_rank + tier2_rank  # Add two ranks
```

## Dependencies

### Core Dependencies (Required)

```
sqlalchemy>=2.0          # ORM for document DB
numpy>=1.20             # Numerical operations
scipy>=1.7              # Scientific computing (PPR)
networkx>=2.6           # Graph algorithms (KG traversal)
pyyaml>=6.0             # Config file parsing
```

### LLM Provider Dependencies (Optional)

```
openai>=1.0             # OpenAI API
anthropic>=0.7          # Anthropic Claude API
requests>=2.28          # HTTP client (local models)
```

### Development Dependencies

```
pytest>=7.0             # Testing
black>=23.0             # Code formatting
ruff>=0.1               # Linting
mypy>=1.0               # Type checking
```

## Performance Considerations

### Optimization Guidelines

1. **Caching**: Use llm_provider_with_caching.py for LLM responses
2. **Batch Operations**: Load embeddings in batches, not one-by-one
3. **Index Usage**: SQLite indexes on frequently queried columns
4. **Lazy Loading**: Load summaries only when needed (Tier 1/2)

### Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... code to profile ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Backwards Compatibility

### Versioning

Project uses semantic versioning: MAJOR.MINOR.PATCH

- **MAJOR**: Breaking changes (e.g., API signature change)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Deprecation

When removing functionality:

```python
import warnings

def old_function():
    """Deprecated: Use new_function instead."""
    warnings.warn(
        "old_function is deprecated. Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    ...
```

## Security Guidelines

### Data Handling

1. **No Query Logging**: Never log user queries or sensitive content
2. **API Key Protection**: Use .env, never commit keys
3. **SQL Injection**: Always use SQLAlchemy ORM (parameterized queries)
4. **Input Validation**: Validate query length, format

### Secret Management

```python
# ✓ Correct
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# ✗ Incorrect
API_KEY = "sk-xxxxx"  # Never hardcode!
```

## Summary

| Topic | Key Points |
|-------|------------|
| Testing | pytest, descriptive names, parametrized tests, ≥70% coverage |
| Imports | stdlib → third-party → local; relative within package |
| Docs | Google-style docstrings; module + function + inline |
| Deps | Core (sqlalchemy, numpy), LLM (openai, anthropic), Dev (pytest, black) |
| Performance | Caching, batching, indexes, lazy loading |
| Security | No hardcoded keys, parameterized queries, input validation |

---

**Last Updated**: 2026-02-14
**Version**: 1.0
