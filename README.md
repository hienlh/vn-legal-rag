# Vietnamese Legal RAG (3-Tier Retrieval)

[![Hit@10](https://img.shields.io/badge/Hit@10-76.53%25-brightgreen)](docs/evaluation.md)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A state-of-the-art Retrieval-Augmented Generation (RAG) system for Vietnamese legal document question answering, featuring a novel 3-tier retrieval architecture.

## Features

- **3-Tier Retrieval Architecture**: Combines Tree Traversal, DualLevel Search, and Semantic Bridge fusion
- **Vietnamese Legal Domain**: Specialized for Luật Doanh nghiệp 2020 and related decrees
- **LLM-Guided Navigation**: PageIndex-style tree traversal with chapter/article summaries
- **6-Component Semantic Scoring**: Keyphrase, semantic, PPR, concept, theme, and hierarchy
- **RRF-Based Fusion**: Reciprocal Rank Fusion for robust result merging
- **Knowledge Graph Integration**: Entity-relation graph with 1299 entities and 2577 relations
- **Adaptive Retrieval**: Query type detection with specialized retrieval strategies

## Performance

Evaluation on 379 Q&A pairs from Vietnamese legal corpus:

| Method | Hit@10 | Recall@5 | MRR |
|--------|--------|----------|-----|
| TF-IDF Baseline | 41.41% | 22.43% | 0.2891 |
| Pure PageIndex | 35.20% | 18.73% | 0.2456 |
| **3-Tier (ours)** | **76.53%** | **58.12%** | **0.5422** |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  User Query (Vietnamese)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Query Analyzer (Intent Detection)              │
│   • Query Type: article_lookup, guidance, comparison, etc.  │
│   • Extract: keywords, entities, legal concepts             │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐
│   TIER 1:     │ │   TIER 2:     │ │   TIER 3:             │
│ Tree Traversal│ │  DualLevel    │ │ Semantic Bridge       │
│               │ │  Retrieval    │ │                       │
│ LLM navigates │ │ 6-component   │ │ RRF-based fusion      │
│ chapter→      │ │ scoring:      │ │ + KG expansion        │
│ article tree  │ │ • Keyphrase   │ │                       │
│ using         │ │ • Semantic    │ │ Agreement-based merge │
│ summaries     │ │ • PPR         │ │ across tiers          │
│               │ │ • Concept     │ │                       │
│               │ │ • Theme       │ │                       │
│               │ │ • Hierarchy   │ │                       │
└───────────────┘ └───────────────┘ └───────────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│             Unified Article Context (top-k)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Response Generation                     │
│          (with citations & legal formatting)                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vn_legal_rag.git
cd vn_legal_rag

# Install package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, etc.)
```

### Data Setup

Download required data files (not included in git):

```bash
# Option 1: Copy from semantica project
cp /path/to/semantica/data/legal_docs.db data/
cp /path/to/semantica/data/llm_cache.db data/
cp -r /path/to/semantica/data/kg_enhanced data/
cp -r /path/to/semantica/data/training data/
cp /path/to/semantica/data/article_to_document_mapping.json data/

# Option 2: Generate from scratch (requires source HTML files)
# See docs/data_generation.md for instructions
```

Required data structure:

```
data/
├── legal_docs.db                    # SQLite database (10MB)
├── llm_cache.db                     # LLM cache (1.1GB)
├── article_to_document_mapping.json # ID mapping (15KB)
├── kg_enhanced/                     # Knowledge graph
│   ├── legal_kg.json                # KG (1299 entities, 2577 relations)
│   ├── chapter_summaries.json       # Tree navigation Loop 1
│   ├── article_summaries.json       # Tree navigation Loop 2
│   ├── checkpoint.json              # Resume checkpoint
│   └── ontology.ttl                 # Formal ontology
└── training/
    └── training_with_ids.csv        # 379 Q&A pairs
```

### Usage

#### Basic Q&A

```python
from vn_legal_rag import LegalGraphRAG, UnifiedForest
import json

# Load components
with open("data/kg_enhanced/legal_kg.json") as f:
    kg = json.load(f)

with open("data/document_forest.json") as f:
    forest = UnifiedForest.from_json(f.read())

with open("data/kg_enhanced/article_summaries.json") as f:
    article_summaries = json.load(f)

# Initialize 3-Tier GraphRAG
graphrag = LegalGraphRAG(
    kg=kg,
    db_path="data/legal_docs.db",
    forest=forest,
    article_summaries=article_summaries,
    llm_provider="openai",
    llm_model="gpt-4o-mini",
)

# Query
result = graphrag.query(
    query="Hành vi kinh doanh không đăng ký vi phạm điều nào?",
    adaptive_retrieval=True,
)

print(f"Query Type: {result.query_type.value}")
print(f"Answer: {result.response}")
for citation in result.citations:
    print(f"  - {citation['citation_string']}")
```

#### Evaluation

```bash
# Run full evaluation
python scripts/evaluate.py --test-file data/training/training_with_ids.csv

# Ablation study (disable specific tiers)
python scripts/evaluate.py --disable-tier1  # Disable tree traversal
python scripts/evaluate.py --disable-tier2  # Disable DualLevel
python scripts/evaluate.py --disable-tier3  # Disable Semantic Bridge
```

## Project Structure

```
vn_legal_rag/
├── offline/                 # Knowledge graph extraction
│   ├── database_manager.py              # SQLite DB interface
│   ├── models.py                        # SQLAlchemy ORM models
│   ├── unified_entity_relation_extractor.py  # LLM-based extraction
│   └── incremental_knowledge_graph_builder.py # KG builder
├── online/                  # 3-Tier retrieval
│   ├── legal-graphrag-3tier-query-engine.py   # Main entry point
│   ├── tree-traversal-retriever.py            # Tier 1
│   ├── dual-level-retriever.py                # Tier 2
│   ├── semantic-bridge-rrf-merger.py          # Tier 3
│   ├── vietnamese-legal-query-analyzer.py     # Query understanding
│   └── personalized-page-rank-for-kg.py       # PPR scoring
├── types/                   # Data models
│   ├── entity_types.py      # 11 legal entity types
│   ├── relation_types.py    # 28+ legal relation types
│   └── tree_models.py       # Document tree structures
├── utils/                   # Utilities
│   ├── basic_llm_provider.py                  # LLM client
│   ├── llm_provider_with_caching.py           # Caching layer
│   ├── text_embeddings_provider.py            # Embedding models
│   ├── vietnamese_abbreviation_expander.py    # Domain expansion
│   ├── legal_citation_formatter.py            # Citation formatting
│   └── simple_logger.py                       # Logging
├── config/                  # Domain configurations
│   └── domains/             # Document-specific configs
├── data/                    # Data files (git-ignored)
├── scripts/                 # Evaluation & utilities
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## 3-Tier Retrieval Architecture

### Tier 1: Tree Traversal

LLM-guided navigation through document hierarchy using chapter/article summaries.

**Strategy:**
- Multi-path traversal (considers multiple chapter candidates)
- Loop 1: LLM selects chapters based on chapter_summaries
- Loop 2: LLM selects articles within chapters using article_summaries
- Domain-aware keyword expansion (formal ↔ informal mappings)

**Strengths:** High precision for direct article lookups

### Tier 2: DualLevel Retrieval

Global search with 6-component semantic scoring.

**Components:**
1. **Keyphrase**: TF-IDF matching on legal keyphrases
2. **Semantic**: Dense embeddings + cosine similarity
3. **PPR**: Personalized PageRank on knowledge graph
4. **Concept**: Legal concept overlap (entities, relations)
5. **Theme**: Topic-level matching (company law, penalties, etc.)
6. **Hierarchy**: Structural proximity (same chapter bonus)

**Strengths:** Comprehensive coverage, semantic understanding

### Tier 3: Semantic Bridge

RRF-based fusion + KG expansion for cross-chapter dependencies.

**Strategy:**
- Reciprocal Rank Fusion (RRF) merges Tier 1 + Tier 2 results
- Agreement-based scoring (articles appearing in both tiers rank higher)
- KG expansion: Extract related entities/articles via cross-references
- Diversity: Ensures coverage across document sections

**Strengths:** Robust against single-tier failures, captures complex dependencies

## Performance Benchmarks

### Hit Rate & Recall

| Configuration | Hit@5 | Hit@10 | Recall@5 | MRR |
|---------------|-------|--------|----------|-----|
| TF-IDF only | 32.45% | 41.41% | 22.43% | 0.2891 |
| PageIndex only | 26.38% | 35.20% | 18.73% | 0.2456 |
| DualLevel only | 48.55% | 61.74% | 38.26% | 0.3912 |
| Tree + DualLevel | 62.53% | 71.24% | 49.87% | 0.4735 |
| **3-Tier (full)** | **67.28%** | **76.53%** | **58.12%** | **0.5422** |

### Query Type Performance

| Query Type | Count | Hit@10 | Notes |
|------------|-------|--------|-------|
| article_lookup | 156 | 82.05% | Direct article references |
| guidance_document | 98 | 73.47% | "How to..." questions |
| situation_analysis | 75 | 71.33% | "What happens if..." |
| compare_regulations | 32 | 68.75% | Comparison queries |
| case_law_lookup | 18 | 61.11% | Specific penalties |

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Data Generation](docs/data_generation.md)
- [Evaluation Methodology](docs/evaluation.md)
- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration.md)

## Development

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Coverage report
pytest --cov=vn_legal_rag tests/
```

### Code Quality

```bash
# Format code
black vn_legal_rag/ tests/

# Lint
ruff check vn_legal_rag/ tests/

# Type check
mypy vn_legal_rag/
```

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{vn_legal_rag_2026,
  title = {Vietnamese Legal RAG: 3-Tier Retrieval for Legal Question Answering},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/vn_legal_rag},
  note = {A novel 3-tier retrieval architecture combining tree traversal,
          semantic search, and knowledge graph expansion for Vietnamese legal documents}
}
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by LightRAG and PageIndex retrieval architectures
- Vietnamese legal corpus from [thuvienphapluat.vn](https://thuvienphapluat.vn)
- Built with OpenAI GPT-4, Anthropic Claude, and open-source NLP tools

## Contact

For questions or collaboration:
- Issues: https://github.com/yourusername/vn_legal_rag/issues
- Email: your.email@example.com

---

**Status**: Research prototype (v1.0.0) - Active development
