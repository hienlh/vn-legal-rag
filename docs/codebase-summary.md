# Codebase Summary

**Project**: Vietnamese Legal RAG (vn_legal_rag)
**Total Lines of Code**: ~8,446 LOC across 36 Python modules
**Total Tokens**: 685,634 tokens (via repomix)

## High-Level Overview

Vietnamese Legal RAG implements a state-of-the-art 3-tier retrieval system for Vietnamese legal document question answering. The system combines tree-based navigation, semantic search, and knowledge graph expansion to achieve 76.53% Hit@10 on a legal Q&A evaluation set.

**Key Performance**:
- Hit@10: 76.53% (vs. 41.41% TF-IDF baseline)
- MRR: 0.5422 (vs. 0.2891 baseline)
- Hit@5: 67.28%, Recall@5: 58.12%

## Module Organization

### 1. Offline Module (`vn_legal_rag/offline/`)

Handles knowledge graph extraction and document structure preprocessing.

| File | Purpose | Key Classes |
|------|---------|-------------|
| `models.py` | SQLAlchemy ORM models for document hierarchy | `Document`, `Chapter`, `Section`, `Article`, `Clause`, `Point`, `CrossReference` |
| `database_manager.py` | SQLite database interface for CRUD operations | `LegalDocumentDB` |
| `unified_entity_relation_extractor.py` | LLM-based entity/relation extraction | `UnifiedEntityRelationExtractor` |
| `incremental_knowledge_graph_builder.py` | Builds KG incrementally with checkpoint support | `IncrementalKGBuilder` |

**Hierarchy**: Document → Chapter (Chương) → Section (Mục) → Article (Điều) → Clause (Khoản) → Point (Điểm)

**ID Format Example**: `59-2020-QH14:d5:k1:a` (document:article:clause:point)

### 2. Online Module (`vn_legal_rag/online/`)

3-tier retrieval engine for query processing and article ranking.

| Tier | File | Purpose | Key Classes |
|------|------|---------|-------------|
| **0** | `vietnamese-legal-query-analyzer.py` | Query intent detection & analysis | `VietnameseLegalQueryAnalyzer`, `QueryIntent`, `LegalQueryType` |
| **1** | `tree-traversal-retriever.py` | LLM-guided document tree navigation | `TreeTraversalRetriever`, `TreeSearchResult` |
| **2** | `dual-level-retriever.py` | 6-component semantic scoring | `DualLevelRetriever`, `DualLevelConfig` |
| **3** | `semantic-bridge-rrf-merger.py` | RRF fusion & KG expansion | `SemanticBridge` |
| Main | `legal-graphrag-3tier-query-engine.py` | Query orchestration & response generation | `LegalGraphRAG`, `GraphRAGResponse` |
| Support | `personalized-page-rank-for-kg.py` | PPR scoring on knowledge graph | `PersonalizedPageRank` |
| Support | `cross-encoder-reranker-for-legal-documents.py` | Cross-encoder reranking | `CrossEncoderReranker` |
| Support | `document-aware-result-filter.py` | Post-retrieval filtering | `DocumentAwareResultFilter` |
| Support | `ontology-based-query-expander.py` | Query expansion via ontology | `OntologyBasedQueryExpander` |

### 3. Types Module (`vn_legal_rag/types/`)

Data models and enumerations.

| File | Purpose | Key Classes |
|------|---------|-------------|
| `entity_types.py` | 11 legal entity types | `LegalEntityType` (ORGANIZATION, PERSON_ROLE, LEGAL_TERM, LEGAL_REFERENCE, MONETARY, PERCENTAGE, DURATION, LOCATION, CONDITION, ACTION, PENALTY) |
| `relation_types.py` | 28+ legal relation types | `LegalRelationType` (REQUIRES, HAS_PENALTY, APPLIES_TO, REFERENCES, AMENDS, etc.) |
| `tree_models.py` | Document tree structures | `TreeNode`, `CrossRefEdge`, `UnifiedForest` |
| `ablation-config-for-rag-component-testing.py` | Configuration for ablation studies | `AblationConfig` |

### 4. Utils Module (`vn_legal_rag/utils/`)

Utility functions and providers.

| File | Purpose | Key Classes |
|------|---------|-------------|
| `basic_llm_provider.py` | Base LLM client interface | `BaseLLMProvider` |
| `llm_provider_with_caching.py` | LLM response caching | `CachedLLMProvider` |
| `text_embeddings_provider.py` | Text embedding generation | `TextEmbeddingsProvider` |
| `vietnamese_abbreviation_expander.py` | Domain-specific abbreviation expansion | `VietnameseAbbreviationExpander` |
| `legal_citation_formatter.py` | Vietnamese legal citation formatting | `LegalCitationFormatter` |
| `simple_logger.py` | Structured logging | `SimpleLogger` |
| `config-loader-with-yaml-support.py` | YAML configuration loading | `ConfigLoader` |
| `data-loaders-for-kg-and-summaries.py` | Data loading utilities | Various loaders for KG, summaries |
| `simple-progress-tracker-for-testing.py` | Progress tracking for evaluation | `ProgressTracker` |

### 5. Configuration Module (`config/`)

Domain-specific YAML configurations for each Vietnamese legal decree.

```
config/domains/
├── 59-2020-QH14.yaml          # Luật Doanh nghiệp 2020 (main legal code)
├── 01-2021-ND.yaml            # Nghị định 01/2021/NĐ-CP (enterprise registration)
├── 16-2023-ND.yaml            # Nghị định 16/2023/NĐ-CP (enterprise management)
├── 23-2022-ND.yaml, 44-2025-ND.yaml, etc.
└── default.yaml               # Default configuration template
```

**Configuration Contents**:
- `abbreviations`: Legal domain abbreviations (TGĐ=Tổng Giám đốc, TNHH=Trách nhiệm hữu hạn)
- `synonyms`: Term synonyms for query matching
- `topic_hints`: Domain-specific topic keywords
- `article_selection_examples`: Few-shot examples for LLM guidance
- `language`: Language code (vi)

### 6. Scripts (`scripts/`)

Evaluation and interactive utilities.

| File | Purpose |
|------|---------|
| `evaluate-retrieval-performance-on-test-set.py` | Evaluate 3-tier system on test dataset (379 Q&A pairs) |
| `offline-kg-extraction-with-checkpoint-resume.py` | KG extraction pipeline with checkpoint resumption |
| `online-interactive-legal-qa-system.py` | Interactive CLI for legal Q&A |
| `run-full-training-test.py` | Full end-to-end training pipeline |

### 7. Data Directory (`data/`)

Runtime data (not in git, downloaded separately).

```
data/
├── legal_docs.db                    # SQLite: Article text (~10MB)
├── llm_cache.db                     # SQLite: LLM response cache (~1.1GB)
├── article_to_document_mapping.json # Article ID → document mapping
├── kg_enhanced/
│   ├── legal_kg.json                # Knowledge graph (1299 entities, 2577 relations)
│   ├── chapter_summaries.json       # Loop 1 summaries for tree traversal
│   ├── article_summaries.json       # Loop 2 summaries for tree traversal
│   ├── checkpoint.json              # Resume checkpoint for extraction
│   └── ontology.ttl                 # RDF ontology (OWL)
└── training/
    └── training_with_ids.csv        # Evaluation dataset (379 Q&A pairs)
```

## Data Structures

### Document Tree Hierarchy

```python
TreeNode (6 types: document, chapter, section, article, clause, point)
├── node_id: str              # Unique identifier
├── node_type: NodeType       # Enum: document | chapter | section | article | clause | point
├── name: str                 # Display name
├── description: str          # Summary/summary
├── content: str              # Full text content
├── metadata: Dict[str, Any]  # Custom metadata (page, section_order, etc.)
└── sub_nodes: List[TreeNode] # Children in hierarchy
```

**ID Format**: `{document}:{type}{number}` (colon-separated)
- Document: `59-2020-QH14`
- Chapter: `59-2020-QH14:c1` (chapter 1)
- Article: `59-2020-QH14:d5` (article 5)
- Clause: `59-2020-QH14:d5:k1` (article 5, clause 1)
- Point: `59-2020-QH14:d5:k1:a` (article 5, clause 1, point a)

### Knowledge Graph

```python
kg = {
    "nodes": [
        {"id": str, "type": LegalEntityType, "content": str, ...},
        ...  # 1299 entities total
    ],
    "edges": [
        {"source_id": str, "target_id": str, "relation_type": LegalRelationType, ...},
        ...  # 2577 relations total
    ]
}
```

### Query Result

```python
GraphRAGResponse(
    response: str,                      # LLM-generated answer
    citations: List[Dict],              # [{"citation_string": "Điều 5...", "source": "..."}]
    reasoning_path: List[str],          # ["tier1", "tier2", "tier3", ...]
    confidence: float,                  # 0.0 - 1.0
    query_type: LegalQueryType,         # article_lookup | guidance | situation_analysis | compare | case_law
    metadata: Dict[str, Any],           # Additional metadata
    tree_search_result: Optional,       # TreeSearchResult from tier 1
    intent: Optional[QueryIntent]       # Analyzed intent from query analyzer
)
```

## Retrieval Pipeline

### Tier 0: Query Analysis

**Module**: `vietnamese-legal-query-analyzer.py`

1. **Query Intent Detection**: Classifies query into one of 5 types:
   - `article_lookup`: Direct article reference (e.g., "Điều 5 quy định gì?")
   - `guidance_document`: "How to" questions
   - `situation_analysis`: "What happens if..." scenarios
   - `compare_regulations`: Comparison between articles
   - `case_law_lookup`: Specific penalties and consequences

2. **Keyword Extraction**: Extracts legal keywords, entities, concepts
3. **Query Expansion**: Expands abbreviations, adds synonyms via Vietnamese abbreviation expander
4. **Domain Detection**: Identifies relevant legal domains

### Tier 1: Tree Traversal

**Module**: `tree-traversal-retriever.py`

**Strategy**: LLM-guided multi-path navigation through document hierarchy

**Implementation**:
- **Loop 0** (Optional): Select documents using document summaries
- **Loop 1**: LLM selects chapters based on chapter_summaries + query
  - Multi-path: Consider top-k candidate chapters
  - Summary matching: Uses dense embeddings or semantic similarity
  - Domain expansion: Applies Vietnamese abbreviation expansion

- **Loop 2**: LLM selects articles within chapters using article_summaries
  - Semantic scoring: Article summary → embedding → cosine similarity
  - Dual scoring: Keyword TF-IDF + semantic embeddings
  - Hierarchy bonus: Same chapter articles ranked higher

**Output**: Sorted list of candidate articles with scores

**Strengths**: High precision for direct article lookups (82.05% Hit@10 on article_lookup queries)

### Tier 2: DualLevel Retrieval

**Module**: `dual-level-retriever.py`

**Strategy**: Global semantic search with 6-component scoring

**6 Components** (unified scoring):
1. **Keyphrase**: TF-IDF matching on legal keyphrases extracted from query
2. **Semantic**: Dense embedding similarity (text_embeddings_provider)
3. **PPR** (Personalized PageRank): Knowledge graph centrality (personalized-page-rank-for-kg.py)
4. **Concept**: Legal concept overlap (entity and relation matching)
5. **Theme**: Topic-level matching (document sections, themes)
6. **Hierarchy**: Structural proximity (same chapter bonus, parent-child bonus)

**Scoring Formula**: Weighted sum of 6 components with tunable weights

**Configuration** (`DualLevelConfig`):
```python
DualLevelConfig(
    keyphrase_weight=0.15,
    semantic_weight=0.30,
    ppr_weight=0.15,
    concept_weight=0.15,
    theme_weight=0.10,
    hierarchy_weight=0.15,
    top_k=50  # Return top 50 candidates
)
```

**Output**: Ranked list of articles with composite scores

**Strengths**: Comprehensive coverage, captures complex semantic relationships

### Tier 3: Semantic Bridge (RRF Fusion)

**Module**: `semantic-bridge-rrf-merger.py`

**Strategy**: Merge and enhance Tier 1 + Tier 2 results via RRF and KG expansion

**Process**:
1. **Reciprocal Rank Fusion (RRF)**: Combines rankings from Tier 1 and Tier 2
   ```
   RRF_score = 1/(k + rank_tier1) + 1/(k + rank_tier2)
   ```

2. **Agreement Scoring**: Articles appearing in both tiers receive ranking boost

3. **KG Expansion**: For top-k articles, extract related entities/articles via:
   - Cross-references (REFERENCES, AMENDS, IMPLEMENTS relations)
   - Structural relations (CONTAINS, PART_OF)
   - Authority relations (AUTHORIZED_BY, RESPONSIBLE_FOR)

4. **Diversity**: Ensures coverage across document sections

**Configuration**:
```python
SemanticBridge(
    rrf_k=60,              # RRF fusion parameter
    kg_expansion_hops=2,   # Traverse 2 hops in KG
    agreement_boost=0.3,   # Boost for tier agreement
    max_results=10         # Final return top-10
)
```

**Output**: Final ranked list of top-10 articles with citations

**Strengths**: Robust fusion, captures cross-chapter dependencies

## Key Design Patterns

### 1. Kebab-Case Module Names

Filenames use kebab-case for clarity:
```python
# Import pattern for kebab-case modules
from importlib import import_module
_module = import_module(".kebab-case-filename", "vn_legal_rag.online")
ClassName = _module.ClassName
```

Benefit: Self-documenting file names, clear module purpose at a glance.

### 2. Type-Safe Data Models

Extensive use of dataclasses and Enums:
- `LegalEntityType`, `LegalRelationType`: Enumerated types
- `TreeNode`, `CrossRefEdge`: Structured data models
- `GraphRAGResponse`: Standardized response format
- `DualLevelConfig`, `AblationConfig`: Configuration objects

### 3. Plug-in LLM Providers

Abstract base class `BaseLLMProvider` with implementations:
- OpenAI provider
- Anthropic provider
- Generic HTTP provider (for local models like Ollama)

**Factory Pattern**:
```python
from vn_legal_rag.utils import create_llm_provider
llm = create_llm_provider(provider="openai", model="gpt-4o-mini")
```

### 4. Domain Configuration

Each legal decree has a YAML config for abbreviations, synonyms, and few-shot examples. Enables per-domain customization without code changes.

### 5. Ablation Configuration

`AblationConfig` class allows disabling specific tiers for evaluation:
```python
ablation = AblationConfig(
    disable_tier1=False,
    disable_tier2=False,
    disable_tier3=False
)
```

### 6. Checkpoint-Based KG Extraction

Incremental KG builder supports checkpointing to resume long-running extraction:
```python
builder = IncrementalKGBuilder(checkpoint_path="data/kg_enhanced/checkpoint.json")
kg, checkpoint = builder.build(documents, resume=True)
```

## File Naming Conventions

**Pattern**: Kebab-case for multi-word module names (not underscore)

| Pattern | Example | Reason |
|---------|---------|--------|
| **Kebab-case** | `tree-traversal-retriever.py` | Self-documenting, easy to identify in glob/grep |
| **Underscore** | `text_embeddings_provider.py` | Used for traditional single-concept utils |

Recommendation: Use kebab-case for tier-specific modules, underscore for small utilities.

## Integration Points

### Public API

Main entry point: `LegalGraphRAG` class

```python
from vn_legal_rag import LegalGraphRAG, UnifiedForest
import json

# Load data
with open("data/kg_enhanced/legal_kg.json") as f:
    kg = json.load(f)
with open("data/document_forest.json") as f:
    forest = UnifiedForest.from_json(f.read())

# Initialize
graphrag = LegalGraphRAG(
    kg=kg,
    forest=forest,
    db_path="data/legal_docs.db",
    llm_provider="openai",
    llm_model="gpt-4o-mini"
)

# Query
result = graphrag.query("Phạt bao nhiêu nếu vi phạm Điều 5?")
print(result.response)
```

### Database Interface

```python
from vn_legal_rag.offline import LegalDocumentDB

db = LegalDocumentDB("data/legal_docs.db")
article = db.get_article("59-2020-QH14:d5")  # Fetch article text
```

### Knowledge Graph

```python
import json
with open("data/kg_enhanced/legal_kg.json") as f:
    kg = json.load(f)
    # kg["nodes"]: List of entities
    # kg["edges"]: List of relations
```

## Performance Characteristics

### Hit Rate by Query Type

| Query Type | Sample Count | Hit@5 | Hit@10 | Performance Notes |
|------------|--------------|-------|--------|------------------|
| article_lookup | 156 | 74.36% | 82.05% | Direct references, tree traversal excels |
| guidance_document | 98 | 64.29% | 73.47% | "How to" procedural queries |
| situation_analysis | 75 | 62.67% | 71.33% | Conditional scenarios, tier 2 + tier 3 important |
| compare_regulations | 32 | 62.50% | 68.75% | Cross-article comparisons |
| case_law_lookup | 18 | 50.00% | 61.11% | Penalty lookups, weaker tier 1 applicability |

### Component Contribution (Ablation Results)

| Configuration | Hit@5 | Hit@10 | Difference |
|---------------|-------|--------|-----------|
| Tier 1 only | 44.33% | 56.20% | - |
| Tier 2 only | 48.55% | 61.74% | - |
| Tier 1 + Tier 2 | 62.53% | 71.24% | +14.5% vs best single |
| **Full 3-Tier** | **67.28%** | **76.53%** | +5.3% with tier 3 fusion |

Insight: Tier 3 fusion provides incremental improvement; tier 1 + tier 2 already very strong.

## Configuration & Environment

### LLM Providers Supported

- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Haiku, Claude 3.5 Sonnet
- **Local Models**: Via HTTP provider (Ollama, vLLM)

### Embedding Models

Default: Dense embeddings via OpenAI API or BAAI/bge models (huggingface)

### Database

SQLite for document storage (legal_docs.db)

### Cache Layer

LLM response caching via SQLite (llm_cache.db) for faster repeat queries

## Testing & Evaluation

### Unit Tests

Located in `tests/` directory:
- `test-online-module-imports.py`: Verify module loading

### Integration Tests

```bash
python scripts/evaluate-retrieval-performance-on-test-set.py \
    --test-file data/training/training_with_ids.csv \
    --disable-tier1  # Optional ablation
```

### Evaluation Metrics

- **Hit@K**: % of queries where ground-truth article in top-K results
- **Recall@K**: Average recall across top-K
- **MRR** (Mean Reciprocal Rank): Average rank of first correct answer

### Code Quality Tools

```bash
black vn_legal_rag/ tests/     # Format
ruff check vn_legal_rag/       # Lint
mypy vn_legal_rag/             # Type check
pytest tests/                  # Unit tests
```

## Dependencies

### Core
- `sqlalchemy>=2.0`: ORM for document database
- `numpy`, `scipy`: Numerical computations (embeddings, PPR)
- `networkx`: Knowledge graph traversal

### LLM Providers
- `openai>=1.0`: OpenAI API client
- `anthropic>=0.7`: Anthropic Claude API
- `requests`: HTTP client for local models

### Utilities
- `pyyaml`: Configuration file parsing
- `python-dotenv`: Environment variable loading
- `tqdm`: Progress bars

### Development
- `pytest`: Testing framework
- `black`: Code formatting
- `ruff`: Linting
- `mypy`: Static type checking

---

**Last Updated**: 2026-02-14
**Codebase Version**: v1.0.0 (Research Prototype)
