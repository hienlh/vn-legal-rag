# Data Directory

This directory contains all data files for the Vietnamese Legal RAG system.

## Required Files

### Database Files

| File | Size | Description | Source |
|------|------|-------------|--------|
| `legal_docs.db` | ~10MB | SQLite database with parsed legal documents | Copy from semantica project |
| `llm_cache.db` | ~1.1GB | LLM response cache for faster re-runs | Copy from semantica project |

### Knowledge Graph Files (`kg_enhanced/`)

| File | Size | Description |
|------|------|-------------|
| `legal_kg.json` | ~2.1MB | Knowledge graph (1299 entities, 2577 relations) |
| `chapter_summaries.json` | ~134KB | Chapter keywords for tree navigation (Loop 1) |
| `article_summaries.json` | ~636KB | Article keywords for retrieval (Loop 2) |
| `checkpoint.json` | ~2.9MB | Resume checkpoint for extraction pipeline |
| `ontology.ttl` | ~4.7KB | Formal ontology (Turtle format) |

### Benchmark (`benchmark/`)

| File | Size | Description |
|------|------|-------------|
| `legal-qa-benchmark.csv` | ~564KB | 379 Q&A pairs with article references |

### Baseline Results (`baseline-results/`)

| File | Size | Description |
|------|------|-------------|
| `02-baseline-384q-full-results.json` | ~1.5MB | TF-IDF/BM25 baseline (41.41% Hit@10) |
| `03-lightrag-384q-full-results.json` | ~400KB | LightRAG results (77.3% Hit@10) |
| `10-pure-pageindex-rag-375q-results.json` | ~124KB | PageIndex baseline (35.2% Hit@10) |

### ID Mapping

| File | Size | Description |
|------|------|-------------|
| `article_to_document_mapping.json` | ~15KB | Article ID → Document ID lookup |

## Data Structure

```
data/
├── legal_docs.db                    # SQLite database
├── llm_cache.db                     # LLM cache
├── article_to_document_mapping.json # ID mapping
├── kg_enhanced/                     # Canonical KG directory
│   ├── legal_kg.json
│   ├── chapter_summaries.json
│   ├── article_summaries.json
│   ├── checkpoint.json
│   └── ontology.ttl
├── benchmark/                       # Evaluation benchmark
│   └── legal-qa-benchmark.csv
└── baseline-results/                # Baseline comparison results
    ├── 02-baseline-384q-full-results.json
    ├── 03-lightrag-384q-full-results.json
    └── 10-pure-pageindex-rag-375q-results.json
```

## Usage

### Loading Database

```python
from vn_legal_rag.database import LegalDocumentDB

db = LegalDocumentDB("data/legal_docs.db")
```

### Loading Knowledge Graph

```python
import json

with open("data/kg_enhanced/legal_kg.json") as f:
    kg = json.load(f)
```

### Loading Training Data

```python
import pandas as pd

df = pd.read_csv("data/benchmark/legal-qa-benchmark.csv")
```

## Git Ignore

These data files are NOT committed to git. Users must:
1. Copy from semantica project, OR
2. Generate using extraction pipeline

See `.gitignore` for excluded files.
