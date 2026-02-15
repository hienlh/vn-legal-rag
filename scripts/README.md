# vn_legal_rag Scripts

Entry point scripts for the Vietnamese Legal RAG system.

## Overview

| Script | Purpose | Mode |
|--------|---------|------|
| `offline-kg-extraction-with-checkpoint-resume.py` | Extract KG from legal documents | Offline |
| `online-interactive-legal-qa-system.py` | Interactive Q&A with 3-Tier GraphRAG | Online |
| `evaluate-retrieval-performance-on-test-set.py` | Evaluate retrieval metrics | Testing |

---

## Offline: KG Extraction

**Script:** `offline-kg-extraction-with-checkpoint-resume.py`

Extract knowledge graph from Vietnamese legal documents stored in SQLite database.

### Features

- Unified entity/relation extraction (LightRAG-style single LLM call)
- Incremental KG building with built-in merge
- Checkpoint/resume capability (fault-tolerant)
- Atomic checkpoint writes (crash-safe)
- Progress tracking with per-article stats

### Usage

```bash
# Basic run (auto-resume if checkpoint exists)
python scripts/offline-kg-extraction-with-checkpoint-resume.py

# Custom database and output
python scripts/offline-kg-extraction-with-checkpoint-resume.py \
    --db data/legal_docs.db \
    --output data/kg_enhanced

# Filter by document and limit articles
python scripts/offline-kg-extraction-with-checkpoint-resume.py \
    --document 59-2020-QH14 \
    --limit 50

# Force fresh start (ignore checkpoint)
python scripts/offline-kg-extraction-with-checkpoint-resume.py \
    --no-resume \
    --output data/kg_fresh

# Use different LLM
python scripts/offline-kg-extraction-with-checkpoint-resume.py \
    --provider openai \
    --model gpt-4o-mini

# Verbose logging
python scripts/offline-kg-extraction-with-checkpoint-resume.py --verbose
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | data/legal_docs.db | SQLite database path |
| `--output` | data/kg_enhanced | Output directory |
| `--config` | config/default.yaml | Config file |
| `--document` | None | Filter by document ID |
| `--limit` | None | Limit articles to process |
| `--provider` | From config | LLM provider (openai, anthropic, gemini) |
| `--model` | From config | LLM model name |
| `--no-resume` | False | Ignore checkpoint, start fresh |
| `--verbose` | False | Enable verbose logging |

### Output Files

```
data/kg_enhanced/
├── legal_kg.json          # Knowledge graph (entities + relations)
└── checkpoint.json        # Resume checkpoint (auto-deleted on success)
```

### Checkpoint Format

```json
{
  "processed_ids": ["59-2020-QH14:d1", "59-2020-QH14:d2"],
  "extraction_results": [...],
  "stats": {
    "successful": 10,
    "failed": 0,
    "total_entities": 45,
    "total_relations": 89
  },
  "last_saved": "2026-02-14T09:54:00"
}
```

---

## Online: Interactive Q&A

**Script:** `online-interactive-legal-qa-system.py`

Interactive legal question answering with 3-Tier GraphRAG architecture.

### Features

- 3-Tier retrieval (Tree + DualLevel + Semantic Bridge)
- Query expansion with keywords, concepts, themes
- Adaptive retrieval based on query type
- Citation tracking with evidence
- Interactive REPL mode

### Usage

```bash
# Single query
python scripts/online-interactive-legal-qa-system.py \
    --query "Điều kiện thành lập công ty cổ phần?"

# Interactive mode
python scripts/online-interactive-legal-qa-system.py --interactive

# Verbose output (show expanded query, retrieval details)
python scripts/online-interactive-legal-qa-system.py \
    --query "..." \
    --verbose

# Custom config
python scripts/online-interactive-legal-qa-system.py \
    --config config/custom.yaml \
    --interactive

# Override LLM provider
python scripts/online-interactive-legal-qa-system.py \
    --provider openai \
    --model gpt-4o \
    --interactive

# Debug mode
python scripts/online-interactive-legal-qa-system.py \
    --interactive \
    --debug
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--query` | None | Single query to process |
| `--interactive` | False | Run in interactive mode |
| `--config` | config/default.yaml | Config file |
| `--db` | From config | Database path override |
| `--provider` | From config | LLM provider override |
| `--model` | From config | LLM model override |
| `--verbose` | False | Show expanded query + retrieval details |
| `--debug` | False | Enable debug logging |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `exit`, `quit`, `q` | Exit session |
| `help`, `h` | Show help |
| `verbose` | Toggle verbose mode |

### Example Session

```
$ python scripts/online-interactive-legal-qa-system.py --interactive

================================================================================
Vietnamese Legal Q&A System - Interactive Mode
================================================================================

Commands:
  - Type your question in Vietnamese
  - Type 'exit' or 'quit' to end session
  - Type 'help' for command list
================================================================================

Question: Điều kiện thành lập CTCP?

================================================================================
Query: Điều kiện thành lập CTCP?
================================================================================

Query Type: situation_analysis
Retrieval Method: 3-Tier (Tree + DualLevel + Semantic Bridge)

Retrieved Articles (5):
  1. 59-2020-QH14:d11
  2. 59-2020-QH14:d12
  3. 59-2020-QH14:d13
  4. 59-2020-QH14:d4
  5. 59-2020-QH14:d5

================================================================================
ANSWER:
================================================================================
[LLM-generated answer with citations]
...
```

---

## Testing: Evaluation

**Script:** `evaluate-retrieval-performance-on-test-set.py`

Evaluate retrieval performance on test dataset with standard metrics.

### Features

- Hit@K, Recall@K, MRR calculation
- Per-query and aggregate metrics
- Failed query analysis
- JSON export with metadata

### Usage

```bash
# Full evaluation (all test questions)
python scripts/evaluate-retrieval-performance-on-test-set.py

# Custom test file
python scripts/evaluate-retrieval-performance-on-test-set.py \
    --test-file data/benchmark/legal-qa-benchmark.csv

# Limit test questions
python scripts/evaluate-retrieval-performance-on-test-set.py --limit 50

# Verbose (per-query results)
python scripts/evaluate-retrieval-performance-on-test-set.py --verbose

# Export results
python scripts/evaluate-retrieval-performance-on-test-set.py \
    --output results/eval_2026-02-14.json

# Custom config
python scripts/evaluate-retrieval-performance-on-test-set.py \
    --config config/experimental.yaml \
    --output results/experimental_eval.json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test-file` | data/benchmark/legal-qa-benchmark.csv | Test CSV path |
| `--config` | config/default.yaml | Config file |
| `--limit` | None | Limit test questions |
| `--output` | None | Output JSON path |
| `--verbose` | False | Print per-query results |
| `--debug` | False | Enable debug logging |

### Test CSV Format

Expected columns:
- `question`: Vietnamese legal question
- `article_ids`: Comma or semicolon separated article IDs
- `document_id`: Document ID (optional)

Example:

```csv
question,article_ids,document_id
"Điều kiện thành lập CTCP?","59-2020-QH14:d11,59-2020-QH14:d12",59-2020-QH14
```

### Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Hit@K** | % queries with ≥1 correct answer in top-K | Binary hit/miss |
| **Recall@K** | Avg % of expected articles found in top-K | \|retrieved ∩ expected\| / \|expected\| |
| **MRR** | Mean reciprocal rank of first correct answer | Avg(1/rank) |

### Output Format

```json
{
  "metadata": {
    "timestamp": "2026-02-14T10:30:00",
    "test_file": "data/benchmark/legal-qa-benchmark.csv",
    "config": "config/default.yaml",
    "total_queries": 379,
    "limit": null
  },
  "metrics": {
    "total_queries": 379,
    "successful_queries": 299,
    "failed_queries": 80,
    "hit_rate@5": 0.7893,
    "hit_rate@10": 0.8234,
    "recall@5": 0.5812,
    "recall@10": 0.6534,
    "mrr": 0.5422
  },
  "results": [
    {
      "index": 0,
      "question": "...",
      "expected_ids": ["59-2020-QH14:d11"],
      "retrieved_ids": ["59-2020-QH14:d11", "..."],
      "query_type": "article_lookup",
      "hit@5": true,
      "recall@5": 1.0,
      "mrr": 1.0,
      "success": true
    },
    ...
  ]
}
```

---

## Configuration

All scripts use `config/default.yaml` by default. Override with `--config` or individual CLI args.

### Config Structure

```yaml
llm:
  provider: "anthropic"
  model: "claude-3-5-haiku-20241022"
  use_cache: true
  cache_db: "data/llm_cache.db"

database:
  path: "data/legal_docs.db"

kg:
  path: "data/kg_enhanced/legal_kg.json"
  chapter_summaries: "data/kg_enhanced/chapter_summaries.json"
  article_summaries: "data/kg_enhanced/article_summaries.json"
  forest: "data/document_forest.json"

retrieval:
  max_documents: 3
  max_chapters: 6
  max_articles: 7
  confidence_threshold: 0.7

dual_level:
  keyphrase_weight: 0.05
  semantic_weight: 0.20
  ppr_weight: 0.25
  concept_weight: 0.20
  theme_weight: 0.15
  hierarchy_weight: 0.15
```

---

## Common Workflows

### 1. Extract KG from Documents

```bash
# Extract KG from all documents
python scripts/offline-kg-extraction-with-checkpoint-resume.py

# If interrupted, resume from checkpoint
python scripts/offline-kg-extraction-with-checkpoint-resume.py
```

### 2. Test Q&A System

```bash
# Interactive testing
python scripts/online-interactive-legal-qa-system.py --interactive

# Single query test
python scripts/online-interactive-legal-qa-system.py \
    --query "Điều kiện thành lập CTCP?" \
    --verbose
```

### 3. Evaluate Performance

```bash
# Run full evaluation
python scripts/evaluate-retrieval-performance-on-test-set.py \
    --output results/eval_$(date +%Y%m%d).json

# Quick evaluation (50 questions)
python scripts/evaluate-retrieval-performance-on-test-set.py \
    --limit 50 \
    --verbose
```

### 4. Experiment with LLM Providers

```bash
# Compare OpenAI vs Anthropic
python scripts/evaluate-retrieval-performance-on-test-set.py \
    --provider openai \
    --model gpt-4o-mini \
    --limit 100 \
    --output results/openai_eval.json

python scripts/evaluate-retrieval-performance-on-test-set.py \
    --provider anthropic \
    --model claude-3-5-haiku-20241022 \
    --limit 100 \
    --output results/anthropic_eval.json
```

---

## Dependencies

All scripts require:
- Python 3.8+
- vn_legal_rag package installed
- Config file at `config/default.yaml` (or custom path)
- Data files (database, KG, summaries) as specified in config

For offline extraction:
- Legal documents database (`data/legal_docs.db`)

For online Q&A and evaluation:
- Knowledge graph (`data/kg_enhanced/legal_kg.json`)
- Document forest (`data/document_forest.json`)
- Chapter/article summaries (optional but recommended)

---

## Troubleshooting

### Checkpoint Recovery

If offline extraction crashes:

```bash
# Resume from last checkpoint (automatic)
python scripts/offline-kg-extraction-with-checkpoint-resume.py

# If checkpoint is corrupted, start fresh
python scripts/offline-kg-extraction-with-checkpoint-resume.py --no-resume
```

### Missing Data Files

If online scripts fail with `FileNotFoundError`:

1. Check config file paths match actual file locations
2. Run offline extraction first to generate KG
3. Generate chapter/article summaries if missing (see main README)

### LLM API Errors

If API calls fail:
- Check API keys in environment variables
- Verify `llm.provider` and `llm.model` in config
- Use `--debug` flag to see detailed error messages
- Check LLM cache database is writable (`data/llm_cache.db`)

### Low Retrieval Performance

If evaluation shows low metrics:
- Adjust `dual_level` weights in config
- Tune `retrieval.confidence_threshold`
- Generate/update chapter and article summaries
- Check training data format matches expected CSV structure

---

## Development

### Adding New Scripts

1. Use kebab-case naming: `new-feature-script-name.py`
2. Add shebang: `#!/usr/bin/env python3`
3. Include docstring with usage examples
4. Use argparse for CLI args
5. Load config from `config/default.yaml`
6. Add to this README with documentation

### Script Template

```python
#!/usr/bin/env python3
"""
Brief description.

Usage:
    python scripts/new-feature-script-name.py --option value
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.utils import load_config, setup_logging

def main():
    parser = argparse.ArgumentParser(description="...")
    # Add arguments
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    # Implementation

    return 0

if __name__ == "__main__":
    sys.exit(main())
```
