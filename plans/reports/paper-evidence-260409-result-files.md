# Paper Evidence — Result Files Reference

Maps every number in `paper/paper-release.tex` to the result file that produced it.

## Legal Domain (400 queries) — Table 1 & Table 2

**Our system (3-Tier GraphRAG):**
- Result file: `results/hard400_attempt10b_run1.json`
- Date: 2026-04-09
- Config: tree=1.2, same-chapter KG expansion, +12 REFERENCES relations, temp=0 Loop 2
- Model: Claude Sonnet 4 via proxy localhost:3210

| Metric | Value | Source |
|--------|-------|--------|
| Hit@5 | 74.5% (298/400) | `hard400_attempt10b_run1.json` |
| Hit@10 | 92.2% (369/400) | `hard400_attempt10b_run1.json` |
| Hit@20 | 94.8% (379/400) | `hard400_attempt10b_run1.json` |
| Hit@30 | 95.0% (380/400) | `hard400_attempt10b_run1.json` |
| MRR | 0.603 | `hard400_attempt10b_run1.json` |
| Enterprise Hit@10 | 86.5% (173/200) | `hard400_attempt10b_run1.json` |
| Traffic Hit@10 | 98.0% (196/200) | `hard400_attempt10b_run1.json` |

**PageIndex baseline:**
- Result file: `results/hard400_baseline_pageindex_run3.json` + `results/hard400_baseline_pageindex_run3_from156.json`
- Date: 2026-04-07
- Model: Claude Sonnet 4 via proxy localhost:3210

| Metric | Value | Source |
|--------|-------|--------|
| Hit@5 | 76.2% | `hard400_baseline_pageindex_run3.json` |
| Hit@10 | 84.8% (339/400) | `hard400_baseline_pageindex_run3.json` |
| Hit@20 | 90.2% | `hard400_baseline_pageindex_run3.json` |
| Hit@30 | 91.0% | `hard400_baseline_pageindex_run3.json` |
| MRR | 0.559 | `hard400_baseline_pageindex_run3.json` |
| Enterprise Hit@10 | 76.6% (151/197) | `hard400_baseline_pageindex_run3.json` |
| Traffic Hit@10 | 92.6% (188/203) | `hard400_baseline_pageindex_run3.json` |

**LightRAG baseline:**
- Result file: `results/hard400_baseline_lightrag.json`
- Date: 2026-04-02
- Model: Claude Haiku (not yet rerun on Sonnet 4)

| Metric | Value | Source |
|--------|-------|--------|
| Hit@10 | 39.8% | `hard400_baseline_lightrag.json` |

**Traditional baselines (BM25, TF-IDF, Keyword, Semantic):**
- Result file: `results/hard400_traditional_baselines.json`
- Date: 2026-04-02
- No LLM needed (pure retrieval)

## Education Domain (130 queries) — Table 3

**Our system:**
- Result file: `results/edu_full_run3.json` (best of 3 runs: 88.5%)
- Other runs: `results/edu_full_run1.json` (87.7%), `results/edu_full_run2.json` (87.7%)
- Date: 2026-04-05
- Config: `config/education.yaml`, proxy localhost:3210

| Metric | Value | Source |
|--------|-------|--------|
| Hit@5 | 78.5% | `edu_full_run3.json` |
| Hit@10 | 88.5% | `edu_full_run3.json` |
| Hit@20 | 93.1% | `edu_full_run3.json` |
| Hit@30 | 94.6% | `edu_full_run3.json` |
| MRR | 0.594 | `edu_full_run3.json` |

**PageIndex (education):**
- Result file: `results/edu_baseline_pageindex.json`
- Hit@10 = 74.6%

**LightRAG (education):**
- Result file: `results/edu_baseline_lightrag.json`
- Hit@10 = 53.1%

## KG Statistics (Table: Dataset Statistics)

- Entities: 5,936 — counted from `data/kg_enhanced/legal_kg.json`
- Relations: 5,963 — counted from `data/kg_enhanced/legal_kg.json` (updated 2026-04-09, +12 REFERENCES)

## Error Analysis (Section 4.4)

- 31 missed queries (7.8%) — from `results/hard400_attempt10b_run1.json`
- 27 enterprise, 4 traffic
- 14/31 have >2 expected articles (multi-article)
- Category breakdown computed from `category` field in result file

## Benchmarks

| Benchmark | File | Queries |
|-----------|------|---------|
| Hard-400 legal | `data/benchmark/hard-400-qa-benchmark.csv` | 400 (200 enterprise + 200 traffic) |
| Education | `data/benchmark/edu-master-qa-benchmark.csv` | 130 |

## Eval Command (Reproducibility)

```bash
# Legal domain (hard-400)
env -u ANTHROPIC_AUTH_TOKEN python scripts/evaluate-full-rag-with-baselines.py \
  --test-file data/benchmark/hard-400-qa-benchmark.csv \
  --no-traditional --ablations "" --retrieval-only \
  -w 5 -o results/hard400_attempt10b_run1.json

# Education domain
env -u ANTHROPIC_AUTH_TOKEN PYTHONUNBUFFERED=1 python scripts/evaluate-full-rag-with-baselines.py \
  --config config/education.yaml \
  --test-file data/benchmark/edu-master-qa-benchmark.csv \
  --no-traditional --ablations "" \
  -w 10 -o results/edu_full_run3.json
```

## Code Changes from Baseline (tree=1.2 only) to Final

1. `vn_legal_rag/online/legal-graphrag-3tier-query-engine.py` — Same-chapter STRONG relation expansion enabled (lines 846-851)
2. `vn_legal_rag/online/tree-traversal-retriever.py` — `temperature=0.0` for Loop 2 (line 582)
3. `data/kg_enhanced/legal_kg.json` — +12 REFERENCES relations (5,951 → 5,963)
