# Quick Reference Index: Semantica RAG Benchmarks

## Critical Benchmark Files

### 1. Primary Results (JSON)
```
/home/hienlh/Projects/semantica/data/consolidated-results/

02-baseline-384q-full-results.json       1.5 MB  TF-IDF: 41.41% Hit@10
03-lightrag-384q-full-results.json       400 KB  LightRAG: 77.3% Hit rate
05-current-system-375q-full-results.json 1.9 MB  3-Tier: 76.53% Hit@10, 92.53% Hit@all
10-pure-pageindex-rag-375q-results.json  124 KB  PageIndex: 35.20% Hit rate
```

### 2. Baseline Scripts
```
/home/hienlh/Projects/semantica/scripts/

test_baseline.py (950 lines)                5 baselines: BM25, TF-IDF, Semantic, Keyword, Semantica
test-lightrag-vs-baseline-comparison.py     LightRAG + baseline comparison
run-ablation-study-for-rag-components.py    Tree/Dual/Full ablation study
```

### 3. Key Reports
```
/home/hienlh/Projects/semantica/plans/reports/

scout-260203-2316-baseline-test-results.md                   Comprehensive inventory
comparison-260128-1734-lightrag-vs-baseline-retrieval.md     20-question comparison
comparison-260128-2100-lightrag-full-384-evaluation.md       Full 384-question eval
docs-manager-260213-1612-pure-pageindex-baseline-integration.md  PageIndex standalone results
```

---

## Performance Comparison Table

| Method | Hit@10 | Hit@all | Precision | Recall | F1 | Notes |
|--------|--------|---------|-----------|--------|-----|-------|
| **TF-IDF** | **41.41%** | 42.45% | 6.7% | 26.8% | 9.9% | BEST baseline |
| BM25 | 35.42% | 36.46% | 5.9% | 22.5% | 8.5% | Classic IR |
| Keyword | 33.85% | 34.90% | 5.2% | 21.0% | 7.7% | Simple match |
| Semantic | 13.02% | 14.06% | 2.5% | 7.3% | 3.3% | Vector only |
| Semantica | 18.49% | 19.53% | 3.1% | 10.6% | 4.3% | 4-step hybrid |
| **LightRAG** | - | **71.35%** | 4.6% | 56.7% | 7.6% | Graph-based |
| **PageIndex** | 35.20% | 35.20% | - | - | - | Tree traversal |
| **3-Tier** | **76.53%** | **92.53%** | - | - | - | CURRENT BEST |

---

## Architecture Comparison

### 1. Baseline Methods (test_baseline.py)
- **BM25:** IDF + term frequency, k1=1.5, b=0.75
- **TF-IDF:** Log-normalized cosine similarity
- **Semantic:** sentence-transformers (384-dim), cosine similarity
- **Keyword:** Substring matching with weighting
- **Semantica:** 4-step: Vector → Graph → Memory → Rank&Merge

### 2. Advanced Methods
- **LightRAG:** KG-based with local/global retrieval modes
- **PageIndex:** Tree navigation with 2-loop hierarchy
- **3-Tier:** Tree (Tier 1) + DualLevel (Tier 2) + Semantic Bridge (Tier 3)

---

## Test Sets

| Dataset | Count | Path | Format | Notes |
|---------|-------|------|--------|-------|
| Training Questions | 384 | data/training/training_with_ids.csv | CSV | Ground truth with article refs |
| Test Results (100) | 100 | data/training/test_results.csv | CSV | 3-tier metrics |
| Ablation (50) | 50 | data/training/test_ablation.csv | CSV | Component breakdown |

---

## How to Reproduce Benchmarks

### Step 1: Run Baseline Tests
```bash
cd /home/hienlh/Projects/semantica
python scripts/test_baseline.py --all --method all \
  --export data/baseline_results.json
```

### Step 2: Run LightRAG Tests
```bash
python scripts/test-lightrag-vs-baseline-comparison.py \
  --index --limit 610
python scripts/test-lightrag-vs-baseline-comparison.py \
  --all --mode hybrid --export data/lightrag_results.json
```

### Step 3: Run Ablation Study
```bash
python scripts/run-ablation-study-for-rag-components.py \
  --all --export data/ablation_results.json
```

---

## Key Insights

| Finding | Evidence | Implication |
|---------|----------|------------|
| TF-IDF ceiling at 41.41% | Baseline results | Simple IR methods insufficient |
| LightRAG 1.87x improvement | 77.3% vs 42.4% | Graph structure critical |
| PageIndex standalone weak (35.2%) | Pure vs 3-tier | Tree alone needs context |
| Tree in 3-tier reaches 68.8% | Ablation study | KG boosts tree retrieval |
| 3-Tier at 92.53% Hit@all | Current system | Ensemble approach wins |

---

## Metrics Explained

```
Hit Rate        = % questions with ≥1 correct article
Hit@K           = % questions with answer in top K
Precision       = correct / retrieved
Recall          = correct / expected
F1              = 2 * (P*R)/(P+R)
```

---

## Data Format

### JSON Results
```json
{
  "summary": {
    "hit_rate": 0.4141,
    "avg_precision": 0.067,
    "ir_metrics_at_k": {"10": {"hit": 0.4141}}
  },
  "results": [{
    "stt": 1,
    "expected_articles": [206],
    "retrieved_articles": [22, 60, 206, ...],
    "hit": true
  }]
}
```

### CSV Format (test_results.csv)
```
STT,Category,Expected,Tree_Articles,KG_Articles,Merged,
Tree_Hit,KG_Hit,Final_Hit,Tree_Conf,Hybrid_Alpha,...
1,Loại 6,[206],[206],[206],[206],1,1,1,0.85,0.7,...
```

---

## Important Notes

1. **Question Counts Vary:**
   - Baseline: 384 questions
   - 3-Tier: 375 questions
   - Ablation: 379 questions

2. **Hit@10 vs Hit@all:**
   - Hit@10: How many found in top 10
   - Hit@all: How many found anywhere in results
   - LightRAG high Hit@all (71.35%) but lower Hit@10 (33.85%)

3. **Trade-offs:**
   - Baselines: Fast, simple, interpretable
   - LightRAG: High recall, low precision
   - 3-Tier: Best overall, complex

4. **Implementation Status:**
   - All baseline methods: ✓ Implemented
   - LightRAG: ✓ Fully integrated
   - PageIndex: ✓ Component in 3-tier
   - 3-Tier: ✓ Current production

---

**For full details, see:**
- Main Report: `/home/hienlh/Projects/vn_legal_rag/plans/reports/scout-260215-1239-semantica-rag-benchmarks.md`
- Test Scripts: `/home/hienlh/Projects/semantica/scripts/`
- Results: `/home/hienlh/Projects/semantica/data/consolidated-results/`
