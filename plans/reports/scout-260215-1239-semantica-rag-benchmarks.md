# Scout Report: Semantica RAG Benchmark Results & Baseline Comparisons

**Date:** 2026-02-15  
**Scout:** AI Agent  
**Work Context:** /home/hienlh/Projects/vn_legal_rag  
**Source Project:** /home/hienlh/Projects/semantica  
**Status:** COMPLETE

---

## Executive Summary

Located comprehensive RAG benchmark results from semantica project spanning multiple baseline implementations and advanced retrieval methods. The codebase shows clear performance progression from simple keyword-based methods (30-42% hit rate) through LightRAG (77.3%) to a current 3-Tier system achieving 84-89% hit rates.

**Key Baseline Results:**
- **TF-IDF:** 41.41% Hit@10 on 384 questions
- **BM25:** 35.42% Hit@10
- **LightRAG:** 77.3% Hit rate (full dataset)
- **PageIndex (Pure):** 35.20% Hit rate
- **3-Tier System (Current):** 76.53% - 84% Hit rate

---

## 1. Consolidated Results Directory

**Path:** `/home/hienlh/Projects/semantica/data/consolidated-results/`

Most comprehensive benchmark data with unified format:

| File | Method | Q Count | Hit@10 | Hit@all | Size |
|------|--------|---------|--------|---------|------|
| 01-baseline-10q-results.json | Baselines | 10 | N/A | N/A | 35 KB |
| **02-baseline-384q-full-results.json** | TF-IDF, BM25, Semantic | 384 | **41.41%** | **42.45%** | 1.5 MB |
| **03-lightrag-384q-full-results.json** | LightRAG Hybrid | 384 | 33.85% | 71.35% | 400 KB |
| 04-lightrag-384q-reeval-results.json | LightRAG Re-eval | 384 | N/A | N/A | 1.6 MB |
| **05-current-system-375q-full-results.json** | 3-Tier System | 375 | **76.53%** | **92.53%** | 1.9 MB |
| 06-ablation-study-379q-results.json | Ablation (tree, dual, full) | 379 | N/A | N/A | 1.5 MB |
| 07-ablation-hit-at-k-50q-results.json | Hit@K metrics | 50 | N/A | N/A | 211 KB |
| 08-training-test-results.csv | Training format | N/A | N/A | N/A | 78 KB |
| **09-training-384q-with-ids.csv** | Ground truth | 384 | N/A | N/A | 564 KB |
| **10-pure-pageindex-rag-375q-results.json** | PageIndex standalone | 375 | 35.20% | 35.20% | 124 KB |
| README.md | Documentation | - | - | - | 4.2 KB |

### Key Metrics Summary

| Baseline Method | Hit@10 | Hit@all | Precision | Recall | F1 |
|-----------------|--------|---------|-----------|--------|-----|
| TF-IDF | **41.41%** | 42.45% | 6.7% | 26.8% | 9.9% |
| BM25 | 35.42% | 36.46% | 5.9% | 22.5% | 8.5% |
| Keyword Match | 33.85% | 34.90% | 5.2% | 21.0% | 7.7% |
| Semantic (vector only) | 13.02% | 14.06% | 2.5% | 7.3% | 3.3% |
| Semantica Original | 18.49% | 19.53% | 3.1% | 10.6% | 4.3% |
| **LightRAG** | 33.85% | **71.35%** | 4.6% | 56.7% | 7.6% |
| **PageIndex (Pure)** | 35.20% | 35.20% | - | - | - |
| **3-Tier System** | **76.53%** | **92.53%** | - | - | - |

---

## 2. Test Results Directory

**Path:** `/home/hienlh/Projects/semantica/data/test_results/`

Recent test runs with detailed retrieval analysis:

| File | Type | Content | Date |
|------|------|---------|------|
| test-260214-1104-retrieval-all.{json,md} | Full retrieval test | All questions with detailed metrics | 2026-02-14 |
| test-260214-1109-retrieval-q1.{json,md} | Single question | Q1 detailed analysis | 2026-02-14 |
| test-260214-1111-retrieval-q1.{json,md} | Single question | Q1 expanded results | 2026-02-14 |
| test-260214-1114-retrieval-q1-20.{json,md} | Range test | Q1-Q20 retrieval results | 2026-02-14 |

**Format:** JSON for machine parsing, MD for human review

---

## 3. Baseline Scripts

### Script 1: test_baseline.py
**Path:** `/home/hienlh/Projects/semantica/scripts/test_baseline.py`

Implements 5 baseline retrieval methods:

1. **BM25** - Classic keyword-based ranking
   - k1=1.5, b=0.75
   - IDF-weighted term frequencies

2. **TF-IDF** - Term frequency-inverse document frequency
   - Sparse vector cosine similarity
   - Log normalization

3. **Semantic Search** - Pure vector similarity
   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
   - Cosine similarity

4. **Keyword Match** - Simple substring matching
   - Name match: 3.0 weight
   - Content match: 1.0 weight

5. **Semantica Original** - 4-step hybrid
   - Step 1: Vector Retrieval (cosine similarity)
   - Step 2: Graph Retrieval (entity matching + BFS, 2-hop expansion)
   - Step 3: Memory Retrieval (skipped)
   - Step 4: Rank & Merge (hybrid weighting with 20% multi-source boost)

**Usage:**
```bash
# Single question
python scripts/test_baseline.py --question 1 --method all

# Range of questions
python scripts/test_baseline.py --range 1-384 --method tfidf

# Export results
python scripts/test_baseline.py --all --method all --export data/baseline_results.json
```

**Data Classes:**
- `ArticleDoc`: article_id, article_name, content, tokens
- `RetrievalResult`: article_id, article_name, score, method
- `TestResult`: stt, question, expected/retrieved articles, metrics, elapsed_ms

---

### Script 2: test-lightrag-vs-baseline-comparison.py
**Path:** `/home/hienlh/Projects/semantica/scripts/test-lightrag-vs-baseline-comparison.py`

Compares LightRAG against all baseline methods:

**Features:**
- Index documents from SQLite database
- Query with multiple retrieval modes (local, global, hybrid, naive)
- Extract article references from LLM responses
- Compare metrics side-by-side
- Export to JSON with full trace

**Usage:**
```bash
# Index documents
python scripts/test-lightrag-vs-baseline-comparison.py --index --limit 610

# Query and compare
python scripts/test-lightrag-vs-baseline-comparison.py --range 1-384 --mode hybrid --compare-baseline

# Export results
python scripts/test-lightrag-vs-baseline-comparison.py --all --export results.json
```

**Configuration:**
- LLM Provider: OpenAI or Anthropic
- Default Model: claude-3-5-haiku-20241022
- Base URL: http://127.0.0.1:3456

---

### Script 3: test_baseline_extended.py
**Path:** `/home/hienlh/Projects/semantica/scripts/run-ablation-study-for-rag-components.py`

Ablation study for component analysis:

**Components Tested:**
- Tree-only retrieval
- Dual-level retrieval
- Full 3-tier system
- Hit@K metrics (K=1,5,10,20)

---

## 4. Comparison Reports

### Report 1: LightRAG vs Baseline (20 Questions)
**Path:** `/home/hienlh/Projects/semantica/plans/reports/comparison-260128-1734-lightrag-vs-baseline-retrieval.md`
**Date:** 2026-01-28

| Method | Hit Rate | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| LightRAG-hybrid | 90.0% | 5.2% | 81.2% | 9.5% |
| BM25 | 30.0% | 4.0% | 25.0% | 6.5% |
| TF-IDF | 30.0% | 4.3% | 18.3% | 6.6% |
| Semantica (4-step) | 20.0% | 2.0% | 5.0% | 2.9% |

---

### Report 2: LightRAG Full 384-Question Evaluation
**Path:** `/home/hienlh/Projects/semantica/plans/reports/comparison-260128-2100-lightrag-full-384-evaluation.md`
**Date:** 2026-01-28

**Test Set:** All 384 questions, 610 articles indexed
**Key Results:**
- LightRAG Hit Rate: **77.3%** (297/384 questions)
- Baseline (TF-IDF): 42.4%
- **Improvement:** 2.6x higher hit rate, 2.3x higher recall
- Graph: 1,094 entities, 1,897 relationships, 431 chunks

**Perfect Match Examples:**
- Q365: Chuyển đổi loại hình doanh nghiệp (F1=1.00)
- Q366: Hộ kinh doanh khoán (F1=1.00)
- Q368: Kê khai hàng tồn kho (F1=1.00)

---

### Report 3: PageIndex Pure RAG Baseline
**Path:** `/home/hienlh/Projects/semantica/plans/reports/docs-manager-260213-1612-pure-pageindex-baseline-integration.md`
**Date:** 2026-02-13

**Method:** Tree traversal-only retrieval (no KG support)
**Results on 375 questions:** 35.20% Hit rate

**Key Finding:** Significant difference from ablation `tree_only`:
- Pure PageIndex standalone: **35.20%** Hit rate
- tree_only in 3-tier ablation: **68.8%** Hit rate
- **Reason:** KG context significantly boosts Tree component performance

---

### Report 4: Baseline Test Results Scout
**Path:** `/home/hienlh/Projects/semantica/plans/reports/scout-260203-2316-baseline-test-results.md`
**Date:** 2026-02-03

Comprehensive inventory of all baseline and evaluation results:

**Performance Progression Timeline:**
```
Baseline (TF-IDF)           → 42.4% (384Q)
LightRAG Hybrid             → 77.3% (384Q)
Generic Prompt              → 71.0% (100Q)
DualLevel Retriever         → 75.0%
3-Tier Retrieval (Iter. 9)  → 84.0% (CURRENT)
```

**Total improvement:** 42.4% → 84.0% = **+41.6 percentage points**

---

## 5. Advanced Retrieval Comparisons

### Report 5: Retrieval Pipeline Architectures
**Path:** `/home/hienlh/Projects/semantica/plans/reports/comparison-260125-1645-retrieval-pipeline-semantica-vs-legal-onto-ts.md`
**Date:** 2026-01-25

Comparison of three architectures:
1. **Semantica GraphRAG** - Entity-based with graph traversal
2. **Legal-Onto-TS** - Ontology-based with tree traversal
3. **Hybrid** - Combines both approaches

---

### Report 6: PageIndex Hierarchical Retrieval
**Path:** `/home/hienlh/Projects/semantica/plans/reports/researcher-260201-1214-hierarchical-retrieval-improvements-pageindex.md`
**Date:** 2026-02-01

**Method:** PageIndex - Hierarchical document chunking with tree navigation
- Loop 1: Chapter-level navigation with summaries
- Loop 2: Article-level navigation with keywords
- Tree Traversal: Guided navigation through document structure

---

### Report 7: PPR & Ontology Evaluation
**Path:** `/home/hienlh/Projects/semantica/plans/reports/brainstorm-260130-0131-ontology-ppr-retrieval-improvement.md`
**Date:** 2026-01-30

Explores PageRank-based retrieval (PPR) for legal documents

---

## 6. Data Format Documentation

### JSON Result Format

**baseline_results.json / lightrag_results.json:**
```json
{
  "summary": {
    "hit_rate": 0.4141,
    "avg_precision": 0.067,
    "avg_recall": 0.268,
    "ir_metrics_at_k": {
      "1": { "hit": 0.10 },
      "5": { "hit": 0.25 },
      "10": { "hit": 0.4141 }
    }
  },
  "results": [
    {
      "stt": 1,
      "question": "...",
      "expected_articles": [206],
      "retrieved_articles": [22, 60, 206, 208, ...],
      "precision": 0.1,
      "recall": 1.0,
      "f1": 0.181,
      "hit": true
    }
  ]
}
```

### CSV Training Data Format

**training_with_ids.csv:**
```csv
STT,Category,Content,Article_IDs,Điều luật tham chiếu
1,Loại 6,"Hộ kinh doanh là gì?","01/2021/ND:d206:k1","Điều 206"
```

**test_results.csv:**
```csv
STT,Category,Expected,Tree_Articles,KG_Articles,Merged,
Tree_Conf,Tree_Weight,KG_Weight,Tree_Hit,KG_Hit,Final_Hit,
Query_Type,Intent,Keywords,Article_Refs_Detected,Retrieval_Method,
Hybrid_Alpha,Max_Hops,Ontology_Terms,Ontology_Classes,Contexts_Count,
Reasoning,Question
```

---

## 7. Evaluation Methodology

### Metrics Computed

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Hit Rate** | % questions with ≥1 correct article | Recall at document level |
| **Precision** | correct articles / retrieved articles | Specificity of results |
| **Recall** | correct articles / expected articles | Coverage of answers |
| **F1** | Harmonic mean of precision & recall | Overall quality |
| **Hit@K** | Hit rate in top-K results | Performance at cutoff |

### Evaluation Data

**Training Set:** `/home/hienlh/Projects/semantica/data/training/training_with_ids.csv`
- 384 questions total (originally)
- Standardized to 375-379 for different tests
- Categories: Loại 6, Loại 7, etc. (Vietnamese law categories)
- Ground truth: Article IDs with full document references

---

## 8. Key Performance Insights

### 1. Baseline Ceiling (30-42% hit rate)
- **Strength:** Fast, no LLM required, interpretable
- **Weakness:** Missing entity relationships, poor handling of complex queries
- **Best:** TF-IDF at 41.41% on Hit@10

### 2. LightRAG Breakthrough (77.3% hit rate)
- **Strength:** Graph-based retrieval captures semantic relationships
- **Weakness:** Low precision (4.6%) due to returning many related articles
- **Trade-off:** High recall preferred for legal retrieval (want comprehensive results)

### 3. PageIndex Tree Traversal
- **Standalone:** 35.20% hit rate (similar to keyword methods)
- **In 3-tier:** Boosts to 68.8% with KG context
- **Implication:** Tree structure alone insufficient; needs semantic enhancement

### 4. 3-Tier System (Current Best)
- **Architecture:** Tree + DualLevel + Semantic Bridge
- **Hit Rate:** 84% documented, 89% measured
- **Components:**
  - Tier 1: Tree traversal (35%)
  - Tier 2: KG dual-level (68%)
  - Tier 3: Semantic bridge (consolidation)

### 5. Weak Query Categories
From generic prompt analysis:
- **Strong:** Giải thể & Phá sản (95% hit rate)
- **Weak:** Hộ kinh doanh (25%), Thành lập DN (40%)
- **Solution:** Category-specific prompts, few-shot examples

---

## 9. Related Implementation Files

### Wrapper Classes
- `semantica/legal/lightrag_vietnamese_legal_wrapper.py` - LightRAG wrapper
- `semantica/embeddings.py` - EmbeddingGenerator (multilingual-MiniLM-L12-v2)

### Database
- `semantica/legal.py` - LegalDocumentDB class
- Database schema: LegalArticleModel with article_number, title, clauses

### Evaluation
- `semantica/ontology/ontology_evaluator.py` - KG evaluation
- `scripts/analyze_failed_questions.py` - Error analysis

---

## 10. Usage Commands

### Run Baseline Evaluation
```bash
# All baselines on all questions
cd /home/hienlh/Projects/semantica
python scripts/test_baseline.py --all --method all \
  --export data/baseline_results.json

# Specific questions with specific method
python scripts/test_baseline.py --range 1-100 --method tfidf \
  --verbose --export data/tfidf_100q.json

# Compare specific methods
python scripts/test_baseline.py --range 1-50 --method all
```

### Run LightRAG Tests
```bash
# Index documents
python scripts/test-lightrag-vs-baseline-comparison.py \
  --index --limit 610

# Run queries in hybrid mode
python scripts/test-lightrag-vs-baseline-comparison.py \
  --range 1-384 --mode hybrid \
  --compare-baseline --export data/lightrag_full.json
```

### Analyze Results
```bash
# Extract metrics from JSON
python3 -c "
import json
with open('data/consolidated-results/02-baseline-384q-full-results.json') as f:
    data = json.load(f)
for k in [1, 5, 10, 20]:
    hits = sum(1 for r in data['results']['tfidf'] 
               if set(r['expected_articles']) & set(r['retrieved_articles'][:k]))
    print(f'TF-IDF Hit@{k}: {hits}/{len(data[\"results\"][\"tfidf\"])} = {hits/len(data[\"results\"][\"tfidf\"])*100:.1f}%')
"
```

---

## 11. Unresolved Questions

1. **PPR Results:** `/home/hienlh/Projects/semantica/data/consolidated-results/ppr_full_results.json` exists but metrics not extracted - may be incomplete
2. **Ablation Study Details:** Exact performance breakdown for tree_only vs dual_only vs full components
3. **Perfect Match Count:** Iteration 9 reports mention "several perfect matches" - need exact count
4. **Re-evaluation Gap:** LightRAG 04-reeval shows larger file size - what changed in re-evaluation methodology?

---

## Summary of Key Files

| Category | Path | Key Metric |
|----------|------|-----------|
| **Baseline Results** | `/home/hienlh/Projects/semantica/data/consolidated-results/02-baseline-384q-full-results.json` | TF-IDF: 41.41% Hit@10 |
| **LightRAG Results** | `/home/hienlh/Projects/semantica/data/consolidated-results/03-lightrag-384q-full-results.json` | 77.3% Hit rate |
| **Current System** | `/home/hienlh/Projects/semantica/data/consolidated-results/05-current-system-375q-full-results.json` | 76.53% Hit@10, 92.53% Hit@all |
| **PageIndex Pure** | `/home/hienlh/Projects/semantica/data/consolidated-results/10-pure-pageindex-rag-375q-results.json` | 35.20% Hit rate |
| **Baseline Script** | `/home/hienlh/Projects/semantica/scripts/test_baseline.py` | 5 baseline methods implemented |
| **Comparison Script** | `/home/hienlh/Projects/semantica/scripts/test-lightrag-vs-baseline-comparison.py` | LightRAG + baseline comparison |
| **Evaluation Report** | `/home/hienlh/Projects/semantica/plans/reports/scout-260203-2316-baseline-test-results.md` | Comprehensive inventory |

---

**Report Generated:** 2026-02-15 12:39 UTC  
**Report Location:** /home/hienlh/Projects/vn_legal_rag/plans/reports/scout-260215-1239-semantica-rag-benchmarks.md
