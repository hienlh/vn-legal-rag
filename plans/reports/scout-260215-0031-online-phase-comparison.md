# Scout Report: Online Phase Comparison

**Date:** 2026-02-15 00:31
**Comparing:** `vn_legal_rag` vs `semantica`

---

## Summary

| Aspect | vn_legal_rag | semantica |
|--------|--------------|-----------|
| Location | `vn_legal_rag/online/` | `semantica/legal/` |
| Main File | `legal-graphrag-3tier-query-engine.py` (~870 LOC) | `graphrag.py` (~1900 LOC) |
| Architecture | **Modular** (9 files) | **Monolithic** (1 main file) |
| File Naming | kebab-case + importlib | Standard Python naming |
| Ecosystem | Standalone | Integrated (AgentContext, ContextGraph) |

---

## Module Structure

### vn_legal_rag (New - Modular)
```
vn_legal_rag/online/
├── __init__.py
├── legal-graphrag-3tier-query-engine.py     # Main entry (LegalGraphRAG)
├── tree-traversal-retriever.py              # Tier 1
├── dual-level-retriever.py                  # Tier 2
├── semantic-bridge-rrf-merger.py            # Tier 3
├── vietnamese-legal-query-analyzer.py       # Query analysis
├── personalized-page-rank-for-kg.py         # PPR scoring
├── ontology-based-query-expander.py         # Query expansion
├── document-aware-result-filter.py          # Doc filtering
└── cross-encoder-reranker-for-legal-documents.py  # Reranking
```

### semantica (Original - Monolithic)
```
semantica/legal/
├── graphrag.py                              # Main + all merge logic (~1900 LOC)
├── tree_traversal_retriever.py              # Tier 1
├── dual_level_retriever.py                  # Tier 2
├── query_analyzer.py                        # Query analysis
├── query_type_classifier.py                 # Type classification
├── ontology_expander.py                     # Ontology expansion
├── provenance_enricher.py                   # Citation formatting
├── db_manager.py                            # Database
└── cross_encoder_reranker_for_legal_documents.py
```

---

## 3-Tier Retrieval Architecture (Common)

Both projects implement identical 3-tier architecture:

```
Query → Analyzer → [Tier 1 + Tier 2 parallel] → Tier 3 → Response
                        ↓           ↓              ↓
                   Tree Nav    DualLevel     Semantic Bridge
                  (3-Loop)   (6-component)    (RRF Merge)
```

### Tier 1: Tree Traversal (Identical)
- Loop 0: Document selection (multi-doc support)
- Loop 1: Chapter navigation (LLM-guided)
- Loop 2: Article selection (keyword + semantic)
- **Performance:** Hit Rate 68.8% standalone

### Tier 2: DualLevel (Identical)
- 6-component scoring: keyphrase, semantic, PPR, concept, theme, hierarchy
- Global search across all articles
- **Performance:** 19.47% standalone, +18.7% boost when combined

### Tier 3: Semantic Bridge (Mostly Identical)
- RRF (Reciprocal Rank Fusion) with k=60
- Source weights: tree=0.8, dual=1.0, kg=1.2
- Cross-chapter linking via KG relations
- Adaptive threshold based on Tree-DualLevel agreement

---

## Key Differences

### 1. Code Organization

| Feature | vn_legal_rag | semantica |
|---------|--------------|-----------|
| Merge logic | Separate file (`semantic-bridge-rrf-merger.py`) | Inline in graphrag.py |
| Helper methods | Distributed across modules | All in graphrag.py |
| Import style | `importlib.import_module()` for kebab-case | Standard imports |
| LOC in main | ~870 | ~1900 |

### 2. Features (semantica has more)

| Feature | vn_legal_rag | semantica |
|---------|--------------|-----------|
| Adjacent article expansion | ❌ Not implemented | ✅ `_expand_adjacent_articles()` |
| LegalProvenanceEnricher | ❌ Basic citation | ✅ Full citation formatting |
| AgentContext integration | ❌ | ✅ Hybrid retrieval via AgentContext |
| ContextGraph | ❌ | ✅ Graph-based context |
| LLM caching | ✅ Via `llm_cache_db` | ✅ Via `cache_db_path` |
| Cross-encoder reranker | ✅ | ✅ |
| Ablation config | ✅ Basic | ✅ More options |

### 3. RRF Merge Implementation

**vn_legal_rag** (in `semantic-bridge-rrf-merger.py`):
```python
# Simpler, uses SemanticBridge class
def merge_tree_dual_results(tree_result, dual_result, kg_results, enable_adjacent):
    # RRF merge without adjacent expansion
```

**semantica** (in `graphrag.py`):
```python
# More comprehensive with adjacent expansion
def _merge_tree_dual_kg(tree_result, dual_result, kg_results, enable_adjacent=True):
    # RRF merge + adjacent article expansion
    adjacent_results = self._expand_adjacent_articles(...)
```

### 4. Cross-validation Logic

**vn_legal_rag** (`_retrieve_contexts`):
```python
# Overlap detection + chapter disagreement expansion
if overlap_ratio < 0.4:
    tree_chapters = self._get_chapters_from_articles(tree_result.target_nodes)
    dual_chapters = self._get_chapters_from_article_ids([...])
    new_chapters = dual_chapters - tree_chapters
    self._expand_to_new_chapters(tree_result, dual_top_articles, new_chapters)
```

**semantica** (identical logic, different location):
```python
# Same logic in graphrag.py:_retrieve_contexts
if overlap_ratio < 0.4:
    # Chapter disagreement expansion...
```

### 5. Database Access

**vn_legal_rag:**
```python
from vn_legal_rag.offline import LegalDocumentDB
# Supports both db object and db_path string
if db is not None:
    self.db = db
elif db_path is not None:
    self.db = LegalDocumentDB(db_path)
```

**semantica:**
```python
from .db_manager import LegalDocumentDB
# Uses enricher for article access
self.enricher = LegalProvenanceEnricher(self.db)
article_ctx = self.enricher.get_article_context(article_num, doc_id)
```

---

## ✅ Features Added (2026-02-15)

The following features from `semantica` have been ported to `vn_legal_rag`:

### 1. Adjacent Article Expansion ✅
- **File:** `semantic-bridge-rrf-merger.py`
- **Method:** `_expand_adjacent_articles()`
- Recovers close-miss cases (articles within ±2)
- Used in `merge_tree_dual_results()` when `enable_adjacent=True`

### 2. Dual+KG Fallback Merge ✅
- **File:** `semantic-bridge-rrf-merger.py`
- **Method:** `merge_dual_and_kg_results()`
- Used when tree is unavailable (ablation configs: no_tree, dual_only)

### 3. No-Tree Fallback Handling ✅
- **File:** `legal-graphrag-3tier-query-engine.py`
- **Method:** `_retrieve_contexts()`
- Added Case 2: No tree but DualLevel available → use DualLevel as primary

### 4. Cross-Encoder Reranking ✅
- **File:** `legal-graphrag-3tier-query-engine.py`
- **Method:** `_apply_reranking()`
- Stage 2 reranking with CrossEncoderReranker
- Controlled by `ablation_config.enable_reranker`

---

## Remaining Differences (Lower Priority)

| Feature | Status | Notes |
|---------|--------|-------|
| LegalProvenanceEnricher | ❌ Not ported | Rich citation formatting |
| AgentContext/ContextGraph | ❌ Not ported | Semantica ecosystem integration |
| Detailed logging | ❌ Not ported | Component contribution logs |

---

## Performance Notes

Both projects target:
- **Hit Rate (Hit@10):** 76.53%
- **Hit Rate (Hit@all):** 92.53%
- **MRR:** 0.5636

vn_legal_rag documentation mentions:
- Hit Rate: 78.93%, Recall@5: 58.12%, MRR: 0.5422

Slight differences may be due to:
- Missing adjacent expansion in vn_legal_rag
- Different test dataset sizes
- Configuration differences

---

## Unresolved Questions

1. Was adjacent expansion intentionally omitted from vn_legal_rag?
2. Should vn_legal_rag integrate with a broader ecosystem like semantica does?
3. Are the performance metrics comparable given different features?
