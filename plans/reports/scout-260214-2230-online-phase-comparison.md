# Scout Report: Online Phase Comparison

**Date:** 2026-02-14
**Projects:** `vn_legal_rag` vs `semantica`

---

## Summary

Cả hai project đều có online phase với **kiến trúc 3-tier retrieval** tương tự, nhưng `vn_legal_rag` là phiên bản **simplified & refactored** từ `semantica`.

| Aspect | vn_legal_rag | semantica |
|--------|--------------|-----------|
| **Structure** | Modular với kebab-case files | Monolithic trong `semantica/legal/` |
| **Lines of Code** | ~760 (graphrag engine) | ~1900 (graphrag) |
| **Dependencies** | Standalone package | Phụ thuộc vào `semantica.*` |
| **Features** | Core 3-tier | Full 3-tier + ablation + reranking |

---

## 1. Module Organization

### vn_legal_rag (`vn_legal_rag/online/`)
```
online/
├── __init__.py                         # Clean re-exports
├── legal-graphrag-3tier-query-engine.py  # Main GraphRAG (~760 LOC)
├── tree-traversal-retriever.py         # Tier 1
├── dual-level-retriever.py             # Tier 2 (~473 LOC)
├── semantic-bridge-rrf-merger.py       # Tier 3
├── vietnamese-legal-query-analyzer.py  # Query analysis
└── personalized-page-rank-for-kg.py    # PPR
```

### semantica (`semantica/legal/`)
```
legal/
├── __init__.py                         # 560 LOC exports!
├── graphrag.py                         # Main GraphRAG (~1900 LOC)
├── tree_traversal_retriever.py         # Tier 1
├── dual_level_retriever.py             # Tier 2 (~650 LOC)
├── document_filter.py                  # Document filtering
├── cross_encoder_reranker_for_legal...py # Stage-2 reranking
├── personalized-page-rank-for-kg...py  # PPR
└── ... (50+ other files)
```

**Key Difference:**
- `vn_legal_rag`: **6 focused files** với kebab-case naming
- `semantica`: **50+ files** trong một module, mixed naming conventions

---

## 2. LegalGraphRAG Class Comparison

### 2.1 Initialization

| Feature | vn_legal_rag | semantica |
|---------|--------------|-----------|
| **API Style** | Flexible (object or string params) | String-based config |
| **LLM Provider** | Factory pattern via `create_llm_provider()` | Inline `create_provider()` |
| **Caching** | Optional `llm_cache_db` param | Built-in `use_cache` + `cache_db_path` |
| **Ablation Config** | Direct `AblationConfig` | Full ablation with all flags |
| **Reranker** | ❌ Not included | ✅ CrossEncoderReranker |
| **Document Filter** | ❌ Not in retriever | ✅ DocumentFilter in DualLevel |

### 2.2 Query Flow

**vn_legal_rag:**
```python
# Simpler flow
Query → Analyzer → _retrieve_contexts → _generate_response → GraphRAGResponse
```

**semantica:**
```python
# Extended flow with more features
Query → Sanitize → Type Classify → Analyze → Ontology Expand →
       _retrieve_contexts (3-tier) → Enrich → _build_reasoning_path →
       _generate_response → LegalGraphRAGResponse
```

### 2.3 Retrieval Logic

Cả hai đều có 3-tier với cross-validation, nhưng `semantica` có thêm:

| Feature | vn_legal_rag | semantica |
|---------|--------------|-----------|
| **Query Type Classification** | ❌ | ✅ 6 query types |
| **Ontology Expansion** | ❌ | ✅ OntologyExpander |
| **Cross-Encoder Reranking** | ❌ | ✅ Stage-2 reranking |
| **RRF Fusion** | ✅ Basic | ✅ Full với MIN_RRF_THRESHOLD |
| **Adjacent Expansion** | ✅ | ✅ với ablation flag |
| **Adaptive Threshold** | ❌ | ✅ `_compute_adaptive_threshold` |
| **Ambiguity Calibration** | ❌ | ✅ `_compute_ambiguity_calibration` |

---

## 3. DualLevelRetriever Comparison

### 3.1 Config Defaults

| Weight | vn_legal_rag | semantica |
|--------|--------------|-----------|
| concept_weight | 0.20 | 0.20 |
| semantic_weight | 0.20 | 0.20 |
| ppr_weight | 0.25 | 0.25 |
| keyphrase_weight | 0.05 | 0.05 |
| theme_weight | 0.15 | 0.15 |
| hierarchy_weight | 0.15 | 0.15 |

**Weights giống nhau** - `vn_legal_rag` copy optimized weights từ semantica.

### 3.2 Features

| Feature | vn_legal_rag | semantica |
|---------|--------------|-----------|
| **Document Filter** | ❌ | ✅ `enable_document_filter` |
| **Filter Modes** | - | "boost", "filter", "rerank" |
| **Ablation Flags** | ✅ 8 flags | ✅ 8 flags + `apply_ablation()` |
| **Logger** | ❌ | ✅ `get_logger()` |
| **Factory Function** | ❌ | ✅ `create_dual_level_retriever()` |

---

## 4. Performance Metrics

### semantica (documented):
- **Hit Rate (Hit@10):** 76.53%
- **Hit Rate (Hit@all):** 92.53%
- **MRR:** 0.5636

### vn_legal_rag (from `__init__.py`):
- **Hit Rate:** 78.93%
- **Recall@5:** 58.12%
- **MRR:** 0.5422

**Lưu ý:** `vn_legal_rag` báo cáo metrics cao hơn một chút nhưng dùng different test set/config.

---

## 5. Key Differences Summary

### vn_legal_rag có:
1. ✅ **Cleaner structure** - 6 files vs 50+
2. ✅ **Kebab-case naming** - LLM-friendly
3. ✅ **Flexible API** - Both object & string params
4. ✅ **Standalone** - No external semantica dependencies
5. ❌ **Missing:** Query type classification, ontology expansion, reranking

### semantica có:
1. ✅ **Full feature set** - Query types, ontology, reranking
2. ✅ **Ablation study support** - Complete flags
3. ✅ **Better logging** - `get_logger()` throughout
4. ✅ **Document filtering** - In DualLevelRetriever
5. ❌ **Complex:** 50+ files, harder to maintain

---

## 6. Code Quality

### vn_legal_rag
- Imports via `import_module()` for kebab-case files ✅
- Clear `__all__` exports ✅
- Minimal dependencies ✅
- Missing docstrings in some places ❌

### semantica
- Rich docstrings throughout ✅
- Comprehensive logging ✅
- Full ablation support ✅
- Module coupling (imports many internal modules) ❌

---

## 7. Recommendations

### If you need simplicity → Use `vn_legal_rag`
- Cleaner codebase
- Easier to understand
- Core 3-tier retrieval works well

### If you need full features → Use `semantica`
- Query type classification
- Ontology expansion
- Cross-encoder reranking
- Complete ablation study support

### Migration Path
Nếu muốn port features từ `semantica` sang `vn_legal_rag`:

1. **Query Type Classification:** Copy `query_type_classifier.py`
2. **Ontology Expansion:** Copy `ontology_expander.py`
3. **Reranking:** Copy `cross_encoder_reranker_for_legal_documents.py`
4. **Document Filter:** Copy `document_filter.py`

---

## Relevant Files

### vn_legal_rag
- [vn_legal_rag/online/__init__.py](vn_legal_rag/online/__init__.py) - Module exports
- [vn_legal_rag/online/legal-graphrag-3tier-query-engine.py](vn_legal_rag/online/legal-graphrag-3tier-query-engine.py) - Main GraphRAG
- [vn_legal_rag/online/dual-level-retriever.py](vn_legal_rag/online/dual-level-retriever.py) - Tier 2

### semantica
- `/home/hienlh/Projects/semantica/semantica/legal/__init__.py` - 560 LOC exports
- `/home/hienlh/Projects/semantica/semantica/legal/graphrag.py` - Full GraphRAG
- `/home/hienlh/Projects/semantica/semantica/legal/dual_level_retriever.py` - Tier 2 + Document Filter
- `/home/hienlh/Projects/semantica/docs/legal-pipeline-online.md` - Architecture docs

---

## Unresolved Questions

1. **Why different MRR?** (0.5422 vs 0.5636) - Possibly different test sets or configs
2. **Document Filter impact?** - semantica has it, vn_legal_rag doesn't
3. **Reranking impact?** - CrossEncoderReranker chỉ có trong semantica
