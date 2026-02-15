# Scout Report: vn_legal_rag vs semantica

**Date:** 2026-02-14
**Objective:** Kiểm tra xem vn_legal_rag đã đủ mọi thứ để chạy `run-full-training-test.py` từ semantica chưa

---

## Summary

| Category | Status | Notes |
|----------|--------|-------|
| **Data Files** | ✅ Đầy đủ | Tất cả data files có sẵn |
| **Core Classes** | ⚠️ Có nhưng khác API | Interface khác semantica |
| **Ablation Testing** | ❌ Thiếu | Cần `AblationConfig` |
| **Utilities** | ❌ Thiếu | Cần `ProgressTracker` |

**Kết luận:** CHƯA thể chạy trực tiếp. Cần điều chỉnh hoặc viết script mới.

---

## 1. Data Files (✅ Match)

| File | semantica | vn_legal_rag | Status |
|------|-----------|--------------|--------|
| `data/legal_docs.db` | 10 MB | 10 MB | ✅ |
| `data/kg_enhanced/legal_kg.json` | 2.1 MB | 2.1 MB | ✅ |
| `data/kg_enhanced/chapter_summaries.json` | 134 KB | 134 KB | ✅ |
| `data/kg_enhanced/article_summaries.json` | 636 KB | 636 KB | ✅ |
| `data/kg_enhanced/document_summaries_loop0.json` | 8.3 KB | 8.3 KB | ✅ |
| `data/training/training_with_ids.csv` | 564 KB | 564 KB | ✅ |

**Kết luận:** Data files đầy đủ và giống hệt nhau.

---

## 2. Core Classes

### 2.1 Database & Models (✅ Match)

| Class | semantica | vn_legal_rag | Status |
|-------|-----------|--------------|--------|
| `LegalDocumentDB` | `legal/db_manager.py` | `offline/database_manager.py` | ✅ |
| `LegalDocumentModel` | `legal/models.py` | `offline/models.py` | ✅ |
| `LegalArticleModel` | `legal/models.py` | `offline/models.py` | ✅ |
| `LegalChapterModel` | `legal/models.py` | `offline/models.py` | ✅ |

### 2.2 Tree Models (✅ Match)

| Class | semantica | vn_legal_rag | Status |
|-------|-----------|--------------|--------|
| `TreeNode` | `legal/tree_node_models.py` | `types/tree_models.py` | ✅ |
| `TreeIndex` | `legal/tree_node_models.py` | `types/tree_models.py` | ✅ |
| `UnifiedForest` | `legal/tree_node_models.py` | `types/tree_models.py` | ✅ |
| `NodeType` | `legal/tree_node_models.py` | `types/tree_models.py` | ✅ |

### 2.3 GraphRAG (⚠️ Interface khác)

| Component | semantica | vn_legal_rag |
|-----------|-----------|--------------|
| **Class** | `LegalGraphRAG` | `LegalGraphRAG` |
| **File** | `legal/graphrag.py` | `online/legal-graphrag-3tier-query-engine.py` |
| **Response** | `LegalGraphRAGResponse` | `GraphRAGResponse` |

**Interface Differences:**

```python
# semantica
LegalGraphRAG(
    kg=kg,
    db_path=args.db,          # string path
    llm_provider=args.provider,  # string: "openai"/"anthropic"
    llm_model=args.model,        # string: "gpt-4o-mini"
    llm_base_url=args.base_url,  # string: "http://..."
    forest=forest,
    article_summaries=...,
    document_summaries=...,
    ablation_config=ablation_config,  # ❌ vn_legal_rag không có
)

# vn_legal_rag
LegalGraphRAG(
    kg=kg,
    forest=forest,
    db=db,                    # object, not path
    llm_provider=llm_provider,   # object, not string
    embedding_gen=embedding_gen,
    article_summaries=...,
    document_summaries=...,
    config=config,  # dict, not AblationConfig
)
```

**Response Differences:**

```python
# semantica LegalGraphRAGResponse
result.tree_search_result.target_nodes  # List[TreeNode]
result.tree_search_result.confidence    # float
result.tree_search_result.reasoning_path # List[str]
result.intent                           # QueryIntent enum
result.intent.value                     # string
result.query_type                       # LegalQueryType enum
result.citations                        # List[Dict]
result.metadata                         # Dict

# vn_legal_rag GraphRAGResponse
result.reasoning_path        # List[str] (flat)
result.confidence            # float
result.query_type            # LegalQueryType enum
result.citations             # List[Dict]
result.metadata              # Dict (contains tree_confidence, intent)
# ❌ Missing: result.tree_search_result, result.intent
```

---

## 3. Missing Components

### 3.1 AblationConfig (❌ Missing)

**Cần từ semantica:**
```
semantica/legal/ablation-config-for-rag-component-testing.py
```

**Classes cần:**
- `AblationConfig` - dataclass cho ablation testing
- `get_paper_ablation_configs()` - function trả về dict configs

**Dùng trong test script:**
```python
from semantica.legal import AblationConfig, get_paper_ablation_configs
ablation_configs = get_paper_ablation_configs()
rag = LegalGraphRAG(..., ablation_config=ablation_configs["full"])
```

### 3.2 ProgressTracker (❌ Missing)

**Cần từ semantica:**
```
semantica/utils/progress_tracker.py
```

**Dùng trong test script:**
```python
from semantica.utils.progress_tracker import ProgressTracker
ProgressTracker.get_instance().disable()  # Tắt logging khi test
```

---

## 4. Khuyến nghị

### Option A: Port Missing Components
1. Copy `ablation-config-for-rag-component-testing.py` → `vn_legal_rag/types/`
2. Copy `progress_tracker.py` → `vn_legal_rag/utils/`
3. Cập nhật `LegalGraphRAG` để nhận `ablation_config` parameter
4. Thêm `tree_search_result` và `intent` vào `GraphRAGResponse`

### Option B: Viết Script Mới (Recommended)
Tạo `scripts/evaluate-retrieval-performance-on-test-set.py` mới tương thích với vn_legal_rag API hiện tại.

### Option C: Adapter Pattern
Tạo wrapper class `SemanticaCompatibleGraphRAG` để bridge giữa 2 APIs.

---

## 5. File Structure Comparison

```
semantica/                          vn_legal_rag/
├── semantica/                      ├── vn_legal_rag/
│   ├── legal/                      │   ├── online/          # RAG components
│   │   ├── __init__.py             │   │   ├── legal-graphrag-3tier-query-engine.py
│   │   ├── graphrag.py             │   │   ├── tree-traversal-retriever.py
│   │   ├── db_manager.py           │   │   ├── dual-level-retriever.py
│   │   ├── models.py               │   │   └── semantic-bridge-rrf-merger.py
│   │   ├── tree_node_models.py     │   ├── offline/         # KG building
│   │   ├── ablation-config-...py   │   │   ├── database_manager.py
│   │   └── ...                     │   │   ├── models.py
│   ├── utils/                      │   │   └── ...
│   │   ├── progress_tracker.py     │   ├── types/           # Type definitions
│   │   └── ...                     │   │   ├── tree_models.py
│   └── ...                         │   │   ├── entity_types.py
├── scripts/                        │   │   └── relation_types.py
│   └── run-full-training-test.py   │   └── utils/           # Utilities
└── data/                           ├── scripts/
    ├── legal_docs.db               │   └── evaluate-retrieval-performance-on-test-set.py
    ├── kg_enhanced/                └── data/
    │   ├── legal_kg.json               ├── legal_docs.db
    │   ├── chapter_summaries.json      ├── kg_enhanced/
    │   ├── article_summaries.json      └── training/
    │   └── document_summaries_loop0.json
    └── training/
        └── training_with_ids.csv
```

---

## 6. Quick Fix for Testing

Nếu chỉ muốn test nhanh, có thể:

1. **Tạo dummy AblationConfig:**
```python
@dataclass
class AblationConfig:
    enable_tree: bool = True
    enable_dual_level: bool = True
    enable_reranker: bool = True
    name: str = "full"

def get_paper_ablation_configs():
    return {"full": AblationConfig()}
```

2. **Tạo dummy ProgressTracker:**
```python
class ProgressTracker:
    @staticmethod
    def get_instance():
        return ProgressTracker()
    def disable(self):
        pass
```

3. **Điều chỉnh script để dùng vn_legal_rag API**

---

## Unresolved Questions

1. vn_legal_rag có cần hỗ trợ ablation testing không?
2. Có nên merge TreeSearchResult vào response type không?
3. API nào nên là chuẩn cho future development?
