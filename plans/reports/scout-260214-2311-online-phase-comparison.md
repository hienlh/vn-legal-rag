# Scout Report: Online Phase Comparison

**vn_legal_rag vs semantica**

## Summary

| Aspect | vn_legal_rag | semantica |
|--------|--------------|-----------|
| Directory | `vn_legal_rag/online/` | `semantica/legal/` |
| Naming | kebab-case | snake_case |
| Module count | 9 dedicated files | Mixed with offline |
| Main entry | `LegalGraphRAG` (3-tier engine) | `LegalGraphRAG` (AgentContext-based) |

---

## Module Structure

### vn_legal_rag (Current Project)
```
vn_legal_rag/online/
├── __init__.py                              # Exports all components
├── legal-graphrag-3tier-query-engine.py     # ⭐ Main orchestrator (873 LOC)
├── tree-traversal-retriever.py              # Tier 1: Tree nav (640 LOC)
├── dual-level-retriever.py                  # Tier 2: 6-component scoring
├── semantic-bridge-rrf-merger.py            # Tier 3: RRF fusion
├── vietnamese-legal-query-analyzer.py       # Query analysis + expansion
├── personalized-page-rank-for-kg.py         # PPR for KG scoring
├── ontology-based-query-expander.py         # Ontology expansion
├── document-aware-result-filter.py          # Result filtering
└── cross-encoder-reranker-for-legal-documents.py  # Stage-2 reranking
```

### semantica (Original Project)
```
semantica/legal/
├── __init__.py                              # 560+ exports (mixed offline/online)
├── graphrag.py                              # ⭐ Main entry (simpler)
├── tree_traversal_retriever.py              # 977 LOC - more features
├── dual_level_retriever.py                  # DualLevel retrieval
├── query_analyzer.py                        # Query analysis
├── query_expander_vietnamese_legal.py       # Query expansion
├── personalized-page-rank-for-kg-article-scoring.py
├── ontology_expander.py                     # Ontology expansion
└── document_filter.py                       # Document filtering
```

---

## Key Differences

### 1. LegalGraphRAG Architecture

**vn_legal_rag** - `legal-graphrag-3tier-query-engine.py` (873 LOC)
- Sophisticated cross-validation between Tree and DualLevel
- Adaptive threshold computation based on overlap
- Ambiguity calibration for vague queries
- KG relation expansion for cross-chapter coverage
- Full ablation config support
- Methods:
  - `_retrieve_contexts()` - 3-tier with cross-validation
  - `_expand_via_kg_relations()` - cross-chapter linking
  - `_compute_adaptive_threshold()` - dynamic threshold
  - `_compute_ambiguity_calibration()` - query ambiguity handling

**semantica** - `graphrag.py` (~400 LOC)
- Simpler, built on `AgentContext` base class
- Less sophisticated result merging
- No cross-validation logic
- Relies on TreeTraversalRetriever for smart selection

### 2. TreeTraversalRetriever

| Feature | vn_legal_rag | semantica |
|---------|--------------|-----------|
| LLM Provider | Object (pre-created) | String → internal creation |
| Semantic Scoring | ❌ Missing | ✅ `_add_semantic_scores()` |
| DualLevel Scoring | ❌ Missing | ✅ `_add_duallevel_scores()` |
| Chapter Hints | ❌ Missing | ✅ `_detect_chapter_hint()` |
| Domain Config | Basic | Full integration |
| LLM Caching | Delegated to provider | Native support |
| Logging | No | `get_logger()` |
| LOC | ~640 | ~977 |

**semantica's extra methods in Loop 2:**
```python
# In semantica's tree_traversal_retriever.py
def _add_semantic_scores(self, query, article_infos):
    """Add embedding similarity scores and ranks to articles."""
    # Computes cosine similarity with embeddings
    # Adds both 'semantic_score' and 'semantic_rank'

def _add_duallevel_scores(self, query, articles, article_infos):
    """Add KG+embedding scores from DualLevelRetriever."""
    # Uses DualLevelRetriever.retrieve() for combined scoring
    # Adds 'dual_score' and 'dual_rank' fields
```

### 3. LLM Provider Handling

**vn_legal_rag:**
```python
# LegalGraphRAG creates provider, passes object to TreeRetriever
llm_provider = create_llm_provider(provider="anthropic", model="claude-3-5-haiku")
tree_retriever = TreeTraversalRetriever(forest=forest, llm_provider=llm_provider)
```

**semantica:**
```python
# TreeRetriever creates provider internally from strings
tree_retriever = TreeTraversalRetriever(
    forest=forest,
    llm_provider="anthropic",       # String
    llm_model="claude-3-5-haiku",   # String
    llm_base_url="http://127.0.0.1:3456"
)
```

### 4. Import Paths

| Component | vn_legal_rag | semantica |
|-----------|--------------|-----------|
| Tree models | `vn_legal_rag.types.tree_models` | `semantica.legal.tree_node_models` |
| LLM factory | `vn_legal_rag.utils.create_llm_provider` | `semantica.semantic_extract.providers.create_provider` |
| DB manager | `vn_legal_rag.offline.LegalDocumentDB` | `semantica.legal.db_manager.LegalDocumentDB` |
| Logging | None | `semantica.utils.logging.get_logger` |

---

## Features Comparison Matrix

| Feature | vn_legal_rag | semantica |
|---------|--------------|-----------|
| 3-Tier Architecture | ✅ | ✅ |
| Loop 0 (Document Selection) | ✅ | ✅ |
| Loop 1 (Chapter Selection) | ✅ | ✅ |
| Loop 2 (Article Selection) | ✅ Basic | ✅ Enhanced |
| Tree-Dual Cross-Validation | ✅ In LegalGraphRAG | ❌ |
| Adaptive Threshold | ✅ In LegalGraphRAG | ❌ |
| Ambiguity Calibration | ✅ In LegalGraphRAG | ❌ |
| Semantic Scoring (Loop 2) | ❌ | ✅ |
| DualLevel Scoring (Loop 2) | ❌ | ✅ |
| Chapter Hints (Config) | ❌ | ✅ |
| LLM Response Caching | Via provider | Native |
| Ablation Config | ✅ Full support | ✅ |
| Kebab-case Filenames | ✅ | ❌ |

---

## Recommendations

### Missing in vn_legal_rag TreeRetriever:
1. **Semantic Scoring** - `_add_semantic_scores()` method
2. **DualLevel Scoring** - `_add_duallevel_scores()` method
3. **Chapter Hints** - `_detect_chapter_hint()` method
4. **Logging** - get_logger integration
5. **Native LLM caching** - currently delegated to provider

### Architecture Trade-off:
- **vn_legal_rag**: Complex logic in `LegalGraphRAG` (cross-validation, adaptive threshold)
- **semantica**: Complex logic in `TreeTraversalRetriever` (semantic scoring, dual scoring)

Both approaches valid - vn_legal_rag centralizes orchestration, semantica distributes intelligence.

---

## Relevant Files

### vn_legal_rag
- [legal-graphrag-3tier-query-engine.py](vn_legal_rag/online/legal-graphrag-3tier-query-engine.py) - Main orchestrator
- [tree-traversal-retriever.py](vn_legal_rag/online/tree-traversal-retriever.py) - Tier 1
- [__init__.py](vn_legal_rag/online/__init__.py) - Module exports

### semantica
- [graphrag.py](~/Projects/semantica/semantica/legal/graphrag.py) - Main entry
- [tree_traversal_retriever.py](~/Projects/semantica/semantica/legal/tree_traversal_retriever.py) - Enhanced retriever
- [__init__.py](~/Projects/semantica/semantica/legal/__init__.py) - 560+ exports

---

## Unresolved Questions
1. Should semantic/dual scoring be added to vn_legal_rag's TreeRetriever?
2. Should cross-validation logic be moved to TreeRetriever for consistency?
3. Is the trade-off between centralized (vn_legal_rag) vs distributed (semantica) orchestration intentional?
