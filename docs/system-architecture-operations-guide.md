# System Architecture: Operations & Extensibility Guide

**Project**: Vietnamese Legal RAG (vn_legal_rag)
**Last Updated**: 2026-02-14
**Related**: [System Architecture (Main)](./system-architecture.md)

## Performance Characteristics

### Query Latency Breakdown (Empirical)

| Phase | Latency | Notes |
|-------|---------|-------|
| Tier 0 (Query analysis) | 0.1-0.3s | LLM classification |
| Tier 1 (Tree traversal) | 0.5-2s | LLM 2 loops + summaries |
| Tier 2 (DualLevel) | 1-3s | Embedding inference + scoring |
| Tier 3 (RRF + KG exp.) | 0.2-0.5s | Graph traversal |
| LLM Answer gen. | 2-5s | GPT-4o-mini generation |
| Total (p95) | 5-10s | Combined end-to-end |

**Optimization strategies**:
- Cache embeddings (Tier 2)
- Cache LLM responses (llm_provider_with_caching.py)
- Pre-compute summaries (Tier 1)
- Batch KG lookups (Tier 3)

### Accuracy by Query Type

| Query Type | Tier 1 | Tier 2 | Tier 1+2 | Tier 3 |
|------------|--------|--------|----------|--------|
| article_lookup | 82.05% | 65.38% | 84.62% | 85.26% |
| guidance_document | 61.22% | 72.45% | 74.49% | 75.51% |
| situation_analysis | 60.00% | 73.33% | 73.33% | 74.67% |
| compare_regulations | 50.00% | 68.75% | 71.88% | 75.00% |
| case_law_lookup | 50.00% | 66.67% | 72.22% | 72.22% |

**Insights**:
- Tier 1 excels on direct lookups; weak on comparisons
- Tier 2 strong on semantic search; moderate on direct lookups
- Tier 3 improves all types slightly; best for comparisons

## Failure Modes & Resilience

### Tier 1 Failure Recovery

**If Tier 1 returns empty**:
1. Log warning
2. Fallback to Tier 2 results only
3. Skip RRF fusion (Tier 3 needs both)
4. Return best single-tier results

**Root causes**:
- Query matches no chapter summaries
- LLM rejects all articles
- Forest is empty/corrupted

### Tier 2 Failure Recovery

**If Tier 2 returns empty**:
1. Reduce embedding threshold
2. Use keyword-only scoring
3. Return all articles (unranked) as last resort

**Root causes**:
- Embedding model fails
- No keyword matches
- Negative PPR scores

### Tier 3 Graceful Degradation

**If KG expansion fails**:
1. Skip expansion, keep RRF results
2. Skip RRF, return union of tier results
3. Return top-k from whichever tier succeeded

## Extensibility & Future Work

### Adding a New Tier

```python
# Step 1: Create new module
# vn_legal_rag/online/new-tier-retriever.py

class NewTierRetriever:
    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        ...

# Step 2: Integrate into LegalGraphRAG
# vn_legal_rag/online/legal-graphrag-3tier-query-engine.py

from importlib import import_module
_new_tier = import_module(".new-tier-retriever", "vn_legal_rag.online")
NewTierRetriever = _new_tier.NewTierRetriever

# Step 3: Add to query pipeline
def query(self, query_text: str, ...):
    if not self.ablation_config.disable_new_tier:
        new_tier_results = self.new_tier_retriever.retrieve(query_text)
    # ... integrate with other tiers
```

### Supporting New Legal Domains

1. Create domain config: `config/domains/new-decree.yaml`
   - Abbreviations
   - Synonyms
   - Topic hints

2. Extract KG: Run offline pipeline on new documents

3. Generate summaries: Use LLM to create chapter/article summaries

4. Fine-tune weights: Ablation study on domain-specific validation set

### Optimizing Weights

Current DualLevel weights (0.30 semantic, 0.15 others) are defaults. For new domains:

```python
from scipy.optimize import minimize

def objective(weights):
    """Minimize 1 - Hit@10 on validation set."""
    config = DualLevelConfig(
        keyphrase_weight=weights[0],
        semantic_weight=weights[1],
        # ... other weights
    )
    retriever = DualLevelRetriever(kg, db, embeddings, config)
    hit_at_10 = evaluate_on_validation_set(retriever)
    return 1 - hit_at_10

# Optimize
result = minimize(objective, x0=default_weights)
optimized_weights = result.x
```

## Security & Privacy Considerations

### Data Protection

1. **User Queries**: Not logged or stored
2. **API Keys**: Via .env only, never in code
3. **Caching**: LLM cache (llm_cache.db) sanitized of PII
4. **Database**: SQLite (local file); no remote access

### Input Validation

```python
# Validate query length
MAX_QUERY_LENGTH = 1000
if len(query) > MAX_QUERY_LENGTH:
    raise ValueError(f"Query exceeds max length {MAX_QUERY_LENGTH}")

# Validate article IDs (SQL injection prevention)
ARTICLE_ID_PATTERN = r"^[a-zA-Z0-9\-:]+$"
if not re.match(ARTICLE_ID_PATTERN, article_id):
    raise ValueError(f"Invalid article ID: {article_id}")
```

---

**Last Updated**: 2026-02-14
**Version**: 1.0
