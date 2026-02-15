# System Architecture & 3-Tier Retrieval Design

**Project**: Vietnamese Legal RAG (vn_legal_rag)
**Last Updated**: 2026-02-14
**Architecture Version**: 1.0

## Architecture Overview

Vietnamese Legal RAG implements a **3-tier hierarchical retrieval system** for Vietnamese legal document question answering. Each tier employs different retrieval strategies that complement each other to achieve superior accuracy.

```
┌──────────────────────────────────────────────────────────┐
│                  User Query (Vietnamese)                  │
│            "Hành vi kinh doanh không đăng ký              │
│             vi phạm điều nào?"                            │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  TIER 0: Query Analyzer       │
         │  • Intent Detection           │
         │  • Keyword Extraction         │
         │  • Query Expansion (abbrev.)  │
         │  • Domain Detection           │
         └────────┬──────────────────────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
    ┌────────┐┌────────┐┌────────┐
    │ TIER 1 ││ TIER 2 ││ TIER 3 │
    │ Tree   ││Dual-   ││Semantic│
    │Travers-││Level   ││Bridge  │
    │al      ││Retriev.││(RRF)   │
    │        ││        ││        │
    │LLM-    ││6-comp. ││Fusion +│
    │guided  ││scoring ││KG Exp. │
    │nav.    ││        ││        │
    └───┬────┘└───┬────┘└───┬────┘
        │ Top-50  │ Top-50  │ Top-10
        └────┬────┴────┬────┘
             ▼         ▼
        ┌──────────────────────────┐
        │  Semantic Bridge Merger  │
        │  • RRF Fusion            │
        │  • Agreement Boosting    │
        │  • KG Expansion (2-hop)  │
        │  • Diversity Ranking     │
        └───────────┬──────────────┘
                    ▼
        ┌──────────────────────────┐
        │ Unified Top-10 Articles  │
        └───────────┬──────────────┘
                    ▼
        ┌──────────────────────────┐
        │  LLM Answer Generation   │
        │  + Citation Formatting   │
        └───────────┬──────────────┘
                    ▼
        ┌──────────────────────────┐
        │ GraphRAGResponse         │
        │ • Answer text            │
        │ • Citations (Điều X...)  │
        │ • Confidence score       │
        │ • Reasoning path         │
        │ • Query type             │
        └──────────────────────────┘
```

## Tier 0: Query Analyzer

**Purpose**: Understand user intent and prepare query for retrieval

**Location**: `vn_legal_rag/online/vietnamese-legal-query-analyzer.py`

### Components

```python
@dataclass
class AnalyzedQuery:
    original_query: str              # User's original question
    query_type: LegalQueryType       # Detected intent category
    keywords: List[str]              # Extracted legal keywords
    entities: List[str]              # Named entities (articles, organizations)
    legal_concepts: List[str]        # Concepts (rights, obligations, penalties)
    expanded_keywords: List[str]     # After abbreviation expansion
    domain_hints: List[str]          # Related topics
    confidence: float                # Classification confidence (0.0-1.0)
```

### Query Type Classification

| Type | Characteristics | Example | Optimal Tier | Hit@10 |
|------|-----------------|---------|--------------|--------|
| `article_lookup` | Direct article reference | "Điều 5 quy định gì?" | Tier 1 | 82.05% |
| `guidance_document` | Procedural "how to" | "Làm thế nào để thành lập doanh nghiệp?" | Tier 1+2 | 73.47% |
| `situation_analysis` | Conditional "what if" | "Nếu vi phạm Điều 5 thì bị phạt bao nhiêu?" | Tier 2+3 | 71.33% |
| `compare_regulations` | Multi-article comparison | "Sự khác biệt giữa Điều 5 và Điều 10?" | Tier 3 | 68.75% |
| `case_law_lookup` | Penalties and consequences | "Vi phạm hành vi gì bị phạt 50 triệu?" | Tier 2 | 61.11% |
| `general` | Other queries | Various | All tiers | ~60% |

### Implementation Details

1. **Intent Detection**: LLM-based classification (5-shot prompting)
2. **Keyword Extraction**: Legal keyword recognition + TF-IDF
3. **Entity Recognition**: Simple pattern matching for legal references (Điều X, Khoản Y)
4. **Query Expansion**: Uses VietnameseAbbreviationExpander
   - TGĐ → Tổng Giám đốc
   - TNHH → Trách nhiệm hữu hạn
   - HĐQT → Hội đồng quản trị
5. **Domain Detection**: Matches query keywords to domain YAML configs

## Tier 1: Tree Traversal Retriever

**Purpose**: Navigate document hierarchy using LLM and summaries

**Location**: `vn_legal_rag/online/tree-traversal-retriever.py`

**Performance**: 56.20% Hit@10 (single tier) | 82.05% on article_lookup queries

### Architecture

```
Query
  │
  ├─→ Loop 0 (Optional): Select documents
  │     • Match query keywords to document names
  │     • Use document summaries if available
  │     • Candidate selection: top-k documents
  │
  ├─→ Loop 1: Select chapters
  │     • LLM reads chapter_summaries.json
  │     • Prompt: "Which chapters are relevant?"
  │     • Multi-path: Consider top-5 chapters
  │     • Scoring: Semantic similarity + keyword match
  │
  └─→ Loop 2: Select articles
        • LLM reads article_summaries.json
        • Prompt: "Which articles in this chapter?"
        • Multi-path: Consider top-10 articles per chapter
        • Scoring: Dual scoring (keyword + semantic)
        • Output: Top-50 article candidates
```

### Data Requirements

```python
# chapter_summaries.json format
{
    "59-2020-QH14:c1": "Chapter 1: Các quy định chung",
    "59-2020-QH14:c2": "Chapter 2: Thành lập doanh nghiệp",
    ...
}

# article_summaries.json format
{
    "59-2020-QH14:d1": "Tên doanh nghiệp phải bao gồm từ chỉ lĩnh vực hoạt động",
    "59-2020-QH14:d2": "Doanh nghiệp cá nhân là đơn vị kinh tế do một cá nhân thành lập",
    ...
}
```

### Key Classes

```python
@dataclass
class TreeSearchResult:
    """Result from tree traversal."""
    chapters: List[TreeNode]           # Selected chapters with summaries
    articles: List[TreeNode]           # Candidate articles (top-50)
    reasoning: List[str]               # Reasoning for each loop
    scores: Dict[str, float]           # Confidence scores
    execution_time: float              # Wall-clock time

class TreeTraversalRetriever:
    def __init__(
        self,
        forest: UnifiedForest,
        llm_provider: BaseLLMProvider,
        article_summaries: Dict[str, str],
        chapter_summaries: Optional[Dict[str, str]] = None,
    ):
        ...

    def retrieve(
        self,
        query: str,
        top_k_chapters: int = 5,
        top_k_articles: int = 10,
        multi_path: bool = True,
    ) -> TreeSearchResult:
        """Execute tree traversal retrieval."""
        ...
```

### Strengths & Weaknesses

**Strengths**:
- Excellent for direct article lookups (82.05% Hit@10)
- Transparent reasoning (user sees chapter selection)
- Efficient (doesn't score all documents)
- Works well with domain summaries

**Weaknesses**:
- LLM-dependent (errors compound across loops)
- Limited to document hierarchy (misses cross-chapter relations)
- Requires pre-computed summaries
- Struggles with ambiguous queries

## Tier 2: DualLevel Semantic Retriever

**Purpose**: Global semantic search with multi-component scoring

**Location**: `vn_legal_rag/online/dual-level-retriever.py`

**Performance**: 61.74% Hit@10 (single tier) | Strong on all query types

### Architecture

```
Query
  │
  ├─→ Component 1: Keyphrase Matching
  │     • Extract keyphrases from query
  │     • TF-IDF scoring on article text
  │     • Weight: 0.15
  │
  ├─→ Component 2: Semantic Embeddings
  │     • Embed query → dense vector
  │     • Embed all articles → dense vectors
  │     • Cosine similarity scoring
  │     • Weight: 0.30 (highest)
  │
  ├─→ Component 3: Personalized PageRank
  │     • Seed nodes: query keywords → matching entities
  │     • Propagate importance through KG
  │     • Rank articles by connected entity importance
  │     • Weight: 0.15
  │
  ├─→ Component 4: Legal Concept Overlap
  │     • Extract entities from query
  │     • Extract entities from articles
  │     • Jaccard similarity on entity sets
  │     • Weight: 0.15
  │
  ├─→ Component 5: Theme Matching
  │     • Document-level theme assignment
  │     • Theme similarity between query and article
  │     • Weight: 0.10 (lowest)
  │
  └─→ Component 6: Hierarchy Scoring
        • Same-chapter bonus for articles
        • Parent-child proximity bonus
        • Weight: 0.15
```

### Configuration

```python
@dataclass
class DualLevelConfig:
    """Configuration for DualLevel scoring."""
    keyphrase_weight: float = 0.15
    semantic_weight: float = 0.30
    ppr_weight: float = 0.15
    concept_weight: float = 0.15
    theme_weight: float = 0.10
    hierarchy_weight: float = 0.15
    top_k: int = 50               # Return top-50 articles
    use_cache: bool = True        # Cache embeddings
```

### Composite Scoring Formula

```
final_score = (
    keyphrase_weight * norm(keyphrase_score) +
    semantic_weight * norm(semantic_score) +
    ppr_weight * norm(ppr_score) +
    concept_weight * norm(concept_score) +
    theme_weight * norm(theme_score) +
    hierarchy_weight * norm(hierarchy_score)
)
```

### Key Classes

```python
class DualLevelRetriever:
    """6-component semantic retriever."""

    def __init__(
        self,
        kg: Dict[str, Any],                    # Knowledge graph
        db: LegalDocumentDB,                   # Document database
        embedding_provider: TextEmbeddingsProvider,
        config: DualLevelConfig = DualLevelConfig(),
    ):
        ...

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve articles with composite scoring.

        Returns:
            List of (article_id, score) tuples, sorted by score descending
        """
        ...

    def _score_keyphrase(self, query: str, article_id: str) -> float:
        ...

    def _score_semantic(self, query: str, article_id: str) -> float:
        ...

    def _score_ppr(self, query: str) -> Dict[str, float]:
        ...

    # Other scoring methods...
```

### Strengths & Weaknesses

**Strengths**:
- Comprehensive coverage (all articles considered)
- Robust to single-signal failures (6 signals combined)
- No pre-computed summaries needed
- Semantic understanding via embeddings
- Graph reasoning via PPR

**Weaknesses**:
- Computational cost (embed all articles)
- Hyperparameter tuning (6 weights)
- Relies on embedding quality
- Slow for first query (caching helps)

## Tier 3: Semantic Bridge (RRF Fusion)

**Purpose**: Merge and enhance Tier 1 & 2 results via fusion and KG expansion

**Location**: `vn_legal_rag/online/semantic-bridge-rrf-merger.py`

**Performance**: +5.3% improvement over best single tier

### Architecture

```
Tier 1 Results (Top-50)        Tier 2 Results (Top-50)
   └─────┬─────┘                    └─────┬─────┘
         │                                │
         └────────────┬──────────────────┘
                      ▼
          ┌──────────────────────────┐
          │ Reciprocal Rank Fusion   │
          │ RRF_score = 1/(k+r1) +   │
          │            1/(k+r2)      │
          │ (k=60, default)          │
          └────────┬─────────────────┘
                   ▼
        ┌─────────────────────────────┐
        │ Agreement-Based Boosting    │
        │ +0.3 score if in both tiers │
        └────────┬────────────────────┘
                 ▼
      ┌──────────────────────────────┐
      │ Knowledge Graph Expansion    │
      │ For each article:            │
      │ • Find related entities (KG) │
      │ • Follow 2-hop relations     │
      │ • Add related articles       │
      └────────┬─────────────────────┘
               ▼
    ┌───────────────────────────────┐
    │ Diversity Ranking             │
    │ Ensure coverage across        │
    │ different chapters/themes     │
    └────────┬──────────────────────┘
             ▼
    ┌───────────────────────────────┐
    │ Final Top-10 Articles         │
    │ Ready for LLM answer gen.     │
    └───────────────────────────────┘
```

### Reciprocal Rank Fusion (RRF)

Standard RRF combines multiple ranked lists:

```
RRF(d) = Σ(1 / (k + rank(d)))

where:
  d = document (article)
  rank(d) = position in ranked list (1-indexed)
  k = constant (default: 60) to handle missing items
```

**Intuition**: Articles appearing in both tiers get boosted scores. Missing from one tier gets penalty but not zero.

### Agreement Scoring

Articles found in both Tier 1 AND Tier 2:
- Receive additional boost (0.3 points)
- Indicates agreement between different strategies
- Higher confidence in relevance

### Knowledge Graph Expansion

For top-k articles from RRF fusion:

1. **Extract entities** from article text
2. **Find entity nodes** in KG
3. **Traverse edges** (up to 2 hops):
   - `REFERENCES` → related articles
   - `AMENDS` → modified/modifying articles
   - `IMPLEMENTS` → implementation articles
   - `HAS_PENALTY` → penalty articles
   - `RELATED_TO` → conceptually related

4. **Add discovered articles** to candidate list
5. **Rank by proximity** (1-hop higher than 2-hop)

### Diversity Ranking

Ensure final top-10 covers different aspects:

```python
# Pseudo-code for diversity ranking
selected = []
for article in rrf_ranked_list:
    # Check if article covers new theme/chapter
    if article_theme not in selected_themes:
        selected.append(article)
        selected_themes.add(article_theme)

    if len(selected) == 10:
        break

# Fill remaining slots if needed (less diverse articles)
```

### Key Classes

```python
class SemanticBridge:
    """RRF fusion and KG expansion for tier merging."""

    def __init__(
        self,
        kg: Dict[str, Any],
        db: LegalDocumentDB,
        rrf_k: int = 60,
        kg_expansion_hops: int = 2,
        agreement_boost: float = 0.3,
        max_results: int = 10,
    ):
        ...

    def merge_and_expand(
        self,
        tier1_results: List[Tuple[str, float]],  # (article_id, score)
        tier2_results: List[Tuple[str, float]],
        query: str,
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Merge tiers via RRF, apply agreement boost, expand via KG.

        Returns:
            List of (article_id, final_score, reasoning_path)
        """
        ...

    def _apply_rrf(
        self,
        tier1_results: List[Tuple[str, float]],
        tier2_results: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion."""
        ...

    def _expand_via_kg(
        self,
        article_ids: List[str],
        max_hops: int = 2,
    ) -> Dict[str, List[str]]:
        """KG-based expansion to related articles."""
        ...
```

### Strengths & Weaknesses

**Strengths**:
- Robustness: Failures in one tier don't eliminate articles
- Capture cross-chapter dependencies (via KG)
- Agreement signal increases confidence
- Diversity ensures comprehensive coverage
- Incremental improvement without much overhead

**Weaknesses**:
- Dependent on both Tier 1 & 2 being run first
- KG quality affects expansion results
- May miss articles neither tier found
- Final ranking still somewhat arbitrary

## Data Flow & Integration Points

### Main Query Engine Integration

**File**: `vn_legal_rag/online/legal-graphrag-3tier-query-engine.py`

```python
class LegalGraphRAG:
    """Main orchestrator for 3-tier retrieval."""

    def query(
        self,
        query_text: str,
        adaptive_retrieval: bool = True,
        top_k: int = 10,
    ) -> GraphRAGResponse:
        """
        Execute full 3-tier pipeline.

        1. Analyze query (Tier 0)
        2. Execute Tier 1 (tree traversal)
        3. Execute Tier 2 (dual-level)
        4. Execute Tier 3 (RRF fusion)
        5. Generate LLM answer
        6. Format citations
        7. Return structured response
        """
        # Tier 0: Analyze query
        analyzed_query = self.query_analyzer.analyze(query_text)

        # Tier 1: Tree traversal
        if not self.ablation_config.disable_tier1:
            tier1_result = self.tier1_retriever.retrieve(query_text)
            tier1_articles = [(a.node_id, score) for a, score in tier1_result.articles]
        else:
            tier1_articles = []

        # Tier 2: DualLevel
        if not self.ablation_config.disable_tier2:
            tier2_articles = self.tier2_retriever.retrieve(query_text, top_k=50)
        else:
            tier2_articles = []

        # Tier 3: RRF Fusion
        if not self.ablation_config.disable_tier3 and tier1_articles and tier2_articles:
            final_articles = self.tier3_merger.merge_and_expand(
                tier1_articles, tier2_articles, query_text
            )
        else:
            # Fallback: use best single tier
            final_articles = tier1_articles or tier2_articles

        # Fetch full article text
        context_articles = self._fetch_articles(final_articles)

        # Generate answer via LLM
        answer = self.llm_provider.generate(
            prompt=self._build_prompt(query_text, context_articles),
        )

        # Format response
        return GraphRAGResponse(
            response=answer,
            citations=self._format_citations(context_articles),
            reasoning_path=[...],  # Track which tier found each article
            confidence=self._calculate_confidence(final_articles),
            query_type=analyzed_query.query_type,
        )
```

## Data Models & Structures

### TreeNode Hierarchy

```python
@dataclass
class TreeNode:
    """Hierarchical node for document structure."""
    node_id: str                    # Unique ID: "doc:c1:d5:k1:a"
    node_type: NodeType             # DOCUMENT | CHAPTER | SECTION | ARTICLE | CLAUSE | POINT
    name: str                       # Display name
    description: str                # Summary
    content: str                    # Full text
    metadata: Dict[str, Any]        # Custom metadata
    sub_nodes: List[TreeNode]       # Children
```

**Hierarchy Example**:
```
Document (59-2020-QH14): Luật Doanh nghiệp 2020
├── Chapter (c1): Các quy định chung
│   ├── Article (d1): Tên doanh nghiệp
│   │   ├── Clause (k1): Định nghĩa tên
│   │   │   ├── Point (a): Bao gồm chỉ lĩnh vực
│   │   │   └── Point (b): Dùng tiếng Việt
│   │   └── Clause (k2): Hạn chế...
```

### Knowledge Graph Schema

```python
kg = {
    "nodes": [
        {
            "id": "entity_001",
            "type": "ORGANIZATION",        # LegalEntityType
            "content": "Công ty Cổ phần",
            "metadata": {"source_article": "d1"}
        },
        ...  # 1299 entities
    ],
    "edges": [
        {
            "source_id": "entity_001",
            "target_id": "entity_002",
            "relation_type": "REQUIRES",   # LegalRelationType
            "confidence": 0.95,
            "evidence": "Từ Điều 5"
        },
        ...  # 2577 relations
    ]
}
```

### Response Structure

```python
@dataclass
class GraphRAGResponse:
    response: str                           # LLM-generated answer
    citations: List[Dict[str, Any]]         # [{"citation_string": "Điều X...", "source": "id", ...}]
    reasoning_path: List[str]               # ["tier1:d5", "tier2:d10", "tier3:d8"]
    confidence: float                       # 0.0-1.0
    query_type: LegalQueryType              # Classification
    metadata: Dict[str, Any]                # Additional info
    tree_search_result: Optional            # Tier 1 detail
    intent: Optional[QueryIntent]           # Query analysis detail
```

## Additional Topics

See [Operations & Extensibility Guide](./system-architecture-operations-guide.md) for:
- Performance characteristics (latency breakdown, accuracy by query type)
- Failure modes & resilience (tier recovery strategies)
- Extensibility & future work (adding new tiers, supporting new domains)
- Security & privacy considerations

---

**Last Updated**: 2026-02-14
**Version**: 1.0
