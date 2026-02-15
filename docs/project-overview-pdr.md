# Project Overview & Product Development Requirements (PDR)

**Project**: Vietnamese Legal RAG (vn_legal_rag)
**Version**: 1.0.0 (Research Prototype)
**Status**: Active Development
**Date**: 2026-02-14

## Executive Summary

Vietnamese Legal RAG is a state-of-the-art Retrieval-Augmented Generation (RAG) system specifically designed for Vietnamese legal document question answering. The system implements a novel 3-tier retrieval architecture that combines tree-based hierarchical navigation, semantic search, and knowledge graph reasoning to achieve exceptional retrieval accuracy on Vietnamese corporate law documents.

**Mission**: Enable Vietnamese legal professionals and citizens to accurately retrieve relevant legal articles in response to natural language legal queries, with transparent source citations and reasoning paths.

## Business Objectives

1. **Improve Legal Information Access**: Reduce time for legal professionals to find relevant legislation (10+ hours → <5 minutes)
2. **Democratize Legal Knowledge**: Enable non-lawyers to understand complex legal requirements via conversational Q&A
3. **Support Legal Tech Ecosystem**: Provide foundation for downstream LLM-based legal analysis services
4. **Research Excellence**: Demonstrate state-of-the-art retrieval performance (76.53% Hit@10 vs. 41.41% baseline)

## Functional Requirements

### FR1: 3-Tier Retrieval Architecture

The system MUST support three independent retrieval tiers that work in concert:

**FR1.1 - Tier 1: Tree Traversal Retrieval**
- LLM-guided navigation through document hierarchy (Document → Chapter → Article)
- Multi-path traversal considering top-k candidate chapters and articles
- Integration with chapter_summaries.json and article_summaries.json
- Support for domain-specific query expansion via abbreviation expansion
- Expected performance: 56.20% Hit@10 (single tier)
- Query types best served: article_lookup (82.05% Hit@10)

**FR1.2 - Tier 2: DualLevel Semantic Retrieval**
- 6-component scoring system:
  1. Keyphrase matching (TF-IDF on legal keywords)
  2. Semantic embeddings (dense vector similarity)
  3. Personalized PageRank on knowledge graph
  4. Concept overlap (entity and relation matching)
  5. Theme matching (document-level topics)
  6. Hierarchy scoring (structural proximity)
- Configurable component weights via DualLevelConfig
- Expected performance: 61.74% Hit@10 (single tier)
- Global coverage across all document sections

**FR1.3 - Tier 3: Semantic Bridge Fusion**
- Reciprocal Rank Fusion (RRF) combining Tier 1 and Tier 2 rankings
- Agreement-based boosting for articles appearing in multiple tiers
- Knowledge graph expansion (multi-hop traversal via cross-references)
- Final ranking diversity to ensure coverage across document sections
- Expected performance: +5.3% improvement when combined with Tier 1+2

### FR2: Adaptive Query Processing

**FR2.1 - Query Type Detection**
- Automatically classify queries into 5 categories:
  - `article_lookup`: Direct article reference (82.05% Hit@10)
  - `guidance_document`: Procedural "how to" questions (73.47% Hit@10)
  - `situation_analysis`: Conditional "what if" scenarios (71.33% Hit@10)
  - `compare_regulations`: Multi-article comparisons (68.75% Hit@10)
  - `case_law_lookup`: Penalty and consequence queries (61.11% Hit@10)
- Route queries to optimal tier based on detected type
- Provide confidence scores for detected intent

**FR2.2 - Intelligent Query Expansion**
- Vietnamese abbreviation expansion (TGĐ → Tổng Giám đốc)
- Legal synonym matching (hợp đồng ↔ khế ước)
- Domain-specific keyword injection from ontology
- Multi-language support (support both formal and colloquial Vietnamese)

### FR3: Knowledge Graph Integration

**FR3.1 - Entity Extraction**
- Support 11 legal entity types (ORGANIZATION, PERSON_ROLE, LEGAL_TERM, LEGAL_REFERENCE, MONETARY, PERCENTAGE, DURATION, LOCATION, CONDITION, ACTION, PENALTY)
- Offline extraction via LLM-based UnifiedEntityRelationExtractor
- Minimum 1,299 entities for legal corpus

**FR3.2 - Relation Extraction**
- Support 28+ legal relation types (REQUIRES, HAS_PENALTY, APPLIES_TO, REFERENCES, AMENDS, SUPERSEDES, etc.)
- Bidirectional relation storage (forward and inverse)
- Minimum 2,577 relations for legal corpus

**FR3.3 - Multi-Hop Traversal**
- Support up to 2-hop expansion from query results
- Traverse cross-reference relations (REFERENCES, AMENDS, IMPLEMENTS)
- Traverse authority relations (AUTHORIZED_BY, RESPONSIBLE_FOR)
- Maintain provenance through traversal path

### FR4: Vietnamese Legal Citation Formatting

**FR4.1 - Citation Generation**
- Format citations according to Vietnamese legal conventions
- Structure: `Điều [NUMBER] ([Description]), [Decree/Law Name]`
- Example: `Điều 5 (Hành vi kinh doanh), Luật Doanh nghiệp 2020`
- Support citations with clause/point references: `Điều 5, Khoản 1, Điểm a`

**FR4.2 - Source Provenance**
- Track article source with unique ID: `document:article:clause:point`
- Maintain confidence scores through retrieval pipeline
- Document reasoning path (which tier found the article)
- Link original source to database for text retrieval

### FR5: Multi-Tier Response Generation

**FR5.1 - Unified Context Assembly**
- Combine top-k articles from all three tiers
- Deduplicate overlapping results
- Maintain ranking scores for each tier
- Prepare context for LLM answer generation

**FR5.2 - Response Format**
- Natural language answer generated by LLM
- List of 3-10 citations with source links
- Confidence score for answer (0.0-1.0)
- Reasoning path showing which tiers contributed
- Query intent classification for transparency

### FR6: Database & Storage

**FR6.1 - Document Storage**
- SQLite database (legal_docs.db) storing full document hierarchy
- Support articles, clauses, points with full text
- Maintain document structure (Document → Chapter → Article → Clause → Point)
- Support cross-reference tracking

**FR6.2 - Knowledge Graph Storage**
- JSON format (legal_kg.json) with nodes and edges
- Nodes: entity_id, entity_type, content, metadata
- Edges: source_id, target_id, relation_type, confidence
- Support incremental updates via IncrementalKGBuilder

**FR6.3 - Summary Caching**
- chapter_summaries.json for Tier 1 Loop 1 navigation
- article_summaries.json for Tier 1 Loop 2 navigation
- Enable fast LLM decision-making without re-generating summaries

**FR6.4 - LLM Response Caching**
- SQLite cache (llm_cache.db) for LLM API responses
- Reduce latency and API costs for repeat queries
- Support cache invalidation and cleanup

## Non-Functional Requirements

### NFR1: Performance

| Metric | Target | Baseline | Status |
|--------|--------|----------|--------|
| Hit@5 | 65%+ | 22.43% | ✓ 67.28% |
| Hit@10 | 75%+ | 41.41% | ✓ 76.53% |
| Recall@5 | 55%+ | 22.43% | ✓ 58.12% |
| MRR (Mean Reciprocal Rank) | 0.50+ | 0.2891 | ✓ 0.5422 |
| Query latency (p95) | <10s | N/A | TBD |
| Cache hit rate | >40% | N/A | TBD |

### NFR2: Scalability

- Support corpus of 2,000+ articles (current: 239)
- Support 10,000+ entities in knowledge graph
- Support 50,000+ relations in knowledge graph
- Enable batch evaluation on 500+ Q&A pairs
- Support concurrent queries (≥10 parallel requests)

### NFR3: Reliability

- Graceful degradation if individual tiers fail
- Fallback to remaining tiers without error
- Checkpoint-based resume for long-running KG extraction
- Database consistency checks on startup

### NFR4: Maintainability

- Type-safe Python codebase with MyPy strict mode
- Comprehensive docstrings on all public functions
- Unit test coverage ≥70%
- Support domain-specific customization via YAML configs
- Modular architecture enabling tier replacement

### NFR5: Security & Privacy

- No storage of user queries (privacy)
- No API keys in source code (use .env)
- SQL injection protection via SQLAlchemy ORM
- LLM cache sanitization (no sensitive data)

### NFR6: Language Support

- Primary: Vietnamese (all interfaces)
- Secondary: English (documentation, code comments)
- Support both formal and colloquial Vietnamese
- Proper Unicode handling (Vietnamese diacritics)

## Technical Requirements

### TR1: Architecture Constraints

- **Modular Design**: Each tier must be independently functional
  - Tier 1 can operate without Tier 2/3
  - Tier 2 can operate without Tier 1
  - Tier 3 requires Tier 1 + Tier 2 for fusion

- **Data Independence**: Each tier operates on same underlying data (KG, summaries)
- **Pluggable LLM Providers**: Support OpenAI, Anthropic, and local models
- **Configuration-Driven**: Domain configs enable per-decree customization

### TR2: Data Format Standards

- **Knowledge Graph**: JSON with nodes/edges arrays
- **Summaries**: JSON dictionaries (article_id → summary text)
- **Document Tree**: JSON serialization of TreeNode hierarchy
- **Domain Configs**: YAML with abbreviations, synonyms, examples

### TR3: External Dependencies

- **LLM APIs**: OpenAI (gpt-4o-mini), Anthropic (claude-3.5-haiku)
- **Embeddings**: OpenAI embeddings or BAAI BGE models
- **Database**: SQLite (embedded)
- **Graph Library**: NetworkX for KG traversal

### TR4: Development Standards

- **Language**: Python 3.10+
- **Code Style**: Black formatter, Ruff linter, MyPy type checker
- **Testing**: Pytest framework, ≥70% coverage
- **Documentation**: Markdown in docs/, docstrings in code
- **Version Control**: Git with conventional commits

## Acceptance Criteria

### AC1: 3-Tier System Works End-to-End
- [ ] Tier 1 tree traversal retrieves 50+ candidates
- [ ] Tier 2 dual-level scoring ranks 50 candidates
- [ ] Tier 3 RRF fusion produces final top-10
- [ ] Combined system achieves ≥76% Hit@10
- [ ] All tiers can be disabled independently (ablation)

### AC2: Query Processing Accuracy
- [ ] Query type detection accuracy ≥85% on test set
- [ ] Query expansion expands ≥95% of known abbreviations
- [ ] Answer confidence scores correlate with actual correctness (Spearman ≥0.7)

### AC3: Citation Accuracy
- [ ] 100% of citations are valid article IDs in database
- [ ] Citation formatting matches Vietnamese legal conventions
- [ ] Provenance tracking shows which tier found each citation
- [ ] All citations are traceable back to database

### AC4: Knowledge Graph Quality
- [ ] KG contains ≥1,299 entities
- [ ] KG contains ≥2,577 relations
- [ ] All entity types match LegalEntityType enum
- [ ] All relation types match LegalRelationType enum
- [ ] Relation extraction accuracy ≥80% (manual validation on sample)

### AC5: Performance Targets
- [ ] Query latency p95 <10 seconds
- [ ] LLM cache hit rate >40% on repeated queries
- [ ] Memory usage <2GB for full system load
- [ ] Database query latency <100ms per article

### AC6: Code Quality & Maintainability
- [ ] MyPy type checking passes with ≥95% coverage
- [ ] All public functions have docstrings
- [ ] Unit test coverage ≥70%
- [ ] Code follows Black formatting + Ruff linting
- [ ] No security warnings from dependency scan

### AC7: Documentation Completeness
- [ ] All modules documented in codebase-summary.md
- [ ] All public APIs documented with examples
- [ ] 3-tier architecture explained with diagrams
- [ ] Data formats specified (KG, summaries, configs)
- [ ] Integration examples for new developers

## Project Scope

### In Scope

- Vietnamese legal documents (Luật Doanh nghiệp 2020, Nghị định, Thông tư)
- Corporate law domain (company registration, management, dissolution)
- Tree-based document navigation
- Semantic search with 6-component scoring
- Knowledge graph extraction and reasoning
- LLM-based answer generation
- Vietnamese legal citation formatting

### Out of Scope

- Legal interpretation or advice (system provides retrieval, not advice)
- Support for non-Vietnamese legal systems
- Real-time document updates (offline extraction only)
- Multi-language translation
- Complex NLP tasks beyond entity/relation extraction (NER/RE)
- Generative law practice tools

## Roadmap

### Phase 1: Core Implementation (Completed)
- [x] Offline KG extraction pipeline
- [x] Online 3-tier retrieval system
- [x] Query type detection
- [x] Citation formatting
- [x] Basic evaluation on 379 Q&A pairs

### Phase 2: Production Hardening (Current)
- [ ] Performance optimization (query latency <5s)
- [ ] Scalability testing (10,000+ articles)
- [ ] Security audit and hardening
- [ ] Comprehensive test coverage (≥80%)
- [ ] API deployment (FastAPI or similar)

### Phase 3: Enhancement (Future)
- [ ] Support for additional legal domains (labor law, tax law)
- [ ] Multi-language support (English, French)
- [ ] Feedback loop for ranking optimization
- [ ] Interactive disambiguation for ambiguous queries
- [ ] Legal concept linking and ontology expansion

### Phase 4: Ecosystem Integration (Future)
- [ ] Integration with legal document management systems
- [ ] API for third-party applications
- [ ] Browser plugin for inline legal citations
- [ ] Mobile application for legal professionals
- [ ] Training pipeline for domain customization

## Success Metrics

### Retrieval Quality
- Hit@10 ≥75% (current: 76.53%)
- MRR ≥0.50 (current: 0.5422)
- Per-query-type analysis showing strength in each category

### System Reliability
- Uptime ≥99.5% in production
- Query success rate ≥99%
- Error recovery within <30 seconds

### User Adoption
- TBD: Legal professional feedback (post-launch)
- TBD: Query volume metrics (post-launch)
- TBD: User satisfaction score (post-launch)

### Engineering Quality
- Code coverage ≥80%
- Zero critical security vulnerabilities
- Documentation completeness ≥95%

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| LLM API rate limits | Medium | High | Implement caching, batch processing, fallback models |
| Knowledge graph drift | Low | Medium | Version KG, track changes, periodic validation |
| Query ambiguity | Medium | Low | Multi-hop expansion, reasoning path, confidence scores |
| Scalability bottleneck | Medium | High | Optimize embeddings, cache layer, index structures |
| Domain-specific coverage | Low | Medium | Modular config system, easy domain addition |

## Dependencies

### Internal
- Legal documents database (SQL schema in models.py)
- Knowledge graph extraction (UnifiedEntityRelationExtractor)
- Configuration files (YAML domain configs)

### External
- LLM APIs (OpenAI, Anthropic)
- Embedding models (OpenAI, BAAI BGE)
- NetworkX (graph algorithms)
- SQLAlchemy (ORM)

## Constraints

1. **Vietnamese Legal Focus**: System optimized for Vietnamese language; other languages require retraining
2. **Offline KG**: Knowledge graph extracted once; real-time updates require pipeline re-run
3. **LLM Dependency**: Relies on external LLM APIs; failures cascade (unless fallback configured)
4. **Evaluation Dataset**: 379 Q&A pairs; may not represent all query types equally
5. **Document Coverage**: Limited to downloaded decrees; incomplete legal system representation

## Success Criteria Summary

The project is **successful** when:

1. ✓ **Performance**: 3-tier system achieves ≥76% Hit@10 on evaluation set
2. ✓ **Integration**: All three tiers work together and can be independently disabled
3. ✓ **Quality**: Knowledge graph has ≥1,299 entities and ≥2,577 relations with ≥80% accuracy
4. ✓ **Usability**: End-to-end query execution completes in <10 seconds
5. ✓ **Maintainability**: Code passes MyPy, ≥70% test coverage, comprehensive documentation
6. ✓ **Deployment**: System can be deployed and reproduced on new datasets

---

**Last Updated**: 2026-02-14
**Version**: 1.0
**Author**: Documentation Team
