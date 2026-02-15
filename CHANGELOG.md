# Changelog

All notable changes to the Vietnamese Legal RAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-14

### Added
- Initial release of Vietnamese Legal RAG system
- 3-Tier retrieval architecture combining Tree Traversal, DualLevel, and Semantic Bridge
- Tier 1: Tree Traversal Retriever with LLM-guided navigation
- Tier 2: DualLevel Retriever with 6-component semantic scoring
- Tier 3: Semantic Bridge with RRF-based fusion and KG expansion
- Vietnamese Legal Query Analyzer with 6 query type detection
- Personalized PageRank (PPR) for knowledge graph scoring
- Offline knowledge graph extraction pipeline
  - Unified entity/relation extractor (LightRAG-style single LLM call)
  - Incremental KG builder with built-in entity merging
  - Entity deduplication using Vietnamese slug normalization
- Database schema for Vietnamese legal documents
  - Support for documents, chapters, sections, articles, clauses, points
  - Cross-reference tracking
  - Abbreviation extraction
- Utility modules
  - LLM provider with caching (OpenAI, Anthropic, Google Gemini)
  - Text embeddings with sentence-transformers
  - Vietnamese abbreviation expansion
  - Legal citation formatting
- Type system with 11 legal entity types and 28+ relation types
- Tree models for document hierarchy navigation
- Comprehensive test suite
- Package distribution setup (pyproject.toml, MANIFEST.in)
- Documentation (README.md, architecture overview)

### Performance
- Hit@10: 76.53% on 379 Q&A pairs
- Recall@5: 58.12%
- Mean Reciprocal Rank (MRR): 0.5422
- 85% improvement over TF-IDF baseline
- 117% improvement over pure PageIndex approach

### Features
- Adaptive retrieval based on query type
- Multi-language LLM support (OpenAI GPT-4, Anthropic Claude, Google Gemini)
- Vietnamese domain-specific abbreviation expansion
- Knowledge graph integration with 1299 entities and 2577 relations
- Cross-document reference resolution
- Legal citation formatting and parsing
- LLM response caching for cost reduction
- Resume capability for fault-tolerant extraction pipeline

### Dependencies
- Python 3.10+
- OpenAI SDK (GPT-4 support)
- Anthropic SDK (Claude support)
- Google Generative AI (Gemini support)
- SQLAlchemy 2.0+ (database ORM)
- Jellyfish (Vietnamese text similarity)
- NumPy, SciPy (scientific computing)
- Optional: sentence-transformers, FAISS (embeddings and vector search)

### Documentation
- README with quick start guide
- Architecture overview diagram
- API reference documentation
- Data structure documentation
- Configuration guide with .env.example
- Type hints throughout codebase (PEP 561 compliant)

### Known Limitations
- Requires pre-generated knowledge graph and summaries
- Vietnamese-only support (no multilingual)
- Tested primarily on Luật Doanh nghiệp 2020 corpus
- LLM API costs for real-time query processing
- No built-in scraping tools (manual data preparation required)

### Future Roadmap
- [ ] Multi-document corpus support beyond Luật Doanh nghiệp
- [ ] Real-time knowledge graph updates
- [ ] Cost optimization with smaller LLMs
- [ ] Multilingual support (English legal documents)
- [ ] Web UI for interactive querying
- [ ] Batch evaluation tools
- [ ] Performance profiling and optimization
- [ ] Docker containerization
- [ ] API server deployment (FastAPI)

---

## Release Notes Format

### [Version] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes to existing functionality

#### Deprecated
- Soon-to-be removed features

#### Removed
- Removed features

#### Fixed
- Bug fixes

#### Security
- Security improvements
