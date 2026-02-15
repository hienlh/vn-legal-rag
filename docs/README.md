# Documentation Index

Welcome to the Vietnamese Legal RAG documentation. This directory contains comprehensive technical documentation for developers, architects, and stakeholders.

## Quick Navigation

### For New Developers (Start Here)

1. **[Codebase Summary](./codebase-summary.md)** (15 min read)
   - Overview of all 36 modules and 8,446 LOC
   - Module structure and organization
   - Key data structures
   - Integration points

2. **[System Architecture](./system-architecture.md)** (30 min read)
   - Detailed 3-tier retrieval design
   - Tier 0: Query analysis
   - Tier 1: Tree traversal
   - Tier 2: Dual-level semantic scoring
   - Tier 3: RRF fusion
   - Data flow and performance characteristics

3. **[Code Standards](./code-standards.md)** (Reference)
   - Coding conventions (kebab-case vs snake_case)
   - Python best practices (type hints, dataclasses, docstrings)
   - Module-specific guidelines
   - Testing standards
   - Security guidelines

### For Project Managers & Stakeholders

1. **[Project Overview & PDR](./project-overview-pdr.md)** (20 min read)
   - Product vision and business objectives
   - 6 functional requirements (FRs)
   - 6 non-functional requirements (NFRs)
   - 7 acceptance criteria
   - Success metrics
   - Risk assessment

2. **[Project Roadmap](./project-roadmap.md)** (25 min read)
   - 4 development phases (18-month timeline)
   - Phase 1: Core system (completed, 76.53% Hit@10)
   - Phase 2: Production hardening (5 epics, $80K)
   - Phase 3: Feature enhancement (domain expansion, reasoning)
   - Phase 4: Ecosystem integration (SaaS, plugins)
   - Release schedule and milestones
   - Risk mitigation strategies

### For System Architects

- **[System Architecture](./system-architecture.md)**: Deep dive into 3-tier design, algorithms, and performance
- **[Code Standards](./code-standards.md)**: Architectural patterns and design decisions
- **[Project Overview & PDR](./project-overview-pdr.md)**: Technical constraints and requirements

### For Backend Engineers

- **[Codebase Summary](./codebase-summary.md)**: Module breakdown and APIs
- **[System Architecture](./system-architecture.md)**: Tier implementation details and data flow
- **[Code Standards](./code-standards.md)**: Implementation guidelines

---

## Documentation Overview

### File Descriptions

| File | Size | Purpose | Audience |
|------|------|---------|----------|
| **codebase-summary.md** | 502 LOC | Complete module breakdown and API overview | Developers, architects |
| **system-architecture.md** | 804 LOC | 3-tier design, algorithms, performance | Architects, backend engineers |
| **code-standards.md** | 820 LOC | Coding conventions and best practices | Developers, code reviewers |
| **project-overview-pdr.md** | 394 LOC | Requirements, success criteria, business goals | Product managers, stakeholders |
| **project-roadmap.md** | 647 LOC | Development timeline, phases, budget | Project leadership |
| **README.md** | This file | Documentation index and navigation | Everyone |

**Total**: 3,167 lines of documentation across 6 files

---

## Key Concepts

### 3-Tier Retrieval Architecture

```
Query → Tier 0 (Analyze) → Tier 1 (Tree) + Tier 2 (Semantic) → Tier 3 (Fusion) → Answer
```

1. **Tier 0**: Intent detection, query expansion, keyword extraction
2. **Tier 1**: LLM-guided tree traversal (56.20% Hit@10)
3. **Tier 2**: 6-component semantic scoring (61.74% Hit@10)
4. **Tier 3**: RRF fusion + KG expansion (+5.3% improvement)

### Performance Metrics

- **Hit@10**: 76.53% (target: 75%+) ✓
- **MRR**: 0.5422 (target: 0.50+) ✓
- **Hit@5**: 67.28% (target: 65%+) ✓
- **Query Latency (p95)**: ~7s (target: <5s) ⏳

### Module Organization

```
vn_legal_rag/
├── offline/          # KG extraction (4 modules)
├── online/           # 3-tier retrieval (9 modules)
├── types/            # Data models (4 modules)
├── utils/            # Utilities (11 modules)
├── config/           # Domain configs (YAML)
└── scripts/          # Evaluation tools (4 scripts)
```

---

## Development Phases

### Phase 1: Core System ✓ COMPLETED
- Hit@10: 76.53% achieved
- 3-tier architecture implemented
- Knowledge graph: 1,299 entities, 2,577 relations
- Status: Ready for production hardening

### Phase 2: Production Hardening 🟡 IN PROGRESS (Q1 2026)
- Performance optimization (<5s latency target)
- Comprehensive testing (80%+ coverage)
- Documentation completion (this package)
- Security hardening & deployment
- Budget: $80K, Duration: 6 weeks

### Phase 3: Feature Enhancement 📋 PLANNED (Q2 2026)
- Query disambiguation & multi-turn conversation
- Advanced multi-hop KG reasoning
- Feedback-driven ranking optimization
- Domain expansion: 6 legal domains
- Budget: $100K, Duration: 8 weeks

### Phase 4: Ecosystem Integration 📋 PLANNED (Q3-Q4 2026)
- SaaS platform for law firms
- Office plugin integrations
- Generative features (drafting, summarization)
- Enterprise features (auth, analytics, white-label)

---

## Getting Started

### Prerequisites

- Python 3.10+
- SQLite3
- 2GB disk space (data directory)
- API keys (OpenAI or Anthropic)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vn_legal_rag.git
cd vn_legal_rag

# Install package
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Download data (optional, required for offline pipeline)
# See codebase-summary.md "Data Directory" section
```

### First Query

```python
from vn_legal_rag import LegalGraphRAG, UnifiedForest
import json

# Load data
with open("data/kg_enhanced/legal_kg.json") as f:
    kg = json.load(f)

with open("data/document_forest.json") as f:
    forest = UnifiedForest.from_json(f.read())

# Initialize
graphrag = LegalGraphRAG(
    kg=kg,
    forest=forest,
    db_path="data/legal_docs.db",
    llm_provider="openai",
    llm_model="gpt-4o-mini"
)

# Query
result = graphrag.query("Phạt bao nhiêu nếu kinh doanh không đăng ký?")
print(f"Answer: {result.response}")
print(f"Type: {result.query_type.value}")
for citation in result.citations:
    print(f"  - {citation['citation_string']}")
```

See **Codebase Summary** for more examples and API documentation.

---

## Common Questions

### Q: Where do I start if I'm new to this project?

**A**: Follow this path:
1. Read **Codebase Summary** (15 min) for module overview
2. Read **System Architecture** (30 min) for design understanding
3. Reference **Code Standards** while implementing

### Q: What's the current status of the project?

**A**: Phase 1 complete (76.53% Hit@10 achieved), Phase 2 in progress. See **Project Roadmap** for detailed timeline.

### Q: How do I contribute?

**A**:
1. Follow **Code Standards** for conventions
2. Make sure new code has ≥80% test coverage
3. Add docstrings (Google style) to all public functions
4. Submit PR with description referencing acceptance criteria

### Q: How do I extend the system?

**A**: See **System Architecture** "Extensibility" section for guidelines on:
- Adding a new retrieval tier
- Supporting new legal domains
- Optimizing component weights

### Q: What are the performance targets?

**A**: See **Project Overview & PDR** "Non-Functional Requirements" section:
- Hit@10 ≥75% (achieved 76.53%)
- MRR ≥0.50 (achieved 0.5422)
- Query latency p95 <5s (currently ~7s, optimizing in Phase 2)

---

## Documentation Standards

### For Code Comments

- Explain *why*, not *what*
- Use clear, concise language
- Reference related sections when relevant

### For Docstrings

- Google style format
- Include Args, Returns, Raises, Examples
- Every public function/class requires docstring

### For Markdown Files

- Clear section hierarchy (H2 for major sections)
- Tables for structured information
- Code blocks with syntax highlighting
- Cross-references for related topics

---

## Keeping Documentation Updated

### When Making Code Changes

1. Update relevant docstrings
2. Update **Codebase Summary** if module structure changes
3. Update **Code Standards** if conventions change
4. Create/update example code in docs

### When Adding New Tiers/Features

1. Document in **System Architecture**
2. Add to module table in **Codebase Summary**
3. Update **Project Roadmap** with new phase/epic
4. Add examples to **Code Standards**

### Monthly Updates

- Review **Project Roadmap** progress
- Update Phase status and next milestones
- Document new patterns in **Code Standards**

---

## Related Resources

### External Links

- **GitHub Repository**: https://github.com/yourusername/vn_legal_rag
- **Issue Tracker**: https://github.com/yourusername/vn_legal_rag/issues
- **Pull Requests**: https://github.com/yourusername/vn_legal_rag/pulls

### Research Papers

- **LightRAG** (Gao et al., 2024): Graph-augmented RAG
- **PageIndex** (Shi et al., 2024): Hierarchical document navigation
- **Vietnamese Legal NLP**: Corpus and NER/RE resources

### Legal References

- **Luật Doanh nghiệp 2020** (Law on Enterprises)
- **thuvienphapluat.vn**: Vietnamese legal document repository

---

## Support & Questions

### Getting Help

1. Check **FAQ** section above
2. Search [GitHub Issues](https://github.com/yourusername/vn_legal_rag/issues)
3. Read relevant documentation file
4. Open new issue with detailed context

### Reporting Bugs

Use GitHub Issues with:
- Python version and OS
- Minimal reproducible example
- Error message and traceback
- Expected vs actual behavior

### Requesting Features

Use GitHub Issues with:
- Clear use case and motivation
- Proposed solution (if applicable)
- Reference to project roadmap/PDR

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-14 | Initial documentation package (5 docs) |

---

## License

This documentation is licensed under the MIT License - see [LICENSE](../LICENSE) file for details.

---

**Last Updated**: 2026-02-14
**Maintainer**: Documentation Team
**Next Review**: 2026-03-31 (End of Phase 2)
