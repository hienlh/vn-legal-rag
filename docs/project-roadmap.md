# Project Roadmap & Development Plan

**Project**: Vietnamese Legal RAG (vn_legal_rag)
**Last Updated**: 2026-02-14
**Current Version**: 1.0.0 (Research Prototype)
**Next Version Target**: 1.1.0 (Beta, Q2 2026)

## Vision & Long-Term Goals

### 5-Year Vision (2026-2031)

**Year 1 (2026)**: Establish foundation and community
- Complete core 3-tier system (done)
- Production-ready API and deployment
- Comprehensive documentation
- Open-source release with MIT license

**Year 2 (2027)**: Expand legal domain coverage
- Support 5+ Vietnamese legal domains (currently: corporate law only)
- Multi-language support (English, French, Chinese)
- Advanced reasoning capabilities (multi-hop Q&A)

**Year 3 (2028)**: Enterprise integration
- SaaS platform for legal firms
- Custom domain fine-tuning service
- Integration with legal tech ecosystem

**Years 4-5 (2029-2031)**: AI-powered legal assistance
- Generative features (document drafting, contract analysis)
- Real-time legal updates
- Predictive legal insights

### Success Metrics (Year 1)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Hit@10 | 75%+ | 76.53% | ✓ Exceeded |
| MRR | 0.50+ | 0.5422 | ✓ Exceeded |
| Code Coverage | 80%+ | TBD | ⏳ In Progress |
| API Response Time (p95) | <5s | ~7s | ⏳ Optimizing |
| Documentation Completeness | 100% | 95% | ⏳ Near Complete |
| GitHub Stars | 500+ | 0 | ⏳ Post-release |
| User Base | 100+ active users | 0 | ⏳ Post-launch |

## Phase Breakdown

### Phase 1: Core Development (COMPLETED - January 2026)

**Status**: ✓ Complete | **Delivery Date**: 2026-01-31

**Deliverables**:
- [x] Offline KG extraction pipeline
- [x] Online 3-tier retrieval system
- [x] Query type detection (5 categories)
- [x] Vietnamese legal citation formatting
- [x] Knowledge graph (1299 entities, 2577 relations)
- [x] Evaluation on 379 Q&A pairs
- [x] 76.53% Hit@10 performance target achieved

**Key Achievements**:
- 3-tier architecture outperforms baselines by 2.85x
- Ablation studies confirm synergy between tiers
- Support for 9 Vietnamese legal decrees
- Clean, modular Python codebase

**Lessons Learned**:
- Tier 1 (tree traversal) excellent for article_lookup (82.05%) but weak for comparisons (50%)
- Semantic embeddings (Tier 2) critical for robustness
- RRF fusion (Tier 3) provides consistent incremental boost (+5.3%)
- LLM response caching crucial for cost/latency optimization

### Phase 2: Production Hardening (CURRENT - Q1 2026)

**Status**: 🟡 In Progress | **Target Completion**: 2026-03-31 | **Duration**: 6 weeks

**Epic 1: Performance Optimization** (Week 1-2)

**Goals**:
- Reduce query latency from ~7s to <5s (p95)
- Improve cache hit rate to >50%
- Reduce memory footprint to <1.5GB

**Tasks**:
- [ ] Profile tier 1 & 2 execution (identify bottlenecks)
- [ ] Implement batch embedding inference (Tier 2)
- [ ] Optimize KG traversal (Tier 3)
- [ ] Cache article summaries in memory (Tier 1)
- [ ] Benchmark on 100-query dataset
- [ ] Document performance tuning guide

**Success Criteria**:
- Query latency p95 <5 seconds
- Cache hit rate ≥50% on repeated queries
- Memory usage <1.5GB during peak load

**Effort**: 80 hours

---

**Epic 2: Comprehensive Testing** (Week 2-4)

**Goals**:
- Achieve ≥80% code coverage
- Add integration tests for all tiers
- Stress test with high concurrency

**Tasks**:
- [ ] Add unit tests for each module (offline, online, types, utils)
- [ ] Add integration tests for 3-tier pipeline
- [ ] Add ablation test (disable each tier independently)
- [ ] Add stress test (100 parallel queries)
- [ ] Add regression test suite (all 379 test queries)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Add code coverage reporting (codecov.io)

**Success Criteria**:
- Unit + integration coverage ≥80%
- All tests pass on CI/CD
- No regressions between versions

**Effort**: 120 hours

---

**Epic 3: Documentation Completion** (Week 3-4)

**Goals**:
- Complete all documentation per PDR spec
- Create developer onboarding guide
- Generate API reference

**Tasks**:
- [x] Create codebase-summary.md (overview of all modules)
- [x] Create project-overview-pdr.md (PDR & requirements)
- [x] Create code-standards.md (coding conventions)
- [x] Create system-architecture.md (3-tier design)
- [x] Create project-roadmap.md (this file)
- [ ] Create api-reference.md (public API docs)
- [ ] Create developer-onboarding-guide.md (quickstart for new devs)
- [ ] Create deployment-guide.md (how to deploy)
- [ ] Create troubleshooting-guide.md (common issues)

**Success Criteria**:
- All docs live in /docs/
- API reference auto-generated from docstrings
- Zero broken links in documentation
- 5+ new developers successfully onboard using guides

**Effort**: 60 hours

---

**Epic 4: Security Hardening** (Week 4-5)

**Goals**:
- Security audit and remediation
- Dependency scanning and updates
- OWASP compliance

**Tasks**:
- [ ] Security code review (3-tier logic)
- [ ] Dependency audit (check for known CVEs)
- [ ] SQL injection test (SQLAlchemy parametrization)
- [ ] API key protection verification (.env only)
- [ ] Rate limiting implementation (for API)
- [ ] Input validation audit (query length, ID format)
- [ ] OWASP top 10 compliance check

**Success Criteria**:
- Zero critical/high severity issues
- All dependencies up-to-date
- Security report published

**Effort**: 60 hours

---

**Epic 5: API & Deployment** (Week 5-6)

**Goals**:
- Build REST API wrapper
- Docker containerization
- Deployment guide

**Tasks**:
- [ ] Implement FastAPI wrapper for LegalGraphRAG
- [ ] Add request/response validation
- [ ] Create OpenAPI specification (Swagger)
- [ ] Write Docker Dockerfile and docker-compose.yml
- [ ] Create deployment guide (local, Docker, cloud)
- [ ] Test deployment on sample cloud provider
- [ ] Create performance benchmark suite

**Success Criteria**:
- REST API responds to legal queries
- API documentation (Swagger UI) complete
- Runs in Docker with <100MB overhead

**Effort**: 100 hours

---

**Phase 2 Summary**:
- **Total Effort**: ~420 hours (6 person-weeks)
- **Key Dependencies**: None (can run in parallel)
- **Risk**: Performance optimization may require algorithm changes (medium risk)

### Phase 3: Feature Enhancement (Q2 2026)

**Status**: 📋 Planned | **Target**: 2026-06-30 | **Duration**: 8 weeks

**Epic 1: Query Disambiguation & Clarification** (Week 1-2)

**Goal**: Enable multi-turn conversations for ambiguous queries

**Tasks**:
- [ ] Implement clarifying question generation
- [ ] Add conversation context tracking
- [ ] Support query refinement loops
- [ ] Store conversation history (per-session)

**Example**:
```
User: "Phạt bao nhiêu khi vi phạm Điều 5?"
System: "Có nhiều loại vi phạm Điều 5 (kinh doanh không đăng ký, vốn điều lệ, v.v.).
         Bạn đang hỏi về loại nào?"
User: "Kinh doanh không đăng ký."
System: [Returns specific penalty]
```

---

**Epic 2: Advanced KG Reasoning** (Week 2-4)

**Goal**: Support complex multi-hop legal reasoning

**Tasks**:
- [ ] Implement 3+ hop KG traversal (currently max 2)
- [ ] Add relation chain reasoning (A requires B, B requires C → A requires C)
- [ ] Implement legal concept graphs (abstract away articles)
- [ ] Add citation path tracing (show full legal chain)

**Example**:
```
Query: "Tôi muốn thành lập công ty cổ phần. Cần làm gì?"
Answer:
1. Thành lập công ty → Điều 10
2. Điều 10 requires vốn điều lệ minimum → Điều 12
3. Điều 12 references corporate structure → Điều 20
4. [Full chain with all intermediate requirements]
```

---

**Epic 3: Feedback Loop & Ranking Optimization** (Week 4-6)

**Goal**: Learn from user feedback to improve rankings

**Tasks**:
- [ ] Add user feedback API (thumbs up/down on answers)
- [ ] Implement feedback collection pipeline
- [ ] Create ranking optimization toolkit
- [ ] A/B test ranking algorithms
- [ ] Continuous online learning (retrain weights)

**Success Criteria**:
- Collect 1000+ feedback signals
- Improve Hit@10 by 2%+ through feedback

---

**Epic 4: Domain Expansion** (Week 6-8)

**Goal**: Support multiple Vietnamese legal domains

**Currently Supported**: Corporate law (Luật Doanh nghiệp)

**New Domains (Phase 3)**:
1. Labor Law (Luật Lao động 2019)
2. Tax Law (Luật Thuế giá trị gia tăng)
3. Environmental Law (Luật Bảo vệ môi trường)
4. Intellectual Property (Luật Sở hữu trí tuệ)
5. Contract Law (Luật Giao dịch dân sự)

**Per-Domain Effort**:
- Extract KG: 20 hours
- Generate summaries: 5 hours
- Tune weights: 10 hours
- Validation: 5 hours
- **Total per domain**: 40 hours

**Tasks**:
- [ ] Acquire legal documents for 5 new domains
- [ ] Extract KG for each domain (offline pipeline)
- [ ] Generate summaries (chapter + article level)
- [ ] Create domain configs (YAML)
- [ ] Tune DualLevel weights per domain
- [ ] Validate on domain-specific test sets
- [ ] Document domain-specific performance

---

**Phase 3 Target State**:
- Support 6 Vietnamese legal domains (corporate + 5 new)
- Multi-turn conversation capability
- Advanced multi-hop reasoning
- Feedback-driven ranking optimization
- Hit@10 improved to 78%+

**Estimated Effort**: 320 hours (8 person-weeks)

### Phase 4: Ecosystem Integration (Q3-Q4 2026)

**Status**: 📋 Planned | **Target**: 2026-12-31

**Epic 1: Legal Tech Integration**

**Goals**: Connect with legal document management systems

**Tasks**:
- [ ] Microsoft Word plugin for inline citations
- [ ] Google Docs add-on
- [ ] PDF annotation support
- [ ] Integration with legal research databases

---

**Epic 2: Enterprise Features**

**Goals**: SaaS platform for law firms

**Tasks**:
- [ ] Multi-user authentication & authorization
- [ ] Custom knowledge base per firm
- [ ] Usage analytics & reporting
- [ ] White-label deployment option
- [ ] SLA & support

---

**Epic 3: Generative Features**

**Goals**: AI-assisted legal document generation

**Tasks**:
- [ ] Contract template matching
- [ ] Clause drafting assistance
- [ ] Legal memo generation
- [ ] Case law summarization

---

## Technology Debt & Maintenance

### Current Technical Debt: LOW

**Code Quality**:
- ✓ Clean, modular architecture
- ✓ Type hints on public APIs
- ✓ Comprehensive docstrings
- ⚠️ Test coverage gaps (target: 80%+)

**Refactoring Opportunities**:
1. **Tier 1 Simplification** (Medium effort)
   - Extract LLM prompting logic to separate module
   - Reduce code duplication in Loop 1 & 2

2. **Configuration Management** (Low effort)
   - Unify DualLevelConfig, AblationConfig into single ConfigManager
   - Add config validation schema

3. **Error Handling** (Medium effort)
   - Standardize exception types across modules
   - Add error recovery strategies for each tier

### Dependency Management

**Current Dependencies**: 15 core, 8 optional LLM providers

**Update Strategy**:
- Monthly security scanning (Dependabot)
- Quarterly major version updates
- Maintain compatibility with Python 3.10-3.12

**High-Risk Dependencies**:
- `sqlalchemy>=2.0` (ORM breaking changes)
- `numpy/scipy` (numerical API changes)
- `openai/anthropic` (API versioning)

## Release Schedule

### Versioning Strategy: Semantic Versioning

**Format**: MAJOR.MINOR.PATCH (e.g., 1.0.0)

| Version | Release Date | Status | Focus |
|---------|--------------|--------|-------|
| 1.0.0 | 2026-01-31 | ✓ Released | Core 3-tier system |
| 1.1.0 | 2026-03-31 | 🟡 Q1 2026 | Performance + testing + docs |
| 1.2.0 | 2026-06-30 | 📋 Q2 2026 | Query disambiguation + domain expansion |
| 2.0.0 | 2026-12-31 | 📋 Q3-Q4 2026 | Ecosystem integration + generative features |
| 2.1.0 | 2027-06-30 | 📋 2027 | Multi-language support |

### Release Checklist (Per Release)

```markdown
## Pre-Release (1 week before)
- [ ] Freeze feature development
- [ ] Run full test suite
- [ ] Update CHANGELOG.md
- [ ] Update version numbers (pyproject.toml, __init__.py)
- [ ] Security audit & vulnerability scan
- [ ] Performance benchmarking
- [ ] Documentation review

## Release Day
- [ ] Create git tag: git tag v1.X.0
- [ ] Push to GitHub: git push origin v1.X.0
- [ ] Build package: python -m build
- [ ] Publish to PyPI: twine upload dist/*
- [ ] Create GitHub release with CHANGELOG
- [ ] Announce on social media/forums

## Post-Release (1 week after)
- [ ] Monitor bug reports
- [ ] Patch critical issues immediately
- [ ] Collect user feedback
- [ ] Plan next phase based on feedback
```

## Key Milestones

| Milestone | Date | Target | Status |
|-----------|------|--------|--------|
| **Core system complete** | 2026-01-31 | Hit@10 ≥75% | ✓ Achieved (76.53%) |
| **Phase 2 complete** | 2026-03-31 | Test coverage ≥80%, API ready | 🟡 In Progress |
| **Open-source release** | 2026-04-15 | GitHub public + PyPI package | 📋 Planned |
| **Domain expansion** | 2026-06-30 | Support 6 legal domains | 📋 Planned |
| **10,000+ GitHub stars** | 2026-12-31 | Community adoption | 📋 Aspirational |
| **100+ law firms using** | 2027-06-30 | Enterprise adoption | 📋 Long-term |

## Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|-----------|-------|
| LLM API rate limits | Medium | High | Implement robust caching, batch inference | Backend |
| KG extraction accuracy <80% | Low | Medium | Manual validation, quality thresholds, retraining | Data |
| Scaling to 10K+ articles | Medium | High | Database indexing, caching, async inference | Backend |
| New domain coverage gaps | Medium | Medium | Flexible config system, fallback strategies | Data |
| Model drift over time | Low | Medium | Monitor evaluation metrics, retraining pipeline | ML |

### Organizational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Slow open-source adoption | Medium | Medium | Strong documentation, examples, community engagement |
| Competing solutions | High | Low | Differentiate on legal-domain specialization |
| Regulatory changes | Low | High | Modular architecture enables quick updates |
| Key person dependency | Low | High | Knowledge sharing, comprehensive documentation |

### Mitigation Strategy

1. **Technical**: Invest in Phase 2 quality (testing, performance)
2. **Organizational**: Strong open-source practices (contrib guidelines, code of conduct)
3. **Regulatory**: Maintain flexibility (modular design, versioned knowledge graphs)
4. **Talent**: Document decisions (architecture decisions record), cross-train team

## Success Criteria by Phase

### Phase 2 (Production Ready)
- [x] Hit@10 ≥75% (target achieved)
- [ ] Query latency p95 <5 seconds
- [ ] Test coverage ≥80%
- [ ] Zero critical security issues
- [ ] Full documentation (codebase summary, API reference, deployment guide)
- [ ] REST API functional and documented
- [ ] Docker deployment working
- [ ] Performance benchmarks published

### Phase 3 (Feature Rich)
- [ ] Support 6 legal domains (corporate + 5 new)
- [ ] Multi-turn conversation support
- [ ] Advanced KG reasoning (3+ hops)
- [ ] User feedback system integrated
- [ ] Hit@10 improved to 78%+
- [ ] 1000+ GitHub stars

### Phase 4 (Ecosystem)
- [ ] SaaS platform operational
- [ ] 10+ law firms using system
- [ ] Office plugin integrations
- [ ] Generative features (drafting, summarization)
- [ ] 10,000+ GitHub stars

## Budget & Resource Allocation

### Phase 2 Budget (6 weeks, $80K)

| Category | Effort | Cost |
|----------|--------|------|
| Performance optimization | 80h | $8,000 |
| Testing & QA | 120h | $12,000 |
| Documentation | 60h | $6,000 |
| Security audit | 60h | $9,000 |
| API & deployment | 100h | $10,000 |
| Infrastructure & tools | - | $20,000 |
| Contingency (20%) | - | $15,000 |
| **Total** | 420h | **$80,000** |

### Phase 3 Budget (8 weeks, $100K)

| Category | Effort | Cost |
|----------|--------|------|
| Query disambiguation | 60h | $6,000 |
| Advanced KG reasoning | 120h | $12,000 |
| Feedback & optimization | 100h | $10,000 |
| Domain expansion (5 domains) | 200h | $20,000 |
| Testing & validation | 80h | $8,000 |
| Infrastructure scaling | - | $30,000 |
| Contingency (20%) | - | $14,000 |
| **Total** | 560h | **$100,000** |

## Communication & Stakeholder Updates

### Monthly Status Report (1st of each month)

**Contents**:
- Milestone progress (% complete)
- Key achievements & blockers
- Performance metrics (Hit@10, latency, cache rate)
- Risk updates
- Next month priorities

**Recipients**: Project sponsors, core team, advisory board

### Quarterly Business Review (Q2/Q3/Q4)

**Agenda**:
- Phase completion review
- Budget vs. actual
- User feedback summary
- Roadmap adjustments
- Next phase kickoff

**Duration**: 2 hours

### Public Roadmap Updates

**Frequency**: Monthly on GitHub Discussions
**Format**: Simplified version of this document
**Transparency**: Show completed, in-progress, planned items

## Contingency Plans

### If Performance Optimization (Phase 2) Slips

**Risk**: Query latency exceeds <5s target

**Mitigation**:
- Push to Phase 3 with "acceptable" 6-7s latency
- Prioritize caching as first optimization in Phase 3
- Consider outsourcing inference (cloud GPU)

### If Domain Expansion (Phase 3) Stalls

**Risk**: Extracting KG for new domains takes longer than expected

**Mitigation**:
- Focus on 3 domains instead of 5
- Leverage transfer learning from corporate law domain
- Reduce per-domain validation requirements

### If Open-Source Adoption Is Low

**Risk**: GitHub stars <1000 by year-end

**Mitigation**:
- Invest in developer marketing (blog posts, talks)
- Create template use-cases (e.g., "Build Your Legal Chatbot")
- Partner with legal tech communities
- Sponsor legal tech conferences

## How to Use This Roadmap

### For Project Managers
- Track phase completion via milestones
- Update risk register quarterly
- Manage budget allocation per phase

### For Developers
- Prioritize epics/tasks in your phase
- Reference success criteria for acceptance
- Report blockers weekly

### For Product Owners
- Validate features against user needs
- Gather user feedback for Phase 3+
- Adjust roadmap based on market response

### For Community
- Track public progress on GitHub
- Submit feature requests via Issues
- Contribute to planned features

---

## Appendix: Historical Context

### Project Genesis (2025)

Vietnamese Legal RAG started as a research project to demonstrate effectiveness of graph-based retrieval for Vietnamese legal documents. Initial inspiration:
- LightRAG (Gao et al., 2024) for graph-augmented RAG
- PageIndex (Shi et al., 2024) for hierarchical navigation
- Vietnamese legal corpus from thuvienphapluat.vn

### Key Design Decisions

1. **3-Tier Architecture**: Chose tiered approach (vs. single monolithic retriever) for:
   - Interpretability: Each tier has clear responsibility
   - Robustness: Tier failure doesn't eliminate all results
   - Tunability: Can adjust per-tier weights

2. **LLM for Tree Navigation (Tier 1)**: Chose LLM-guided traversal instead of ML classifier because:
   - Faster to implement (prompt engineering vs. training)
   - Transparent reasoning (users see LLM's logic)
   - Adaptable to new domains without retraining

3. **Knowledge Graph Instead of Just Vectors**: Chose KG because:
   - Explicit entity/relation representation
   - Multi-hop reasoning capability
   - Explainability (show reasoning chains)

### Lessons Learned (Phase 1)

1. **Domain Specificity Matters**: Generic embeddings (OpenAI) good but specialized legal embeddings better
2. **Tier Synergy**: RRF fusion (Tier 3) provides consistent +5% improvement
3. **Query Type Diversity**: Single tier dominates different query types (Tier 1 for lookups, Tier 2 for complex)
4. **LLM Caching Critical**: LLM response cache reduces latency 50% + cost 90% on repeated queries

---

**Last Updated**: 2026-02-14
**Version**: 1.0
**Maintainer**: Project Management Team
**Next Review Date**: 2026-03-31 (End of Phase 2)
