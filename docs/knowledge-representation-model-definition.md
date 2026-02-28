# Mô hình Biểu diễn Tri thức - VN Legal RAG

## 1. Tổng quan

Hệ thống sử dụng mô hình **Đồ thị Tri thức tăng cường Ontology với Suy luận dựa trên Luật**
(Ontology-enhanced Knowledge Graph with Rule-based Inference) để biểu diễn
tri thức pháp luật Việt Nam.

Mô hình được định nghĩa formal:

```
𝕄 = (O, G, R, Φ)
```

Trong đó:
- **O**: Ontology Schema — tầng terminological
- **G**: Knowledge Graph — tầng assertional
- **R**: Rules — tầng suy luận
- **Φ**: Functions — tầng truy vấn

Mô hình kết hợp nhiều paradigm biểu diễn tri thức: OWL ontology cho schema,
đồ thị tri thức cho instances, luật suy diễn cho inference, và các hàm
xử lý ngôn ngữ tự nhiên tiếng Việt cho truy vấn.

---

## 2. O (Ontology Schema) — Tầng Terminological

```
O = (C, P, H, L)
```

### C — Tập Classes

Tập hợp các lớp khái niệm trong miền pháp luật, hiện tại gồm 30+ classes.

Phân cấp chính:
- **LegalEntity**: Chủ thể pháp luật
  - Organization → LegalOrganization → Enterprise, GovernmentAgency
  - Individual → Person
- **LegalDocument**: Văn bản pháp luật
  - Legislation → Law, Decree, Circular, Decision, Resolution
- **LegalConcept**: Khái niệm pháp luật
  - Term, Role, Action, Condition
- **Right**, **Obligation**: Quyền và nghĩa vụ
- **QuantitativeEntity**: Đại lượng định lượng
  - Percentage, Duration, MonetaryValue, Asset

### P — Tập Properties

39 properties gồm:
- **Object properties (28)**: hasRight, hasObligation, manages, references,
  requires, prohibits, permits, contains, isTypeOf, governedBy...
- **Data properties (11)**: hasName, effectiveDate, hasConfidence,
  hasText, hasSourceId, hasDuration, hasPercentage...

### H — Class Hierarchy

Quan hệ phân cấp `rdfs:subClassOf`, tối đa 4 mức.

Ví dụ: `StateOwnedEnterprise ⊑ Enterprise ⊑ LegalOrganization ⊑ Organization ⊑ Thing`

### L — Vietnamese Label Function

Ánh xạ mỗi class sang nhãn tiếng Việt:
- L(JointStockCompany) = "công ty cổ phần"
- L(LimitedLiabilityCompany) = "công ty trách nhiệm hữu hạn"
- L(BoardOfDirectors) = "hội đồng quản trị"

**Files:** `data/ontologies/base/legal-core.ttl`, `data/kg_enhanced/ontology.json`

---

## 3. G (Knowledge Graph) — Tầng Assertional

```
G = (E, R_inst, X)
```

### E — Tập Entities

1,299 thực thể trích xuất từ văn bản pháp luật.

Mỗi entity: `eᵢ = (id, type, name, properties, confidence)`
- `type(eᵢ) ∈ C` — tuân thủ ontology schema

11 entity types: ORGANIZATION, PERSON_ROLE, LEGAL_TERM, LEGAL_REFERENCE,
MONETARY, PERCENTAGE, DURATION, LOCATION, CONDITION, ACTION, PENALTY

### R_inst — Tập Relation Instances

2,709 quan hệ giữa các entities.

Mỗi relation: `rⱼ = (source, target, type, confidence, evidence)`
- `type(rⱼ) ∈ P` — tuân thủ ontology schema

### X — Cross-References

Tập con của R_inst, biểu diễn tham chiếu chéo giữa các điều khoản:
- references (tham chiếu)
- amends (sửa đổi)
- supersedes (thay thế)
- repeals (bãi bỏ)
- implements (hướng dẫn thi hành)

**File:** `data/kg_enhanced/legal_kg.json`

---

## 4. R (Rules) — Tầng Suy luận

```
R = (R_sym, R_trans, R_inv, R_trig)
```

### R_sym — Luật Đối xứng

```
A r B ⟺ B r A    (với r ∈ R_sym)
```

R_sym = {RELATED_TO, CONTRADICTS, SYNONYM, EXCLUDES}

Ví dụ: Nếu "Luật DN 2020" RELATED_TO "Nghị định 01/2021"
thì "Nghị định 01/2021" RELATED_TO "Luật DN 2020"

### R_trans — Luật Bắc cầu

```
A r B ∧ B r C → A r C    (với r ∈ R_trans)
```

R_trans = {IS_A, PART_OF, CONTAINS, PRECEDES, FOLLOWS, REQUIRES}

Ví dụ: Nếu "CTCP" IS_A "Công ty" và "Công ty" IS_A "Tổ chức"
thì suy ra "CTCP" IS_A "Tổ chức"

### R_inv — Luật Nghịch đảo

```
A r B → B inv(r) A
```

| Relation | Inverse |
|----------|---------|
| REQUIRES | CONDITION_FOR |
| CONTAINS | PART_OF |
| IS_A | CONTAINS |
| PRECEDES | FOLLOWS |
| REFERENCES | MENTIONED_IN |
| DELEGATES_TO | AUTHORIZED_BY |

### R_trig — Ánh xạ Trigger Words

```
trigger_phrase ∈ text → relation_type ∈ P
```

Ánh xạ các cụm từ tiếng Việt sang loại quan hệ pháp luật:

| Trigger phrases | Relation type |
|----------------|---------------|
| "phải có", "cần có", "yêu cầu" | REQUIRES |
| "theo", "căn cứ", "quy định tại" | REFERENCES |
| "có quyền", "được quyền" | CÓ_QUYỀN |
| "có nghĩa vụ", "phải" | CÓ_NGHĨA_VỤ |
| "nghiêm cấm", "cấm" | NGHIÊM_CẤM |
| "bao gồm", "gồm có" | INCLUDES |
| "sửa đổi", "bổ sung" | AMENDS |
| "là một", "thuộc loại" | IS_A |
| ... | (34+ relation types) |

**File:** `vn_legal_rag/offline/relation_types.py`

---

## 5. Φ (Functions) — Tầng Truy vấn

```
Φ = (φ_match, φ_expand, φ_rank)
```

### φ_match — Khớp thuật ngữ tiếng Việt

```
φ_match: Query → {terms}
```

- Giải quyết viết tắt: "ctcp" → JointStockCompany, "tnhh" → LLC, "hđqt" → BoardOfDirectors
- Pattern matching: regex cho thuật ngữ pháp luật
- NLP matcher: khớp không dấu, word segmentation

### φ_expand — Mở rộng dựa trên ontology

```
φ_expand: term → {terms}
```

- Mở rộng lên (parent): Enterprise → Organization → LegalEntity
- Mở rộng xuống (children): Company → {CTCP, TNHH, DNTN, Hợp Danh, HKD}
- Mở rộng ngang (siblings): CTCP ↔ TNHH ↔ DNTN

### φ_rank — Xếp hạng đa thành phần

```
φ_rank: {entities} × Query → ranked results
```

- Personalized PageRank trên KG (xử lý symmetric edges)
- 6 thành phần scoring: keyphrase, semantic, PPR, concept, theme, hierarchy

**Files:**
- `vn_legal_rag/online/ontology-based-query-expander.py`
- `vn_legal_rag/online/vietnamese-nlp-term-matcher.py`
- `vn_legal_rag/online/personalized-page-rank-for-kg.py`

---

## 6. Diagram tổng quan

```
┌──────────────────────────────────────────────────────────────┐
│                      𝕄 = (O, G, R, Φ)                        │
│       Ontology-enhanced KG with Rule-based Inference         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐       ┌───────────────────────────┐   │
│  │  O (Ontology)     │       │   G (Knowledge Graph)     │   │
│  │  Schema Layer     │◄─────►│   Instance Layer           │   │
│  │                   │ type  │                            │   │
│  │  C: 30 classes    │ check │  E: 1,299 entities        │   │
│  │  P: 39 properties │       │  R: 2,709 relations       │   │
│  │  H: 4-level hier  │       │  X: cross-references      │   │
│  │  L: VN labels     │       │  confidence scores        │   │
│  └─────────┬─────────┘       └──────────┬────────────────┘   │
│            │                            │                     │
│            └────────────┬───────────────┘                     │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                R (Rules Layer)                        │    │
│  │                                                      │    │
│  │  R_sym:   A r B ⟺ B r A                             │    │
│  │  R_trans: A r B ∧ B r C → A r C                     │    │
│  │  R_inv:   A r B → B inv(r) A                        │    │
│  │  R_trig:  trigger_phrase → relation_type             │    │
│  └─────────────────────┬────────────────────────────────┘    │
│                        ▼                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Φ (Query Functions)                      │    │
│  │                                                      │    │
│  │  φ_match:  Query → terms (NLP + abbreviations)      │    │
│  │  φ_expand: term → terms (ontology hierarchy)        │    │
│  │  φ_rank:   entities × query → ranked results        │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Liên hệ với Legal Rela-model

Mô hình Legal Rela-model [1] được định nghĩa: `𝕏 = (C, R, RULES) + (Key, Rel, weight)`

### Component mapping

| Legal Rela-model | VN Legal RAG | Ghi chú |
|-----------------|-------------|---------|
| C (Concepts) | O.C (OWL classes) | Tương tự, ta dùng chuẩn OWL |
| R_concept (is-a, has-a, part-of) | O.H + O.P | Tương đương |
| R_database | G.R_inst + G.X | Tương đương |
| RULES symmetric | R.R_sym | Giống |
| RULES transitive | R.R_trans | Giống |
| RULES property | R.R_inv | Giống |
| (Key, Rel, weight) | Φ functions | Khác approach, cùng mục đích |
| weight (tf-idf) | Φ.φ_rank (PPR + 6-comp) | Khác kỹ thuật |

### Điểm tương đồng

- Concept hierarchy phân cấp
- Luật suy diễn: symmetric, transitive
- Trigger words ↔ keyphrases dictionary
- Quan hệ giữa concepts ↔ R_concept

### Điểm khác biệt

| Aspect | Legal Rela-model | VN Legal RAG |
|--------|-----------------|-------------|
| Schema format | Custom structure | OWL 2 standard |
| Similarity | tf-idf keyphrase | PPR + multi-component scoring |
| Keyphrase graph | Explicit (Key, Rel, weight) | Distributed trong Φ |
| Scope | Luật Đất đai | Pháp luật doanh nghiệp VN (general) |
| Architecture | Integrated model | Modular components |

### Thành phần chưa có

- Explicit keyphrase-to-keyphrase similarity graph
- InnerRel (ánh xạ concept → article cụ thể)

---

## References

- [1] Nguyen et al., "Legal-Onto: An Ontology-based Model for Representing
      the Knowledge of a Legal Document", ENASE 2022
- [2] Ontology: `data/ontologies/base/legal-core.ttl`
- [3] Knowledge Graph: `data/kg_enhanced/legal_kg.json`
- [4] Rules: `vn_legal_rag/offline/relation_types.py`
- [5] Query Expander: `vn_legal_rag/online/ontology-based-query-expander.py`
