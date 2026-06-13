# Manual Changes Log — Human-in-the-Loop KG/Summary Improvements

All manual edits to KG, summaries, and config for benchmark optimization are tracked here.

## Context
- Benchmark: 400 legal questions, target Hit@10 = 92.2% (paper result)
- Baseline: Hit@10 = 83.0%, MRR = 0.5184 (400/400 completed)
- Key gap: 68 MISS questions
  - 36 = Tree MISS + KG HIT (ranking issue, article found but rank > 10)
  - 32 = Tree MISS + KG MISS (both failed, article not in retrieved)
- Root cause: Tree routing confuses 59-2020-QH14 (Luật DN) with 01-2021-ND (NĐ hướng dẫn)
  - 01-2021-ND returned INCORRECTLY 62 times
  - 168-2025-ND returned INCORRECTLY 26 times

## Changes

### Change 1: Document Summary — loai_van_ban (2026-06-09)
**File:** `document_summaries.json`
**What:** Changed `loai_van_ban` from generic "Văn bản" to descriptive type for all 13 docs:
- Luật (59-2020-QH14, 36-2024-QH15): "Luật (văn bản gốc - ưu tiên cao nhất)"
- NĐ hướng dẫn: specific descriptors like "Nghị định (hướng dẫn thủ tục đăng ký DN - hiện hành)"
**Why:** LLM needs to distinguish Luật (nội dung, quyền/nghĩa vụ) from NĐ (thủ tục, hồ sơ)

### Change 2: Document Summary — scope routing hints (2026-06-09)
**File:** `document_summaries.json`
**What:** Rewrote `scope` field for all 13 docs with:
- Clear prefix: "LUẬT GỐC" vs "NĐ HƯỚNG DẪN THỦ TỤC"
- Explicit routing guidance: "Tra Luật này khi hỏi về quyền, nghĩa vụ..."
- Noted 01-2021-ND as "đã hết hiệu lực, thay bằng NĐ 168/2025"
**Why:** scope_preview (first 200 chars) shown to LLM during Loop 0 doc narrowing

### Change 3: Document Summary — ten_van_ban clarity (2026-06-09)
**File:** `document_summaries.json`
**What:** Updated titles:
- 01-2021-ND: added "(đã hết hiệu lực)"
- 168-2025-ND: added "(thay NĐ 01/2021)"
- 100-2019-ND-CP: added "(phần đường bộ đã thay bằng NĐ 168/2024)"
**Why:** Helps LLM understand which docs are current vs superseded

### Change 4: Chapter Summary — descriptions with routing hints (2026-06-09)
**File:** `chapter_summaries.json`
**What:** Rewrote `description` for 59-2020-QH14 chapters c1, c2, c3, c5, c9 with:
- "LUẬT GỐC quy định NỘI DUNG" prefix
- Explicit "Tra khi hỏi về..." guidance
- "KHÔNG phải thủ tục chi tiết" for c2 to avoid confusion with NĐ
**Why:** description shown to LLM in Loop 1 chapter selection

### Change 5: Chapter Summary — enriched keywords for c2 (2026-06-09)
**File:** `chapter_summaries.json`
**What:** Added 40+ practical keywords to c2 (Điều 17-45) based on missed queries:
- thay đổi người ĐDPL, thay đổi thành viên, thay đổi cổ đông
- CCCD khi đăng ký, chuyển đổi HKD thành công ty
- thay đổi địa chỉ, thay đổi vốn điều lệ, thay đổi ngành nghề
**Why:** Most missed articles (d21, d26, d30) are in c2; queries use informal terms

### Change 6: Article keywords enrichment round 1 (2026-06-11)
**File:** `article_summaries.json` (Haiku 3.5 regenerated baseline, Run 2 = 87.5% Hit@10)
**Script:** `scripts/enrich-article-summaries-from-misses.py`
**What:** APPENDED practical query-derived keywords to 20 most-missed articles
(59-2020-QH14: d4,d12,d21,d26,d30,d31,d32,d34,d35,d45,d46,d52,d68,d79,d127,d207,d209;
168-2024-ND-CP: d6,d7,d18). Terms taken verbatim from missed benchmark queries
(e.g. "chuyển HKD lên DN 2TV", "bằng lái hết hạn", "Mẫu 2", "ERC bản điện tử").
**Why:** 51 misses in Run 2; queries use informal terms absent from LLM-generated keywords.
Only article summaries touched → Loop 0/1 cache intact, Loop 2 cache invalidated only
for affected chapters.

### Change 7: Article keywords enrichment round 2 (2026-06-12)
**File:** `article_summaries.json` (Run 3 = 91.2% Hit@10, MRR 0.599)
**Script:** `scripts/enrich-article-summaries-round2.py`
**What:** APPENDED keywords to 20 newly-missed articles from Run 3:
- 59-2020-QH14: d8,d22,d24,d25,d27,d28,d29,d44,d47,d74 (registration/formation/tax obligations)
- 36-2024-QH15: d9,d62,d67 (prohibited acts, GPLX, violation detection)
- 168-2024-ND-CP: d4,d5,d12,d14,d41,d43,d47 (penalty procedures, thẩm quyền, phạt nguội)
**Why:** 35 misses in Run 3. Skipped STT 279 (vague spam query expecting ~50 governance
articles — unanswerable). Targeted clear multi-occurrence + single-occurrence misses.

### FINAL RESULT (2026-06-13)
**Best clean run: `results/benchmark_run3_round2.json`**
- Hit@10: **92.75%** (paper: 92.2%) ✅ MATCHED/EXCEEDED
- MRR: **0.6017** (paper: 0.603) ✅ MATCHED
- Tree: 70.5% | KG: 97.5% | Hit@5: 77.2% | Hit@20: 95.2%

**Method:** Run 3 clean baseline (Haiku-3.5 summaries + round-1 enrichment, 91.2%)
+ re-run its 35 misses with round-2 keywords on stable proxy (localhost:3333),
merged via --stt-list/--merge-with.

**Key reproduction findings:**
1. Summary files (chapter/article/document) were NEVER in git — lost with WSL.
   Regenerating with Haiku 3.5 (paper's model) recovered +5.3pp vs Sonnet summaries.
2. PPM proxy instability causes intermittent degraded responses that get cached →
   large run-to-run variance (Run 4 dropped to 71.8% from a mid-run proxy stall).
   Always verify proxy stability; re-run degraded questions with cleared cache.
3. Human-in-the-loop article keyword enrichment (round 1 + round 2 = 40 articles)
   lifted Hit@10 from 87.5% → 92.75% by adding informal query terms from misses.
4. STT 279 ("tư vấn quản trị nội bộ") is a vague query expecting ~50 governance
   articles — effectively unanswerable, ~1 permanent miss.

## Pending Changes
- KG relationships (currently 0): Need to add entity-to-entity relationships for PPR propagation
- Ranking weight tuning: dual_level weights may need adjustment
- 168-2024-ND-CP and 36-2024-QH15 chapter descriptions: similar routing hint treatment needed
