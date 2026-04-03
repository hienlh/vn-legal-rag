"""Add REFERENCES relations to KG for Round 3 optimization.

Usage:
  python scripts/kg-round3-add-relations.py --batch 1
  python scripts/kg-round3-add-relations.py --batch 1 --dry-run  # preview only
  python scripts/kg-round3-add-relations.py --batch all
"""

import json
import argparse
from pathlib import Path

KG_FILE = "data/kg_enhanced/legal_kg.json"


def make_ref(source: str, target: str, desc: str, confidence: float = 0.95) -> dict:
    return {
        "source": source,
        "target": target,
        "type": "REFERENCES",
        "description": desc,
        "confidence": confidence,
        "weight": 1.0,
    }


def make_pair(a: str, b: str, desc_ab: str, desc_ba: str = "", confidence: float = 0.95) -> list[dict]:
    """Create bidirectional REFERENCES pair."""
    if not desc_ba:
        desc_ba = desc_ab
    return [make_ref(a, b, desc_ab, confidence), make_ref(b, a, desc_ba, confidence)]


# ============================================================
# BATCH 1: d7/d8/d6 hub → procedure & law articles
# Target: d47(21 miss), d6(21), d5(7), d48(6), d32(11)
# d7 appears in top-10 of 50+ MISS queries
# ============================================================
BATCH_1 = []

ND = "168-2024-ND-CP"
QH = "36-2024-QH15"

# d7 (xe máy penalty) → procedure articles
BATCH_1 += make_pair(f"{ND}:d7", f"{ND}:d47", "Phạt xe máy → tạm giữ phương tiện vi phạm")
BATCH_1 += make_pair(f"{ND}:d7", f"{ND}:d5", "Phạt xe máy → phạt xe đạp/người đi bộ")
BATCH_1 += make_pair(f"{ND}:d7", f"{ND}:d48", "Phạt xe máy → trả lại GPLX/phương tiện")
BATCH_1 += make_pair(f"{ND}:d7", f"{ND}:d32", "Phạt xe máy → phạt sang tên/đăng ký xe")
BATCH_1 += make_pair(f"{ND}:d7", f"{ND}:d14", "Phạt xe máy → phạt hành khách/người ngồi trên xe")
BATCH_1 += make_pair(f"{ND}:d7", f"{ND}:d18", "Phạt xe máy → phạt không có GPLX")

# d7 → law articles (cross-doc)
BATCH_1 += make_pair(f"{ND}:d7", f"{QH}:d11", "Phạt xe máy → tín hiệu đèn giao thông")
BATCH_1 += make_pair(f"{ND}:d7", f"{QH}:d9", "Phạt xe máy → điều kiện người lái xe")
BATCH_1 += make_pair(f"{ND}:d7", f"{QH}:d67", "Phạt xe máy → đăng kiểm phương tiện")
BATCH_1 += make_pair(f"{ND}:d7", f"{QH}:d42", "Phạt xe máy → quyền chủ phương tiện")
BATCH_1 += make_pair(f"{ND}:d7", f"{QH}:d18", "Phạt xe máy → quyền/nghĩa vụ người lái xe")
BATCH_1 += make_pair(f"{ND}:d7", f"{QH}:d13", "Phạt xe máy → quy tắc giao thông đường bộ")

# d8 (xe thô sơ penalty) → key articles
BATCH_1 += make_pair(f"{ND}:d8", f"{ND}:d47", "Phạt xe thô sơ → tạm giữ phương tiện")
BATCH_1 += make_pair(f"{ND}:d8", f"{ND}:d6", "Phạt xe thô sơ → phạt xe ô tô")
BATCH_1 += make_pair(f"{ND}:d8", f"{ND}:d5", "Phạt xe thô sơ → phạt xe đạp/người đi bộ")
BATCH_1 += make_pair(f"{ND}:d8", f"{QH}:d11", "Phạt xe thô sơ → tín hiệu đèn giao thông")
BATCH_1 += make_pair(f"{ND}:d8", f"{QH}:d9", "Phạt xe thô sơ → điều kiện lái xe")
BATCH_1 += make_pair(f"{ND}:d8", f"{QH}:d67", "Phạt xe thô sơ → đăng kiểm phương tiện")

# d6 (ô tô penalty) → procedure articles
BATCH_1 += make_pair(f"{ND}:d6", f"{ND}:d47", "Phạt xe ô tô → tạm giữ phương tiện")
BATCH_1 += make_pair(f"{ND}:d6", f"{ND}:d48", "Phạt xe ô tô → trả lại GPLX/phương tiện")
BATCH_1 += make_pair(f"{ND}:d6", f"{ND}:d5", "Phạt xe ô tô → phạt xe đạp/người đi bộ")
BATCH_1 += make_pair(f"{ND}:d6", f"{QH}:d11", "Phạt xe ô tô → tín hiệu đèn giao thông")
BATCH_1 += make_pair(f"{ND}:d6", f"{QH}:d9", "Phạt xe ô tô → điều kiện lái xe")
BATCH_1 += make_pair(f"{ND}:d6", f"{QH}:d67", "Phạt xe ô tô → đăng kiểm phương tiện")

# d5 (xe đạp/người đi bộ) → procedure
BATCH_1 += make_pair(f"{ND}:d5", f"{ND}:d47", "Phạt xe đạp/người đi bộ → tạm giữ phương tiện")
BATCH_1 += make_pair(f"{ND}:d5", f"{ND}:d48", "Phạt xe đạp/người đi bộ → trả lại GPLX/phương tiện")

# d41/d42/d43 (thẩm quyền) → d47 (already d41→d43 exists, need →d47)
BATCH_1 += make_pair(f"{ND}:d41", f"{ND}:d47", "Thẩm quyền CSGT cấp huyện → tạm giữ phương tiện")
BATCH_1 += make_pair(f"{ND}:d42", f"{ND}:d47", "Thẩm quyền CSGT cấp tỉnh → tạm giữ phương tiện")
BATCH_1 += make_pair(f"{ND}:d43", f"{ND}:d47", "Thẩm quyền Chủ tịch UBND → tạm giữ phương tiện")
BATCH_1 += make_pair(f"{ND}:d41", f"{ND}:d48", "Thẩm quyền CSGT cấp huyện → trả lại GPLX")
BATCH_1 += make_pair(f"{ND}:d42", f"{ND}:d48", "Thẩm quyền CSGT cấp tỉnh → trả lại GPLX")
BATCH_1 += make_pair(f"{ND}:d43", f"{ND}:d48", "Thẩm quyền Chủ tịch UBND → trả lại GPLX")

# d46 (biên bản vi phạm) → d48
BATCH_1 += make_pair(f"{ND}:d46", f"{ND}:d48", "Biên bản vi phạm → trả lại GPLX/phương tiện")

# ============================================================
# BATCH 2: Cross-doc traffic decree ↔ law
# Target: 36-QH15:d11(9), d9(8), d67(6), d36(5), d18(4), d42(3)
# ============================================================
BATCH_2 = []

# 168-ND penalty → 36-QH15 law articles
BATCH_2 += make_pair(f"{ND}:d6", f"{QH}:d36", "Phạt ô tô → đăng ký phương tiện")
BATCH_2 += make_pair(f"{ND}:d7", f"{QH}:d36", "Phạt xe máy → đăng ký phương tiện")
BATCH_2 += make_pair(f"{ND}:d32", f"{QH}:d36", "Phạt sang tên xe → đăng ký phương tiện")
BATCH_2 += make_pair(f"{ND}:d32", f"{QH}:d42", "Phạt sang tên xe → quyền chủ phương tiện")
BATCH_2 += make_pair(f"{ND}:d32", f"{QH}:d43", "Phạt sang tên xe → nghĩa vụ chủ phương tiện")
BATCH_2 += make_pair(f"{ND}:d14", f"{QH}:d18", "Phạt hành khách → quyền/nghĩa vụ hành khách")
BATCH_2 += make_pair(f"{ND}:d47", f"{QH}:d62", "Tạm giữ phương tiện → tạm giữ (luật)")
BATCH_2 += make_pair(f"{ND}:d18", f"{QH}:d9", "Phạt không GPLX → điều kiện lái xe")

# 36-QH15 intra-doc
BATCH_2 += make_pair(f"{QH}:d25", f"{QH}:d11", "Quy tắc giao thông → tín hiệu đèn")
BATCH_2 += make_pair(f"{QH}:d24", f"{QH}:d11", "Quy tắc xe cơ giới → tín hiệu đèn")
BATCH_2 += make_pair(f"{QH}:d39", f"{QH}:d36", "Đăng ký xe mô tô → đăng ký phương tiện chung")
BATCH_2 += make_pair(f"{QH}:d42", f"{QH}:d43", "Quyền chủ PT → nghĩa vụ chủ PT")
BATCH_2 += make_pair(f"{QH}:d56", f"{QH}:d67", "GPLX → đăng kiểm phương tiện")
BATCH_2 += make_pair(f"{QH}:d56", f"{QH}:d9", "GPLX → điều kiện lái xe")
BATCH_2 += make_pair(f"{QH}:d36", f"{QH}:d67", "Đăng ký PT → đăng kiểm PT")
BATCH_2 += make_pair(f"{QH}:d12", f"{QH}:d11", "Biển báo → tín hiệu đèn")

# ============================================================
# BATCH 3: Cross-decree 100-2019 → 168-2024
# Target: queries getting old decree instead of new
# ============================================================
BATCH_3 = []

OLD = "100-2019-ND-CP"

BATCH_3 += make_pair(f"{OLD}:d5", f"{ND}:d5", "NĐ100-2019 phạt xe đạp cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d6", f"{ND}:d6", "NĐ100-2019 phạt ô tô cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d6", f"{ND}:d7", "NĐ100-2019 phạt ô tô/xe máy cũ → NĐ168-2024 xe máy mới")
BATCH_3 += make_pair(f"{OLD}:d8", f"{ND}:d8", "NĐ100-2019 phạt xe thô sơ cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d74", f"{ND}:d41", "NĐ100-2019 thẩm quyền cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d75", f"{ND}:d42", "NĐ100-2019 thẩm quyền cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d76", f"{ND}:d43", "NĐ100-2019 thẩm quyền cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d80", f"{ND}:d47", "NĐ100-2019 tạm giữ cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d82", f"{ND}:d48", "NĐ100-2019 trả GPLX cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d63", f"{ND}:d32", "NĐ100-2019 sang tên xe cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d16", f"{ND}:d14", "NĐ100-2019 phạt hành khách cũ → NĐ168-2024 mới")
BATCH_3 += make_pair(f"{OLD}:d15", f"{ND}:d13", "NĐ100-2019 phạt xe tải cũ → NĐ168-2024 mới")

# Cross: old decree → law
BATCH_3 += make_pair(f"{OLD}:d6", f"{QH}:d11", "NĐ100-2019 phạt ô tô → tín hiệu đèn GT")
BATCH_3 += make_pair(f"{OLD}:d6", f"{QH}:d9", "NĐ100-2019 phạt ô tô → điều kiện lái xe")
BATCH_3 += make_pair(f"{OLD}:d56", f"{QH}:d13", "NĐ100-2019 quy tắc GT → quy tắc GT đường bộ")

# ============================================================
# BATCH 4: Enterprise intra-doc REFERENCES
# Target: d31(6), d30(6), d45(5), d28(5), d29(4), d206(4), d203(3)
# ============================================================
BATCH_4 = []

LDN = "59-2020-QH14"
ND168 = "168-2025-ND"
ND01 = "01-2021-ND"

# d29 ↔ d30 ↔ d31 cluster (ĐKKD)
BATCH_4 += make_pair(f"{LDN}:d29", f"{LDN}:d30", "Nội dung GCNĐKDN → ĐKKD thành lập DN")
BATCH_4 += make_pair(f"{LDN}:d30", f"{LDN}:d31", "ĐKKD thành lập → thay đổi nội dung ĐKKD")
BATCH_4 += make_pair(f"{LDN}:d29", f"{LDN}:d31", "Nội dung GCNĐKDN → thay đổi ĐKKD")

# d74 → d30, d31 (cổ đông sáng lập → ĐKKD)
BATCH_4 += make_pair(f"{LDN}:d74", f"{LDN}:d30", "Cổ đông sáng lập → ĐKKD thành lập DN")
BATCH_4 += make_pair(f"{LDN}:d74", f"{LDN}:d31", "Cổ đông sáng lập → thay đổi ĐKKD")

# d46/d47 → d45 (TNHH 2TV → chi nhánh)
BATCH_4 += make_pair(f"{LDN}:d46", f"{LDN}:d45", "Cơ cấu tổ chức TNHH 2TV → chi nhánh/VPĐD")
BATCH_4 += make_pair(f"{LDN}:d47", f"{LDN}:d45", "Hội đồng thành viên → chi nhánh/VPĐD")

# d202 ↔ d203 (chuyển đổi loại hình)
BATCH_4 += make_pair(f"{LDN}:d202", f"{LDN}:d203", "Chuyển đổi TNHH→CP → CP→TNHH")

# Cross-doc decree → d28 (cấp GCNĐKDN)
BATCH_4 += make_pair(f"{ND168}:d24", f"{LDN}:d28", "NĐ168-2025 hồ sơ ĐKKD → cấp GCNĐKDN")
BATCH_4 += make_pair(f"{ND168}:d33", f"{LDN}:d28", "NĐ168-2025 thay đổi ĐKKD → cấp GCNĐKDN")
BATCH_4 += make_pair(f"{ND168}:d49", f"{LDN}:d30", "NĐ168-2025 ĐKKD online → ĐKKD thành lập")
BATCH_4 += make_pair(f"{ND01}:d56", f"{LDN}:d30", "NĐ01-2021 hồ sơ TNHH → ĐKKD thành lập")
BATCH_4 += make_pair(f"{ND01}:d56", f"{LDN}:d31", "NĐ01-2021 hồ sơ TNHH → thay đổi ĐKKD")

# d206 cluster
BATCH_4 += make_pair(f"{ND168}:d60", f"{LDN}:d45", "NĐ168-2025 tạm ngừng → chi nhánh/VPĐD")
BATCH_4 += make_pair(f"{ND01}:d66", f"{LDN}:d206", "NĐ01-2021 tạm ngừng KD → tạm ngừng (LDN)")

# d17 → d28 (quyền DN → cấp GCNĐKDN)
BATCH_4 += make_pair(f"{LDN}:d17", f"{LDN}:d28", "Quyền của doanh nghiệp → cấp GCNĐKDN")

BATCHES = {1: BATCH_1, 2: BATCH_2, 3: BATCH_3, 4: BATCH_4}


def add_batch(batch_num: int, dry_run: bool = False):
    if batch_num not in BATCHES:
        print(f"Invalid batch number: {batch_num}. Valid: {list(BATCHES.keys())}")
        return

    with open(KG_FILE) as f:
        kg = json.load(f)

    relations = kg["relations"]
    existing = {(r["source"], r["target"], r.get("type", "")) for r in relations}

    new_rels = BATCHES[batch_num]
    added = []
    skipped = []

    for rel in new_rels:
        key = (rel["source"], rel["target"], rel["type"])
        if key in existing:
            skipped.append(rel)
        else:
            added.append(rel)
            existing.add(key)

    print(f"\n=== Batch {batch_num} ===")
    print(f"Total relations in batch: {len(new_rels)}")
    print(f"New (to add): {len(added)}")
    print(f"Skipped (already exist): {len(skipped)}")

    if skipped:
        print(f"\nSkipped:")
        for r in skipped:
            print(f"  {r['source']} -> {r['target']} (exists)")

    print(f"\nAdding:")
    for i, r in enumerate(added, 1):
        print(f"  {i}. {r['source']} -> {r['target']}: {r['description']}")

    if dry_run:
        print(f"\n[DRY RUN] Would add {len(added)} relations to {KG_FILE}")
        return

    relations.extend(added)
    kg["relations"] = relations

    # Update stats
    if "stats" in kg:
        kg["stats"]["total_relations"] = len(relations)
        ref_count = sum(1 for r in relations if r.get("type") == "REFERENCES")
        kg["stats"]["references_count"] = ref_count

    with open(KG_FILE, "w") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Added {len(added)} relations. Total now: {len(relations)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="Batch number (1-4) or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if args.batch == "all":
        for b in sorted(BATCHES.keys()):
            add_batch(b, args.dry_run)
    else:
        add_batch(int(args.batch), args.dry_run)


if __name__ == "__main__":
    main()
