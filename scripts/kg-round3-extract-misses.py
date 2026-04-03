"""Extract and analyze MISS queries from hard-400 results for KG Round 3 optimization."""

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_FILE = "results/hard400_combined.json"
BENCHMARK_FILE = "data/benchmark/hard-400-qa-benchmark.csv"
OUTPUT_DIR = Path("plans/260331-1425-kg-optimization-round3")
MANIFEST_FILE = OUTPUT_DIR / "miss-manifest.json"
SUMMARY_FILE = OUTPUT_DIR / "miss-summary.md"

TRAFFIC_DOCS = ["168-2024-ND-CP", "36-2024-QH15", "100-2019-ND-CP"]


def is_traffic_article(article_id: str) -> bool:
    return any(td in article_id for td in TRAFFIC_DOCS)


def classify_domain(expected_articles: list[str]) -> str:
    if any(is_traffic_article(a) for a in expected_articles):
        return "traffic"
    return "enterprise"


def find_best_rank(expected: list[str], retrieved: list[str]) -> int:
    """Find best rank (1-indexed) of any expected article in retrieved. -1 if not found."""
    for i, art in enumerate(retrieved):
        if art in expected:
            return i + 1
    return -1


def extract_misses():
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    results = data["results"]

    misses = []
    for r in results:
        if r.get("skipped") or r.get("full_hit@10") == 1:
            continue

        expected = r.get("expected_articles", [])
        retrieved = r.get("full_retrieved", [])
        tree = r.get("full_tree_articles", [])
        kg = r.get("full_kg_articles", [])

        hit20 = r.get("full_hit@20", 0)
        hit30 = r.get("full_hit@30", 0)

        if hit20:
            severity = "near_miss_20"
        elif hit30:
            severity = "near_miss_30"
        else:
            severity = "far_miss"

        best_rank = find_best_rank(expected, retrieved)

        misses.append({
            "stt": r["stt"],
            "category": r.get("category", ""),
            "question": r.get("question", "")[:150],
            "domain": classify_domain(expected),
            "severity": severity,
            "best_rank": best_rank,
            "expected_articles": expected,
            "num_expected": len(expected),
            "retrieved_top10": retrieved[:10],
            "retrieved_full": retrieved,
            "tree_articles": tree,
            "kg_articles": kg,
            "hit@20": hit20,
            "hit@30": hit30,
        })

    misses.sort(key=lambda m: (m["domain"], m["severity"], m["stt"]))

    with open(MANIFEST_FILE, "w") as f:
        json.dump({"total_miss": len(misses), "misses": misses}, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(misses)} MISS queries → {MANIFEST_FILE}")
    return misses


def generate_summary(misses: list[dict]):
    lines = []
    lines.append("# MISS Query Analysis — Hard-400 Round 3\n")
    lines.append(f"**Total MISS:** {len(misses)} / 400 queries")
    lines.append(f"**Target:** reduce to ≤80 (need +13 flips minimum)\n")

    # --- Severity breakdown ---
    sev_counts = Counter(m["severity"] for m in misses)
    lines.append("## 1. Severity Breakdown\n")
    lines.append("| Severity | Count | Description |")
    lines.append("|----------|-------|-------------|")
    lines.append(f"| near_miss_20 | {sev_counts.get('near_miss_20', 0)} | Hit@20 but not @10 — easiest to flip |")
    lines.append(f"| near_miss_30 | {sev_counts.get('near_miss_30', 0)} | Hit@30 but not @20 — moderate effort |")
    lines.append(f"| far_miss | {sev_counts.get('far_miss', 0)} | Not even Hit@30 — need new KG paths |")
    lines.append("")

    # --- Domain breakdown ---
    domain_counts = Counter(m["domain"] for m in misses)
    lines.append("## 2. Domain Breakdown\n")
    lines.append(f"- **Enterprise:** {domain_counts.get('enterprise', 0)} misses")
    lines.append(f"- **Traffic:** {domain_counts.get('traffic', 0)} misses\n")

    # Domain x severity
    lines.append("| Domain | near_miss_20 | near_miss_30 | far_miss | Total |")
    lines.append("|--------|-------------|-------------|----------|-------|")
    for domain in ["enterprise", "traffic"]:
        dm = [m for m in misses if m["domain"] == domain]
        sc = Counter(m["severity"] for m in dm)
        lines.append(f"| {domain} | {sc.get('near_miss_20',0)} | {sc.get('near_miss_30',0)} | {sc.get('far_miss',0)} | {len(dm)} |")
    lines.append("")

    # --- Category breakdown ---
    cat_counts = Counter(m["category"] for m in misses)
    lines.append("## 3. Category Breakdown (sorted by count)\n")
    lines.append("| Category | Count | Domain |")
    lines.append("|----------|-------|--------|")
    for cat, count in cat_counts.most_common():
        dm = classify_domain([m["expected_articles"][0] for m in misses if m["category"] == cat and m["expected_articles"]])
        lines.append(f"| {cat[:80]} | {count} | {dm} |")
    lines.append("")

    # --- Most-wanted articles ---
    article_miss_count = Counter()
    article_miss_stts = defaultdict(list)
    for m in misses:
        for art in m["expected_articles"]:
            article_miss_count[art] += 1
            article_miss_stts[art].append(m["stt"])

    lines.append("## 4. Most-Wanted Articles (expected but not in top-10)\n")
    lines.append("| Article | MISS Count | Domain | STTs |")
    lines.append("|---------|-----------|--------|------|")
    for art, count in article_miss_count.most_common(25):
        dm = "traffic" if is_traffic_article(art) else "enterprise"
        stts = ",".join(str(s) for s in sorted(article_miss_stts[art]))
        lines.append(f"| {art} | {count} | {dm} | {stts} |")
    lines.append("")

    # --- Retrieval pattern: what's retrieved INSTEAD for top missing articles ---
    lines.append("## 5. Retrieval Patterns (what's retrieved instead)\n")
    for art, count in article_miss_count.most_common(15):
        if count < 3:
            break
        lines.append(f"### {art} (expected in {count} queries)\n")
        relevant_misses = [m for m in misses if art in m["expected_articles"]]

        # Count what appears in top-10 of these queries
        instead_counter = Counter()
        for m in relevant_misses:
            for ret_art in m["retrieved_top10"]:
                instead_counter[ret_art] += 1

        lines.append("| Retrieved Instead | Count (/{}) | Doc |".format(count))
        lines.append("|-------------------|-------|-----|")
        for ret_art, ret_count in instead_counter.most_common(8):
            dm = "traffic" if is_traffic_article(ret_art) else "enterprise"
            lines.append(f"| {ret_art} | {ret_count} | {dm} |")
        lines.append("")

    # --- Near-miss details (most actionable) ---
    near20 = [m for m in misses if m["severity"] == "near_miss_20"]
    lines.append("## 6. Near-Miss@20 Detail (easiest to flip)\n")
    lines.append("| STT | Domain | Expected | Best Rank | Retrieved Top-5 |")
    lines.append("|-----|--------|----------|-----------|-----------------|")
    for m in sorted(near20, key=lambda x: x["best_rank"] if x["best_rank"] > 0 else 999):
        exp = ", ".join(m["expected_articles"][:3])
        top5 = ", ".join(m["retrieved_top10"][:5])
        lines.append(f"| {m['stt']} | {m['domain']} | {exp} | {m['best_rank']} | {top5} |")
    lines.append("")

    # --- Far-miss summary ---
    far = [m for m in misses if m["severity"] == "far_miss"]
    lines.append("## 7. Far-Miss Summary (not in top-30)\n")
    lines.append(f"**Count:** {len(far)}\n")
    lines.append("| STT | Domain | Category | Expected |")
    lines.append("|-----|--------|----------|----------|")
    for m in far:
        exp = ", ".join(m["expected_articles"][:3])
        cat_short = m["category"][:60]
        lines.append(f"| {m['stt']} | {m['domain']} | {cat_short} | {exp} |")
    lines.append("")

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary written → {SUMMARY_FILE}")


if __name__ == "__main__":
    misses = extract_misses()
    generate_summary(misses)
