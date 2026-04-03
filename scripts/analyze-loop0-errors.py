#!/usr/bin/env python3
"""Analyze Loop 0 (document selection) errors in hard-400 benchmark results.

Uses tree_articles domain as ground truth for Loop 0 decision (tree traversal
only runs on the selected domain). If tree articles are from wrong domain,
Loop 0 made the wrong call.

Categorizes MISS queries into:
- loop0_wrong_domain: tree traversal ran on wrong domain (Loop 0 error)
- tree_traversal_error: right domain, wrong articles (tree/KG failure)
- Near-miss: expected articles in positions 11-20 vs completely absent
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# --- Domain definitions ---
ENTERPRISE_DOCS = {
    "59-2020-QH14", "01-2021-ND", "16-2023-ND", "168-2025-ND",
    "23-2022-ND", "248-2025-ND", "44-2025-ND", "47-2021-ND",
    "89-2024-ND", "65-2022-ND",
}
TRAFFIC_DOCS = {"36-2024-QH15", "168-2024-ND-CP", "100-2019-ND-CP"}


def get_domain(article: str) -> str:
    """Return 'enterprise', 'traffic', or 'unknown' for an article ID."""
    doc = article.split(":")[0]
    if doc in ENTERPRISE_DOCS:
        return "enterprise"
    elif doc in TRAFFIC_DOCS:
        return "traffic"
    return "unknown"


def domain_distribution(articles: list[str]) -> dict[str, int]:
    """Count how many articles belong to each domain."""
    counts = Counter(get_domain(a) for a in articles)
    return dict(counts)


def majority_domain(articles: list[str]) -> str:
    """Return the domain that has the most articles."""
    dist = domain_distribution(articles)
    if not dist:
        return "unknown"
    return max(dist, key=dist.get)


def tree_went_to_wrong_domain(expected_articles, tree_articles):
    """Check if tree traversal ran on the wrong domain.

    Returns (is_wrong, expected_domain, tree_domain).
    """
    if not tree_articles:
        return False, "unknown", "unknown"

    expected_domain = get_domain(expected_articles[0])
    tree_docs = set(a.split(":")[0] for a in tree_articles)

    if expected_domain == "traffic":
        tree_in_expected = tree_docs.intersection(TRAFFIC_DOCS)
    elif expected_domain == "enterprise":
        tree_in_expected = tree_docs.intersection(ENTERPRISE_DOCS)
    else:
        return False, expected_domain, "unknown"

    tree_domain = majority_domain(tree_articles)
    return len(tree_in_expected) == 0, expected_domain, tree_domain


def main():
    results_path = Path("results/hard400_round3_batch5_run1.json")
    output_path = Path("results/hard400_loop0_analysis.json")

    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)

    # =========================================================
    # SECTION 1: Loop 0 accuracy across ALL queries
    # =========================================================
    print("=" * 70)
    print("LOOP 0 ACCURACY ACROSS ALL 400 QUERIES")
    print("=" * 70)

    loop0_correct = 0
    loop0_wrong = 0
    loop0_wrong_but_hit = 0
    loop0_wrong_and_miss = 0
    loop0_no_tree = 0
    loop0_wrong_details = []

    for r in results:
        tree = r.get("full_tree_articles", [])
        if not tree:
            loop0_no_tree += 1
            continue

        is_wrong, exp_domain, tree_domain = tree_went_to_wrong_domain(
            r["expected_articles"], tree
        )

        if is_wrong:
            loop0_wrong += 1
            top10 = set(r["full_retrieved"][:10])
            expected = set(r["expected_articles"])
            is_hit = bool(expected.intersection(top10))
            if is_hit:
                loop0_wrong_but_hit += 1
            else:
                loop0_wrong_and_miss += 1
            loop0_wrong_details.append({
                "stt": r["stt"],
                "question": r["question"][:120],
                "category": r["category"],
                "expected_articles": r["expected_articles"],
                "expected_domain": exp_domain,
                "tree_domain": tree_domain,
                "tree_articles": tree[:5],
                "is_hit": is_hit,
            })
        else:
            loop0_correct += 1

    total_with_tree = loop0_correct + loop0_wrong
    print(f"  Queries with tree articles: {total_with_tree}")
    print(f"  No tree articles:           {loop0_no_tree}")
    print(f"  Loop 0 correct domain:      {loop0_correct}/{total_with_tree}"
          f" ({loop0_correct/total_with_tree*100:.1f}%)")
    print(f"  Loop 0 wrong domain:        {loop0_wrong}/{total_with_tree}"
          f" ({loop0_wrong/total_with_tree*100:.1f}%)")
    print(f"    - Still hit (KG rescued): {loop0_wrong_but_hit}")
    print(f"    - Missed:                 {loop0_wrong_and_miss}")
    print()

    # Direction of confusion
    direction_counts = Counter()
    for d in loop0_wrong_details:
        direction_counts[(d["expected_domain"], d["tree_domain"])] += 1
    print("  Domain confusion direction:")
    for (exp, tree_d), count in direction_counts.most_common():
        print(f"    Expected {exp:12s} -> Tree went to {tree_d:12s}: {count}")
    print()

    # =========================================================
    # SECTION 2: Miss classification
    # =========================================================
    misses = []
    hits = []
    for r in results:
        expected = set(r["expected_articles"])
        top10 = r["full_retrieved"][:10]
        if not expected.intersection(set(top10)):
            misses.append(r)
        else:
            hits.append(r)

    print("=" * 70)
    print("MISS OVERVIEW")
    print("=" * 70)
    print(f"  Total queries: {total}")
    print(f"  Hits @10:      {len(hits)} ({len(hits)/total*100:.1f}%)")
    print(f"  Misses @10:    {len(misses)} ({len(misses)/total*100:.1f}%)")
    print()

    # Classify each miss using tree-based detection
    classified = []
    type_counts = Counter()
    near_miss_counts = {"in_11_20": 0, "in_21_30": 0, "absent": 0}

    for r in misses:
        tree = r.get("full_tree_articles", [])
        top10 = r["full_retrieved"][:10]
        top20 = r["full_retrieved"][:20]
        top30 = r["full_retrieved"][:30]
        expected_set = set(r["expected_articles"])
        expected_domain = get_domain(r["expected_articles"][0])

        # Check Loop 0 via tree articles
        is_wrong_domain, _, tree_domain = tree_went_to_wrong_domain(
            r["expected_articles"], tree
        )

        if is_wrong_domain:
            error_type = "loop0_wrong_domain"
        else:
            error_type = "tree_traversal_error"

        # Near-miss analysis
        positions_11_20 = set(top20[10:]) if len(top20) > 10 else set()
        positions_21_30 = set(top30[20:]) if len(top30) > 20 else set()
        near_miss_11_20 = sorted(expected_set.intersection(positions_11_20))
        near_miss_21_30 = sorted(expected_set.intersection(positions_21_30))
        completely_absent = sorted(expected_set - set(top30))

        if near_miss_11_20:
            near_miss_status = "near_miss_11_20"
            near_miss_counts["in_11_20"] += 1
        elif near_miss_21_30:
            near_miss_status = "near_miss_21_30"
            near_miss_counts["in_21_30"] += 1
        else:
            near_miss_status = "absent_from_top30"
            near_miss_counts["absent"] += 1

        type_counts[error_type] += 1

        entry = {
            "stt": r["stt"],
            "question": r["question"][:150],
            "category": r["category"],
            "expected_articles": r["expected_articles"],
            "expected_domain": expected_domain,
            "error_type": error_type,
            "tree_domain": tree_domain,
            "near_miss_status": near_miss_status,
            "top10_domain_dist": domain_distribution(top10),
            "top10_retrieved": top10,
            "tree_articles": tree[:10],
            "kg_articles": r.get("full_kg_articles", [])[:15],
            "near_miss_11_20": near_miss_11_20,
            "near_miss_21_30": near_miss_21_30,
            "completely_absent": completely_absent,
        }
        classified.append(entry)

    print("=" * 70)
    print("MISS ERROR TYPE BREAKDOWN (tree-based classification)")
    print("=" * 70)
    for etype, count in type_counts.most_common():
        pct = count / len(misses) * 100
        print(f"  {etype:30s}: {count:3d} ({pct:.1f}%)")
    print()

    print("=" * 70)
    print("NEAR-MISS ANALYSIS (where do expected articles end up?)")
    print("=" * 70)
    print(f"  In positions 11-20 (near-miss): {near_miss_counts['in_11_20']:3d}")
    print(f"  In positions 21-30:             {near_miss_counts['in_21_30']:3d}")
    print(f"  Absent from top 30:             {near_miss_counts['absent']:3d}")
    print()

    # =========================================================
    # SECTION 3: Cross-tabulations
    # =========================================================
    print("=" * 70)
    print("MISSES BY EXPECTED DOMAIN")
    print("=" * 70)
    domain_miss_counts = Counter()
    domain_total_counts = Counter()
    for r in results:
        exp_domain = get_domain(r["expected_articles"][0])
        domain_total_counts[exp_domain] += 1
        top10 = r["full_retrieved"][:10]
        if not set(r["expected_articles"]).intersection(set(top10)):
            domain_miss_counts[exp_domain] += 1

    for domain in ["enterprise", "traffic"]:
        total_d = domain_total_counts.get(domain, 0)
        miss_d = domain_miss_counts.get(domain, 0)
        hit_d = total_d - miss_d
        rate = hit_d / total_d * 100 if total_d else 0
        print(f"  {domain:12s}: {hit_d}/{total_d} hit ({rate:.1f}%), {miss_d} misses")
    print()

    print("=" * 70)
    print("ERROR TYPE x EXPECTED DOMAIN")
    print("=" * 70)
    cross_tab = defaultdict(Counter)
    for entry in classified:
        cross_tab[entry["error_type"]][entry["expected_domain"]] += 1
    for etype in sorted(cross_tab.keys()):
        print(f"  {etype}:")
        for domain, count in cross_tab[etype].most_common():
            print(f"    {domain:12s}: {count}")
    print()

    print("=" * 70)
    print("ERROR TYPE x NEAR-MISS STATUS")
    print("=" * 70)
    cross_nm = defaultdict(Counter)
    for entry in classified:
        cross_nm[entry["error_type"]][entry["near_miss_status"]] += 1
    for etype in sorted(cross_nm.keys()):
        print(f"  {etype}:")
        for status, count in cross_nm[etype].most_common():
            print(f"    {status:25s}: {count}")
    print()

    # =========================================================
    # SECTION 4: Loop 0 wrong-domain miss details
    # =========================================================
    loop0_misses = [
        e for e in classified if e["error_type"] == "loop0_wrong_domain"
    ]
    print("=" * 70)
    print(f"LOOP 0 WRONG-DOMAIN MISSES ({len(loop0_misses)} total)")
    print("=" * 70)
    for e in loop0_misses:
        print(f"\n  Q#{e['stt']}: {e['question']}")
        print(f"    Category: {e['category']}")
        print(f"    Expected: {e['expected_articles']}"
              f" (domain: {e['expected_domain']})")
        print(f"    Tree went to: {e['tree_domain']}"
              f" | Tree: {e['tree_articles'][:5]}")
        print(f"    Top10 domains: {e['top10_domain_dist']}")
        if e["near_miss_11_20"]:
            print(f"    Near-miss 11-20: {e['near_miss_11_20']}")
        if e["near_miss_21_30"]:
            print(f"    Near-miss 21-30: {e['near_miss_21_30']}")
        if e["completely_absent"]:
            print(f"    ABSENT from top 30: {e['completely_absent']}")
    print()

    # =========================================================
    # SECTION 5: Loop 0 wrong-domain hits (KG rescued)
    # =========================================================
    loop0_rescued = [
        d for d in loop0_wrong_details if d["is_hit"]
    ]
    print("=" * 70)
    print(f"LOOP 0 WRONG-DOMAIN BUT KG RESCUED ({len(loop0_rescued)} total)")
    print("=" * 70)
    for d in loop0_rescued:
        print(f"  Q#{d['stt']}: {d['question']}")
        print(f"    Category: {d['category']}")
        print(f"    Expected: {d['expected_articles']}"
              f" (domain: {d['expected_domain']})")
        print(f"    Tree went to: {d['tree_domain']}"
              f" | Tree: {d['tree_articles'][:3]}")
    print()

    # =========================================================
    # SECTION 6: Tree traversal error examples
    # =========================================================
    tree_errors = [
        e for e in classified if e["error_type"] == "tree_traversal_error"
    ]
    print("=" * 70)
    print(f"TREE TRAVERSAL ERROR EXAMPLES ({len(tree_errors)} total,"
          f" showing first 15)")
    print("=" * 70)
    for e in tree_errors[:15]:
        print(f"\n  Q#{e['stt']}: {e['question']}")
        print(f"    Category: {e['category']}")
        print(f"    Expected: {e['expected_articles']}"
              f" (domain: {e['expected_domain']})")
        print(f"    Tree: {e['tree_articles'][:5]}")
        if e["near_miss_11_20"]:
            print(f"    Near-miss 11-20: {e['near_miss_11_20']}")
        if e["completely_absent"]:
            print(f"    ABSENT from top 30: {e['completely_absent']}")
    if len(tree_errors) > 15:
        print(f"\n  ... and {len(tree_errors) - 15} more tree traversal errors")
    print()

    # =========================================================
    # SECTION 7: Top miss categories
    # =========================================================
    print("=" * 70)
    print("TOP MISS CATEGORIES")
    print("=" * 70)
    cat_counts = Counter()
    cat_loop0 = Counter()
    for e in classified:
        cat_counts[e["category"]] += 1
        if e["error_type"] == "loop0_wrong_domain":
            cat_loop0[e["category"]] += 1
    for cat, count in cat_counts.most_common(15):
        l0 = cat_loop0.get(cat, 0)
        print(f"  {count:2d} misses ({l0:2d} loop0): {cat}")
    print()

    # =========================================================
    # SECTION 8: Tree traversal sub-analysis
    # =========================================================
    print("=" * 70)
    print("TREE TRAVERSAL ERROR SUB-ANALYSIS")
    print("=" * 70)

    # For tree errors, check if tree articles are from right doc
    # but wrong articles within that doc
    same_doc_wrong_article = 0
    different_doc_same_domain = 0
    for e in tree_errors:
        expected_docs = set(a.split(":")[0] for a in e["expected_articles"])
        tree_docs = set(a.split(":")[0] for a in e["tree_articles"])
        if expected_docs.intersection(tree_docs):
            same_doc_wrong_article += 1
        else:
            different_doc_same_domain += 1

    print(f"  Right doc, wrong articles:       {same_doc_wrong_article}")
    print(f"  Different doc (same domain):     {different_doc_same_domain}")
    print()

    # Check how many tree errors have expected article in KG articles
    kg_has_expected = 0
    kg_missing_expected = 0
    for e in tree_errors:
        expected_set = set(e["expected_articles"])
        # Get full KG articles from original data
        orig = next(
            r for r in results if r["stt"] == e["stt"]
        )
        kg_set = set(orig.get("full_kg_articles", []))
        if expected_set.intersection(kg_set):
            kg_has_expected += 1
        else:
            kg_missing_expected += 1

    print(f"  Expected article found in KG expansion: {kg_has_expected}")
    print(f"  Expected article NOT in KG expansion:   {kg_missing_expected}")
    print(f"  (KG found it but RRF ranked it too low: {kg_has_expected - len([e for e in tree_errors if e['near_miss_status'] != 'absent_from_top30'])})")
    print()

    # =========================================================
    # Save JSON
    # =========================================================
    output = {
        "summary": {
            "total_queries": total,
            "total_hits": len(hits),
            "total_misses": len(misses),
            "hit_at_10_rate": round(len(hits) / total * 100, 1),
            "loop0_accuracy": {
                "total_with_tree": total_with_tree,
                "correct_domain": loop0_correct,
                "wrong_domain": loop0_wrong,
                "accuracy_pct": round(
                    loop0_correct / total_with_tree * 100, 1
                ),
                "wrong_but_hit": loop0_wrong_but_hit,
                "wrong_and_miss": loop0_wrong_and_miss,
            },
            "miss_error_types": dict(type_counts),
            "near_miss_counts": near_miss_counts,
            "misses_by_domain": {
                domain: {
                    "total": domain_total_counts.get(domain, 0),
                    "misses": domain_miss_counts.get(domain, 0),
                    "hit_rate": round(
                        (domain_total_counts[domain] - domain_miss_counts[domain])
                        / domain_total_counts[domain] * 100, 1,
                    ) if domain_total_counts.get(domain, 0) else 0,
                }
                for domain in ["enterprise", "traffic"]
            },
        },
        "loop0_wrong_domain_all": loop0_wrong_details,
        "classified_misses": classified,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
