"""
Analyze tree traversal errors from hard-400 benchmark.

Focus on the 76 tree traversal errors (correct domain, wrong articles):
- What tree found vs expected
- Pattern analysis: default articles, per-document failures
- Cross-run consistency
- Tree vs KG contribution for hits
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_doc(article_id):
    """Extract document ID from article ID like '59-2020-QH14:d206' -> '59-2020-QH14'"""
    return article_id.split(":")[0] if ":" in article_id else article_id


def extract_article_num(article_id):
    """Extract article number like '59-2020-QH14:d206' -> 'd206'"""
    return article_id.split(":")[-1] if ":" in article_id else article_id


def main():
    base_dir = Path("results")

    # Load primary data
    loop0 = load_json(base_dir / "hard400_loop0_analysis.json")
    run1 = load_json(base_dir / "hard400_round3_batch5_run1.json")

    # Load all 3 runs for cross-run comparison
    runs = {}
    for i in [1, 2, 3]:
        p = base_dir / f"hard400_round3_batch5_run{i}.json"
        if p.exists():
            runs[i] = load_json(p)

    # Build run1 lookup by stt
    run1_by_stt = {str(r["stt"]): r for r in run1["results"]}

    # Get tree traversal errors (76 queries)
    all_misses = loop0["classified_misses"]
    tree_errors = [m for m in all_misses if m["error_type"] == "tree_traversal_error"]
    loop0_errors = [m for m in all_misses if m["error_type"] == "loop0_wrong_domain"]

    print("=" * 80)
    print("TREE TRAVERSAL ERROR ANALYSIS — hard-400 benchmark")
    print("=" * 80)
    print(f"\nTotal misses: {len(all_misses)}")
    print(f"  Loop 0 errors (wrong domain): {len(loop0_errors)}")
    print(f"  Tree traversal errors (right domain, wrong articles): {len(tree_errors)}")

    # ========================================================================
    # SECTION 1: What tree finds vs what's expected
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: Tree articles vs Expected articles")
    print("=" * 80)

    tree_article_counts = []
    same_doc_counts = 0
    diff_doc_counts = 0
    partially_same_doc = 0

    tree_articles_all = Counter()  # All tree articles across miss queries
    expected_articles_all = Counter()  # All expected articles that were missed
    expected_docs_all = Counter()

    tree_found_expected_in_tree = 0
    tree_found_expected_in_kg = 0

    details = []

    for m in tree_errors:
        stt = m["stt"]
        r = run1_by_stt.get(stt, {})
        tree_arts = r.get("full_tree_articles", m.get("tree_articles", []))
        expected = m["expected_articles"]

        tree_article_counts.append(len(tree_arts))

        # Count tree articles
        for art in tree_arts:
            tree_articles_all[art] += 1

        # Check if tree articles are from same doc as expected
        expected_docs = set(extract_doc(a) for a in expected)
        tree_docs = set(extract_doc(a) for a in tree_arts)

        for exp_art in expected:
            exp_doc = extract_doc(exp_art)
            expected_articles_all[exp_art] += 1
            expected_docs_all[exp_doc] += 1

        overlap = expected_docs & tree_docs
        if overlap == expected_docs:
            same_doc_counts += 1
        elif overlap:
            partially_same_doc += 1
        else:
            diff_doc_counts += 1

        details.append({
            "stt": stt,
            "question": m["question"][:80],
            "category": m["category"],
            "expected": expected,
            "expected_docs": list(expected_docs),
            "tree_articles": tree_arts,
            "tree_docs": list(tree_docs),
            "same_doc": overlap == expected_docs,
            "near_miss": m.get("near_miss_status", ""),
        })

    print(f"\nTree article count per miss query:")
    print(f"  Mean: {sum(tree_article_counts)/len(tree_article_counts):.1f}")
    print(f"  Min: {min(tree_article_counts)}, Max: {max(tree_article_counts)}")
    count_dist = Counter(tree_article_counts)
    for c in sorted(count_dist.keys()):
        print(f"  {c} articles: {count_dist[c]} queries")

    print(f"\nDocument matching (tree found right doc?):")
    print(f"  Same doc as expected: {same_doc_counts} ({same_doc_counts/len(tree_errors)*100:.1f}%)")
    print(f"  Partially same doc: {partially_same_doc} ({partially_same_doc/len(tree_errors)*100:.1f}%)")
    print(f"  Different doc: {diff_doc_counts} ({diff_doc_counts/len(tree_errors)*100:.1f}%)")

    # ========================================================================
    # SECTION 2: Most common tree articles (defaults?)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Most frequently returned tree articles (across all 76 miss queries)")
    print("=" * 80)

    print("\nTop 30 most common tree articles in miss queries:")
    for art, count in tree_articles_all.most_common(30):
        pct = count / len(tree_errors) * 100
        print(f"  {art:30s} — {count:3d} times ({pct:.1f}%)")

    # Group by document
    tree_doc_counts = Counter()
    for art, count in tree_articles_all.items():
        tree_doc_counts[extract_doc(art)] += count

    print("\nTree articles by document (total appearances):")
    for doc, count in tree_doc_counts.most_common():
        print(f"  {doc:25s} — {count:4d} article appearances")

    # ========================================================================
    # SECTION 3: Expected articles never found by tree
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: Expected articles that tree NEVER found")
    print("=" * 80)

    print(f"\nAll expected articles in tree-traversal-error misses:")
    for art, count in expected_articles_all.most_common():
        in_tree = "YES" if art in tree_articles_all else "NO"
        print(f"  {art:30s} — missed {count:2d} times | ever in tree: {in_tree}")

    # ========================================================================
    # SECTION 4: Misses grouped by expected document
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: Tree traversal failures by expected document")
    print("=" * 80)

    misses_by_doc = defaultdict(list)
    for d in details:
        for doc in d["expected_docs"]:
            misses_by_doc[doc].append(d)

    for doc in sorted(misses_by_doc.keys(), key=lambda x: -len(misses_by_doc[x])):
        queries = misses_by_doc[doc]
        same_doc_q = sum(1 for q in queries if q["same_doc"])
        print(f"\n  {doc}: {len(queries)} miss queries ({same_doc_q} had tree in same doc)")
        # Expected articles for this doc
        exp_arts = Counter()
        for q in queries:
            for a in q["expected"]:
                if extract_doc(a) == doc:
                    exp_arts[a] += 1
        for a, c in exp_arts.most_common(10):
            print(f"    Expected {a}: {c} times")

    # ========================================================================
    # SECTION 5: Misses grouped by category
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: Tree traversal failures by category")
    print("=" * 80)

    misses_by_cat = defaultdict(list)
    for d in details:
        misses_by_cat[d["category"]].append(d)

    for cat in sorted(misses_by_cat.keys(), key=lambda x: -len(misses_by_cat[x])):
        queries = misses_by_cat[cat]
        print(f"\n  [{len(queries):2d}] {cat}")
        # Show expected articles
        exp_arts = Counter()
        for q in queries:
            for a in q["expected"]:
                exp_arts[a] += 1
        for a, c in exp_arts.most_common(5):
            print(f"       Expected: {a} ({c}x)")

    # ========================================================================
    # SECTION 6: Cross-run consistency
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: Cross-run consistency of tree articles for miss queries")
    print("=" * 80)

    if len(runs) >= 2:
        # Build lookups for each run
        run_lookups = {}
        for run_id, run_data in runs.items():
            run_lookups[run_id] = {str(r["stt"]): r for r in run_data["results"]}

        consistent_count = 0
        partially_consistent = 0
        totally_different = 0
        tree_overlap_ratios = []

        miss_stts = [m["stt"] for m in tree_errors]

        for stt in miss_stts:
            # Get tree articles from each run
            tree_per_run = {}
            for run_id, lookup in run_lookups.items():
                r = lookup.get(stt, {})
                tree_per_run[run_id] = set(r.get("full_tree_articles", []))

            # Compare all pairs
            run_ids = sorted(tree_per_run.keys())
            if len(run_ids) < 2:
                continue

            all_same = True
            any_overlap = False
            pairwise_overlaps = []

            for i in range(len(run_ids)):
                for j in range(i + 1, len(run_ids)):
                    s1 = tree_per_run[run_ids[i]]
                    s2 = tree_per_run[run_ids[j]]
                    if s1 == s2:
                        pairwise_overlaps.append(1.0)
                    else:
                        all_same = False
                        union = s1 | s2
                        inter = s1 & s2
                        if inter:
                            any_overlap = True
                        if union:
                            pairwise_overlaps.append(len(inter) / len(union))
                        else:
                            pairwise_overlaps.append(0.0)

            avg_overlap = sum(pairwise_overlaps) / len(pairwise_overlaps) if pairwise_overlaps else 0
            tree_overlap_ratios.append(avg_overlap)

            if all_same:
                consistent_count += 1
            elif any_overlap:
                partially_consistent += 1
            else:
                totally_different += 1

        print(f"\nAcross {len(runs)} runs, for {len(miss_stts)} tree-error miss queries:")
        print(f"  Fully consistent (identical tree articles): {consistent_count} ({consistent_count/len(miss_stts)*100:.1f}%)")
        print(f"  Partially consistent (some overlap): {partially_consistent} ({partially_consistent/len(miss_stts)*100:.1f}%)")
        print(f"  Totally different (no overlap): {totally_different} ({totally_different/len(miss_stts)*100:.1f}%)")
        if tree_overlap_ratios:
            avg = sum(tree_overlap_ratios) / len(tree_overlap_ratios)
            print(f"  Mean Jaccard similarity: {avg:.3f}")

        # Also check: how many queries are hits in some runs but misses in others?
        hit_variation = Counter()
        for stt in miss_stts:
            hits = 0
            for run_id, lookup in run_lookups.items():
                r = lookup.get(stt, {})
                if r.get("full_hit@10", 0) == 1:
                    hits += 1
            hit_variation[hits] += 1

        print(f"\n  Hit@10 variation across {len(runs)} runs for these 76 queries:")
        for h in sorted(hit_variation.keys()):
            print(f"    Hit in {h}/{len(runs)} runs: {hit_variation[h]} queries")

        # Show examples of queries that hit in some runs but miss in run1
        if hit_variation.get(1, 0) > 0 or hit_variation.get(2, 0) > 0:
            print(f"\n  Examples of queries that hit in OTHER runs but miss in run1:")
            shown = 0
            for stt in miss_stts:
                r1 = run_lookups.get(1, {}).get(stt, {})
                if r1.get("full_hit@10", 0) == 0:
                    for run_id in [2, 3]:
                        r_other = run_lookups.get(run_id, {}).get(stt, {})
                        if r_other.get("full_hit@10", 0) == 1:
                            # Find what was different
                            tree1 = set(r1.get("full_tree_articles", []))
                            tree_other = set(r_other.get("full_tree_articles", []))
                            diff = tree_other - tree1
                            m = next((x for x in tree_errors if x["stt"] == stt), None)
                            if m and shown < 5:
                                exp = m["expected_articles"]
                                print(f"    STT {stt}: expected={exp}")
                                print(f"      Run1 tree: {sorted(tree1)}")
                                print(f"      Run{run_id} tree: {sorted(tree_other)}")
                                print(f"      New in run{run_id}: {sorted(diff)}")
                                shown += 1
                            break
    else:
        print("Only 1 run file found, skipping cross-run comparison")

    # ========================================================================
    # SECTION 7: Tree contribution to HIT queries
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: Tree traversal contribution to HIT queries")
    print("=" * 80)

    hits = [r for r in run1["results"] if r.get("full_hit@10", 0) == 1]

    tree_found_expected = 0
    kg_found_expected = 0
    both_found = 0
    neither_found = 0
    only_tree = 0
    only_kg = 0

    for r in hits:
        expected = r["expected_articles"]
        tree_arts = set(r.get("full_tree_articles", []))
        kg_arts = set(r.get("full_kg_articles", []))

        # Check if ANY expected article is in tree / kg
        exp_in_tree = any(a in tree_arts for a in expected)
        exp_in_kg = any(a in kg_arts for a in expected)

        if exp_in_tree and exp_in_kg:
            both_found += 1
        elif exp_in_tree:
            only_tree += 1
        elif exp_in_kg:
            only_kg += 1
        else:
            neither_found += 1

        if exp_in_tree:
            tree_found_expected += 1
        if exp_in_kg:
            kg_found_expected += 1

    total_hits = len(hits)
    print(f"\nFor {total_hits} HIT queries (expected article found in top-10):")
    print(f"  Tree found expected: {tree_found_expected} ({tree_found_expected/total_hits*100:.1f}%)")
    print(f"  KG found expected: {kg_found_expected} ({kg_found_expected/total_hits*100:.1f}%)")
    print(f"  Both found: {both_found} ({both_found/total_hits*100:.1f}%)")
    print(f"  Only tree: {only_tree} ({only_tree/total_hits*100:.1f}%)")
    print(f"  Only KG: {only_kg} ({only_kg/total_hits*100:.1f}%)")
    print(f"  Neither (via RRF merge): {neither_found} ({neither_found/total_hits*100:.1f}%)")

    # ========================================================================
    # SECTION 8: Deep dive — chapter-level analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: Chapter-level pattern analysis for tree errors")
    print("=" * 80)

    # For tree traversal errors, check if the tree consistently picks certain chapters
    # We can infer chapter from article number ranges (heuristic)

    # Count which article ranges tree picks (as proxy for chapters)
    tree_article_nums = Counter()
    expected_article_nums = Counter()

    for d in details:
        for art in d["tree_articles"]:
            doc = extract_doc(art)
            num = extract_article_num(art)
            tree_article_nums[(doc, num)] += 1
        for art in d["expected"]:
            doc = extract_doc(art)
            num = extract_article_num(art)
            expected_article_nums[(doc, num)] += 1

    # Check for "default" patterns: articles that appear in >20% of miss queries
    print("\nArticles appearing in >15% of tree-error miss queries (likely 'default' picks):")
    threshold = len(tree_errors) * 0.15
    default_articles = []
    for art, count in tree_articles_all.most_common():
        if count >= threshold:
            pct = count / len(tree_errors) * 100
            doc = extract_doc(art)
            default_articles.append(art)
            print(f"  {art:30s} — {count} times ({pct:.1f}%)")

    if default_articles:
        # How many miss queries are ONLY default articles?
        only_defaults = 0
        for d in details:
            if all(a in default_articles for a in d["tree_articles"]):
                only_defaults += 1
        print(f"\n  Queries where tree returned ONLY default articles: {only_defaults}/{len(tree_errors)}")

    # ========================================================================
    # SECTION 9: Enterprise vs Traffic breakdown
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: Enterprise vs Traffic tree errors")
    print("=" * 80)

    for domain in ["enterprise", "traffic"]:
        domain_errors = [d for d in details if (
            ("59-2020" in d["expected_docs"][0] or "01-2021" in d["expected_docs"][0] or
             "23-2022" in d["expected_docs"][0] or "168-2025" in d["expected_docs"][0])
            if domain == "enterprise" else
            ("168-2024" in d["expected_docs"][0] or "36-2024" in d["expected_docs"][0] or
             "100-2019" in d["expected_docs"][0])
        )]

        if not domain_errors:
            continue

        print(f"\n{domain.upper()} ({len(domain_errors)} tree errors):")

        # Near miss distribution
        near_miss_dist = Counter()
        for d in domain_errors:
            m = next((x for x in tree_errors if x["stt"] == d["stt"]), None)
            if m:
                near_miss_dist[m.get("near_miss_status", "unknown")] += 1

        for status, count in near_miss_dist.most_common():
            print(f"  {status}: {count}")

        # Same doc analysis
        same = sum(1 for d in domain_errors if d["same_doc"])
        print(f"  Tree in same doc as expected: {same}/{len(domain_errors)}")

    # ========================================================================
    # SECTION 10: Specific examples for debugging
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 10: Example tree traversal errors (for debugging)")
    print("=" * 80)

    # Show 10 representative examples
    # Prioritize: near misses (11-20), then completely absent
    near_miss_examples = [d for d in details if any(
        m.get("near_miss_status") == "near_miss_11_20"
        for m in tree_errors if m["stt"] == d["stt"]
    )]
    absent_examples = [d for d in details if any(
        m.get("near_miss_status") == "absent"
        for m in tree_errors if m["stt"] == d["stt"]
    )]

    print(f"\n--- Near-miss examples (expected at rank 11-20, salvageable) ---")
    for d in near_miss_examples[:5]:
        m = next((x for x in tree_errors if x["stt"] == d["stt"]), None)
        print(f"\n  STT {d['stt']}: {d['question']}")
        print(f"  Category: {d['category']}")
        print(f"  Expected: {d['expected']}")
        print(f"  Tree found: {d['tree_articles']}")
        if m:
            print(f"  Near miss at: {m.get('near_miss_11_20', [])}")
        print(f"  Same doc: {d['same_doc']}")

    print(f"\n--- Absent examples (expected article not in top-30 at all) ---")
    for d in absent_examples[:5]:
        m = next((x for x in tree_errors if x["stt"] == d["stt"]), None)
        print(f"\n  STT {d['stt']}: {d['question']}")
        print(f"  Category: {d['category']}")
        print(f"  Expected: {d['expected']}")
        print(f"  Tree found: {d['tree_articles']}")
        print(f"  Same doc: {d['same_doc']}")

    # ========================================================================
    # SECTION 11: Actionable recommendations
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 11: Actionable analysis summary")
    print("=" * 80)

    # Count queries where tree found right doc but wrong chapter/article
    right_doc_wrong_article = same_doc_counts + partially_same_doc
    print(f"\n1. Right doc, wrong chapter/article: {right_doc_wrong_article}/{len(tree_errors)} ({right_doc_wrong_article/len(tree_errors)*100:.1f}%)")
    print(f"   → These are Loop 1 (chapter) or Loop 2 (article) errors")

    print(f"\n2. Wrong doc entirely: {diff_doc_counts}/{len(tree_errors)} ({diff_doc_counts/len(tree_errors)*100:.1f}%)")
    print(f"   → These may be Loop 0 selecting correct domain but wrong specific doc")

    # Compute how many tree articles are from penalty/noise articles
    noise_articles = {
        "59-2020-QH14": {"d1", "d2", "d3", "d14", "d15", "d18", "d19", "d20", "d23", "d33",
                          "d40", "d48", "d61", "d71", "d83", "d85", "d184", "d197", "d204", "d216"},
        "168-2024-ND-CP": {"d1", "d2", "d8", "d16", "d27", "d42"},
        "36-2024-QH15": {"d1"},
        "100-2019-ND-CP": {"d1", "d2", "d8", "d15", "d63", "d76"},
    }

    noise_in_tree = 0
    total_tree_articles = sum(tree_articles_all.values())
    for art, count in tree_articles_all.items():
        doc = extract_doc(art)
        num = extract_article_num(art)
        if doc in noise_articles and num in noise_articles[doc]:
            noise_in_tree += count

    print(f"\n3. Noise/penalty articles in tree output: {noise_in_tree}/{total_tree_articles} ({noise_in_tree/total_tree_articles*100:.1f}%)")
    print(f"   → Tree is actively picking known-bad articles")

    # ========================================================================
    # Generate report
    # ========================================================================
    report = generate_report(
        tree_errors, details, tree_articles_all, expected_articles_all,
        misses_by_doc, misses_by_cat, tree_article_counts,
        same_doc_counts, partially_same_doc, diff_doc_counts,
        hits, tree_found_expected, kg_found_expected,
        both_found, only_tree, only_kg, neither_found,
        runs, run_lookups if len(runs) >= 2 else {},
        consistent_count if len(runs) >= 2 else 0,
        partially_consistent if len(runs) >= 2 else 0,
        totally_different if len(runs) >= 2 else 0,
        tree_overlap_ratios if len(runs) >= 2 else [],
        noise_in_tree, total_tree_articles,
        default_articles, near_miss_examples, absent_examples,
        hit_variation if len(runs) >= 2 else {},
    )

    report_path = Path("plans/reports/analysis-260401-1619-tree-traversal-errors.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def generate_report(
    tree_errors, details, tree_articles_all, expected_articles_all,
    misses_by_doc, misses_by_cat, tree_article_counts,
    same_doc_counts, partially_same_doc, diff_doc_counts,
    hits, tree_found_expected, kg_found_expected,
    both_found, only_tree, only_kg, neither_found,
    runs, run_lookups, consistent_count, partially_consistent, totally_different,
    tree_overlap_ratios, noise_in_tree, total_tree_articles,
    default_articles, near_miss_examples, absent_examples,
    hit_variation,
):
    total_hits = len(hits)
    n = len(tree_errors)
    avg_overlap = sum(tree_overlap_ratios) / len(tree_overlap_ratios) if tree_overlap_ratios else 0

    lines = []
    lines.append("# Tree Traversal Error Analysis — hard-400 Benchmark")
    lines.append(f"**Date:** 2026-04-01 | **Source:** `results/hard400_round3_batch5_run1.json`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **85 total misses**, 76 are tree traversal errors (correct domain, wrong articles), 9 are Loop 0 errors")
    lines.append(f"- Tree picks right doc: {same_doc_counts + partially_same_doc}/{n} ({(same_doc_counts+partially_same_doc)/n*100:.0f}%) — problem is chapter/article selection, not doc selection")
    lines.append(f"- Tree avg articles per miss query: {sum(tree_article_counts)/n:.1f}")
    lines.append(f"- Noise/penalty articles in tree output: {noise_in_tree}/{total_tree_articles} ({noise_in_tree/total_tree_articles*100:.0f}%)")
    lines.append(f"- Cross-run consistency (Jaccard): {avg_overlap:.3f} — {'high variance (LLM random)' if avg_overlap < 0.5 else 'moderate consistency' if avg_overlap < 0.8 else 'highly consistent'}")
    lines.append(f"- For {total_hits} HIT queries: tree found expected in {tree_found_expected} ({tree_found_expected/total_hits*100:.0f}%), KG in {kg_found_expected} ({kg_found_expected/total_hits*100:.0f}%)")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    # Finding 1: Doc matching
    lines.append("### 1. Tree picks correct document most of the time")
    lines.append(f"- Same doc: {same_doc_counts}/{n} ({same_doc_counts/n*100:.0f}%)")
    lines.append(f"- Partially same: {partially_same_doc}/{n}")
    lines.append(f"- Wrong doc: {diff_doc_counts}/{n} ({diff_doc_counts/n*100:.0f}%)")
    lines.append(f"- **Implication:** Loop 1 (chapter selection) and Loop 2 (article selection) are the bottlenecks, not Loop 0")
    lines.append("")

    # Finding 2: Default articles
    lines.append("### 2. 'Default' articles dominate tree output")
    lines.append("Articles appearing in >15% of miss queries:")
    lines.append("")
    lines.append("| Article | Count | % of misses |")
    lines.append("|---------|-------|-------------|")
    for art, count in tree_articles_all.most_common():
        if count >= n * 0.15:
            lines.append(f"| `{art}` | {count} | {count/n*100:.0f}% |")
    lines.append("")

    # Finding 3: Failed docs
    lines.append("### 3. Failures by expected document")
    lines.append("")
    lines.append("| Document | Miss queries | Tree in same doc |")
    lines.append("|----------|-------------|------------------|")
    for doc in sorted(misses_by_doc.keys(), key=lambda x: -len(misses_by_doc[x])):
        queries = misses_by_doc[doc]
        same = sum(1 for q in queries if q["same_doc"])
        lines.append(f"| `{doc}` | {len(queries)} | {same} |")
    lines.append("")

    # Finding 4: Categories
    lines.append("### 4. Failures by category (top 10)")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    cat_sorted = sorted(misses_by_cat.items(), key=lambda x: -len(x[1]))
    for cat, queries in cat_sorted[:10]:
        lines.append(f"| {cat} | {len(queries)} |")
    lines.append("")

    # Finding 5: Cross-run
    lines.append("### 5. Cross-run consistency")
    if runs:
        lines.append(f"- Fully consistent: {consistent_count}/{n} ({consistent_count/n*100:.0f}%)")
        lines.append(f"- Partially consistent: {partially_consistent}/{n}")
        lines.append(f"- Totally different: {totally_different}/{n}")
        lines.append(f"- Mean Jaccard: {avg_overlap:.3f}")
        lines.append("")
        if hit_variation:
            lines.append("Hit@10 variation across runs:")
            lines.append("")
            for h in sorted(hit_variation.keys()):
                lines.append(f"- Hit in {h}/{len(runs)} runs: {hit_variation[h]} queries")
        lines.append("")
        lines.append("**Interpretation:** " + (
            "High variance — LLM is randomly picking different chapters/articles each run. Prompt improvement or deterministic fallbacks could help."
            if avg_overlap < 0.5 else
            "Moderate consistency — LLM tends to pick same wrong articles. The chapter/article prompts need better cues, not just temperature tuning."
            if avg_overlap < 0.8 else
            "Very consistent — LLM deterministically picks wrong articles. Structural issue in prompts or chapter descriptions."
        ))
    lines.append("")

    # Finding 6: Tree contribution to hits
    lines.append("### 6. Tree vs KG contribution to HITs")
    lines.append(f"- Both found: {both_found}/{total_hits} ({both_found/total_hits*100:.0f}%)")
    lines.append(f"- Only tree: {only_tree}/{total_hits} ({only_tree/total_hits*100:.0f}%)")
    lines.append(f"- Only KG: {only_kg}/{total_hits} ({only_kg/total_hits*100:.0f}%)")
    lines.append(f"- Neither (RRF merge): {neither_found}/{total_hits} ({neither_found/total_hits*100:.0f}%)")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### High impact (address Loop 1 chapter selection)")
    lines.append("1. **Improve chapter descriptions/keywords** — LLM picks wrong chapters because descriptions don't contain enough discriminative terms")
    lines.append("2. **Increase max_chapters** beyond current limit — more chapters = higher chance of including right one (but watch for noise)")
    lines.append("3. **Add article-level keywords to chapter prompt** — give LLM article titles within each chapter so it can better judge relevance")
    lines.append("")
    lines.append("### Medium impact (address Loop 2 article selection)")
    lines.append("4. **Penalize frequently-wrong articles in tree output** — articles that appear in >15% of miss queries are likely generic/noise")
    lines.append("5. **Improve article summaries/keywords** — ensure every article has discriminative keywords")
    lines.append("6. **Use semantic scores more aggressively** — current semantic_rank hint may not be strong enough")
    lines.append("")
    lines.append("### Lower impact")
    lines.append("7. **Cross-chapter expansion** — if confidence is low, try adjacent chapters (already partially implemented)")
    lines.append("8. **Temperature=0 for deterministic picks** — reduce run-to-run variance")
    lines.append("9. **Two-pass tree traversal** — first pass broad (more chapters), second pass narrow (filter articles)")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
