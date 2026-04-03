#!/usr/bin/env python3
"""Analyze near-miss queries from hard-400 benchmark to find RRF ranking improvements.

Near-miss = expected article appears in positions 11-20 (just outside top-10).
Finds blocker articles, patterns, and recommends new penalties.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml


# ---------------------------------------------------------------------------
# 1. Load penalty configs from domain YAMLs
# ---------------------------------------------------------------------------
def load_all_penalties(domains_dir: str = "config/domains") -> Dict[str, Dict[int, float]]:
    """Load penalty configs from all domain YAMLs."""
    penalty_config: Dict[str, Dict[int, float]] = {}
    domains_path = Path(domains_dir)
    if not domains_path.exists():
        return penalty_config
    for yaml_file in domains_path.glob("*.yaml"):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                domain = yaml.safe_load(f) or {}
        except Exception:
            continue
        article_penalties = domain.get("article_penalties", {})
        if not article_penalties:
            continue
        doc_id = yaml_file.stem
        doc_penalties: Dict[int, float] = {}
        for _cat_name, cat_data in article_penalties.items():
            if not isinstance(cat_data, dict):
                continue
            multiplier = cat_data.get("multiplier", 1.0)
            articles = cat_data.get("articles", [])
            for art_num in articles:
                doc_penalties[int(art_num)] = multiplier
        if doc_penalties:
            penalty_config[doc_id] = doc_penalties
    return penalty_config


def get_penalty(article_id: str, penalty_config: Dict[str, Dict[int, float]]) -> float:
    """Get penalty multiplier for an article ID (1.0 = no penalty)."""
    if ":d" not in article_id:
        return 1.0
    parts = article_id.rsplit(":d", 1)
    if len(parts) != 2:
        return 1.0
    doc_id = parts[0]
    try:
        art_num = int(parts[1].split(":")[0])
    except ValueError:
        return 1.0
    return penalty_config.get(doc_id, {}).get(art_num, 1.0)


def parse_doc_id(article_id: str) -> str:
    """Extract doc_id from article ID like '59-2020-QH14:d206' -> '59-2020-QH14'."""
    if ":d" in article_id:
        return article_id.rsplit(":d", 1)[0]
    return article_id


def parse_art_num(article_id: str) -> int:
    """Extract article number from ID like '59-2020-QH14:d206' -> 206."""
    if ":d" in article_id:
        try:
            return int(article_id.rsplit(":d", 1)[1].split(":")[0])
        except ValueError:
            return -1
    return -1


# ---------------------------------------------------------------------------
# 2. Load results and find near-misses
# ---------------------------------------------------------------------------
def load_results(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def classify_doc(article_id: str) -> str:
    """Classify article as enterprise or traffic domain."""
    traffic_docs = {"36-2024-QH15", "168-2024-ND-CP", "100-2019-ND-CP"}
    doc = parse_doc_id(article_id)
    return "traffic" if doc in traffic_docs else "enterprise"


def find_near_misses(results: List[Dict], penalty_config: Dict) -> List[Dict]:
    """Find queries where expected article is at position 11-20."""
    near_misses = []
    for r in results:
        if r.get("skipped"):
            continue
        full_retrieved = r.get("full_retrieved", [])
        expected_articles = r.get("expected_articles", [])
        hit_at_10 = r.get("full_hit@10", 0)

        # Only look at queries that MISSED at hit@10
        # (some expected articles may be in top-10 but others missed)
        for exp_art in expected_articles:
            # Normalize: expected might be "59-2020-QH14:d206:k1" but retrieved is "59-2020-QH14:d206"
            exp_base = exp_art.split(":k")[0] if ":k" in exp_art else exp_art

            if exp_base in full_retrieved:
                pos = full_retrieved.index(exp_base) + 1  # 1-indexed
                if 11 <= pos <= 20:
                    # This is a near-miss
                    blockers = full_retrieved[:pos - 1]  # articles above it (top 1 to pos-1)
                    top10_blockers = full_retrieved[:10]

                    # Which retriever found it?
                    tree_arts = r.get("full_tree_articles", [])
                    kg_arts = r.get("full_kg_articles", [])
                    in_tree = exp_base in tree_arts
                    in_kg = exp_base in kg_arts

                    # Blocker analysis
                    blocker_details = []
                    for i, b in enumerate(top10_blockers):
                        b_penalty = get_penalty(b, penalty_config)
                        b_doc = parse_doc_id(b)
                        same_doc = (b_doc == parse_doc_id(exp_base))
                        blocker_details.append({
                            "article_id": b,
                            "rank": i + 1,
                            "doc_id": b_doc,
                            "same_doc_as_expected": same_doc,
                            "current_penalty": b_penalty,
                            "already_penalized": b_penalty < 1.0,
                            "domain": classify_doc(b),
                        })

                    near_misses.append({
                        "stt": r.get("stt"),
                        "question": r.get("question", "")[:120],
                        "category": r.get("category", ""),
                        "expected_article": exp_base,
                        "expected_doc": parse_doc_id(exp_base),
                        "expected_domain": classify_doc(exp_base),
                        "position": pos,
                        "in_tree": in_tree,
                        "in_kg": in_kg,
                        "retriever_source": (
                            "both" if (in_tree and in_kg)
                            else "tree" if in_tree
                            else "kg" if in_kg
                            else "none_tracked"
                        ),
                        "top10_blockers": blocker_details,
                        "hit_at_10": hit_at_10,
                        "hit_at_20": r.get("full_hit@20", 0),
                    })

    return near_misses


# ---------------------------------------------------------------------------
# 3. Aggregate analysis
# ---------------------------------------------------------------------------
def aggregate_analysis(
    near_misses: List[Dict],
    all_expected_articles: Set[str],
    penalty_config: Dict,
) -> Dict:
    """Compute aggregate stats from near-misses."""

    # Position distribution
    pos_dist = Counter(nm["position"] for nm in near_misses)

    # Retriever source distribution
    source_dist = Counter(nm["retriever_source"] for nm in near_misses)

    # Domain distribution
    domain_dist = Counter(nm["expected_domain"] for nm in near_misses)

    # Most common blocker articles across all near-misses
    blocker_counter = Counter()
    blocker_unpenalized_counter = Counter()
    for nm in near_misses:
        for b in nm["top10_blockers"]:
            blocker_counter[b["article_id"]] += 1
            if not b["already_penalized"]:
                blocker_unpenalized_counter[b["article_id"]] += 1

    # Pure noise blockers: appear as blockers but NEVER as expected answer
    pure_noise = {}
    for art_id, count in blocker_counter.most_common(50):
        if art_id not in all_expected_articles:
            pure_noise[art_id] = {
                "count": count,
                "already_penalized": get_penalty(art_id, penalty_config) < 1.0,
                "current_penalty": get_penalty(art_id, penalty_config),
                "domain": classify_doc(art_id),
            }

    # How many near-misses could be fixed by penalizing pure-noise blockers?
    fixable = 0
    fixable_details = []
    for nm in near_misses:
        # Count how many of top-10 blockers are pure noise and unpenalized
        noise_in_top10 = sum(
            1 for b in nm["top10_blockers"]
            if b["article_id"] in pure_noise and not b["already_penalized"]
        )
        if noise_in_top10 > 0:
            fixable += 1
            fixable_details.append({
                "stt": nm["stt"],
                "position": nm["position"],
                "noise_blockers": noise_in_top10,
                "expected": nm["expected_article"],
            })

    # Cross-document blockers (from different doc than expected)
    cross_doc_blockers = Counter()
    for nm in near_misses:
        for b in nm["top10_blockers"]:
            if not b["same_doc_as_expected"]:
                cross_doc_blockers[b["article_id"]] += 1

    return {
        "total_near_misses": len(near_misses),
        "position_distribution": dict(sorted(pos_dist.items())),
        "retriever_source_distribution": dict(source_dist),
        "domain_distribution": dict(domain_dist),
        "top_blocker_articles": [
            {
                "article_id": art,
                "times_blocking": cnt,
                "penalty": get_penalty(art, penalty_config),
                "is_pure_noise": art not in all_expected_articles,
                "domain": classify_doc(art),
            }
            for art, cnt in blocker_counter.most_common(30)
        ],
        "pure_noise_blockers": {
            k: v for k, v in sorted(
                pure_noise.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )[:25]
        },
        "fixable_by_new_penalties": fixable,
        "fixable_details": fixable_details,
        "top_cross_doc_blockers": [
            {"article_id": art, "count": cnt, "domain": classify_doc(art)}
            for art, cnt in cross_doc_blockers.most_common(20)
        ],
    }


# ---------------------------------------------------------------------------
# 4. Cross-run consistency
# ---------------------------------------------------------------------------
def cross_run_analysis(run_files: List[str], penalty_config: Dict) -> Dict:
    """Check if the same queries are near-miss across multiple runs."""
    run_near_miss_stts = {}
    run_near_miss_positions = {}

    for rf in run_files:
        if not os.path.exists(rf):
            continue
        run_name = Path(rf).stem
        results = load_results(rf)
        nms = find_near_misses(results, penalty_config)
        stt_set = set()
        pos_map = {}
        for nm in nms:
            key = f"{nm['stt']}_{nm['expected_article']}"
            stt_set.add(key)
            pos_map[key] = nm["position"]
        run_near_miss_stts[run_name] = stt_set
        run_near_miss_positions[run_name] = pos_map

    if len(run_near_miss_stts) < 2:
        return {"note": "Need at least 2 runs for cross-run analysis"}

    all_keys = set()
    for s in run_near_miss_stts.values():
        all_keys |= s

    run_names = list(run_near_miss_stts.keys())
    consistent = []  # in ALL runs
    partial = []     # in some runs
    unique = []      # only in 1 run

    for key in sorted(all_keys):
        in_runs = [rn for rn in run_names if key in run_near_miss_stts[rn]]
        positions = {rn: run_near_miss_positions[rn].get(key, -1) for rn in in_runs}

        entry = {"query_key": key, "in_runs": len(in_runs), "positions": positions}

        if len(in_runs) == len(run_names):
            consistent.append(entry)
        elif len(in_runs) == 1:
            unique.append(entry)
        else:
            partial.append(entry)

    return {
        "num_runs": len(run_names),
        "total_unique_near_miss_keys": len(all_keys),
        "consistent_across_all_runs": len(consistent),
        "partial_overlap": len(partial),
        "unique_to_single_run": len(unique),
        "consistent_details": consistent[:30],
        "partial_details": partial[:20],
    }


# ---------------------------------------------------------------------------
# 5. Build all expected articles set
# ---------------------------------------------------------------------------
def get_all_expected(results: List[Dict]) -> Set[str]:
    """Collect all expected article base IDs from benchmark."""
    expected = set()
    for r in results:
        for ea in r.get("expected_articles", []):
            base = ea.split(":k")[0] if ":k" in ea else ea
            expected.add(base)
    return expected


# ---------------------------------------------------------------------------
# 6. Generate recommendation
# ---------------------------------------------------------------------------
def generate_recommendations(agg: Dict, near_misses: List[Dict]) -> List[str]:
    """Generate actionable penalty recommendations."""
    recs = []

    # Group pure noise by doc
    noise_by_doc = defaultdict(list)
    for art_id, info in agg["pure_noise_blockers"].items():
        if not info["already_penalized"] and info["count"] >= 2:
            doc = parse_doc_id(art_id)
            art_num = parse_art_num(art_id)
            noise_by_doc[doc].append((art_num, info["count"]))

    for doc_id, arts in sorted(noise_by_doc.items(), key=lambda x: -sum(a[1] for a in x[1])):
        arts_sorted = sorted(arts, key=lambda x: -x[1])
        art_nums = [str(art_num) for art_num, _ in arts_sorted]
        total_blocks = sum(cnt for _, cnt in arts_sorted)
        recs.append(
            f"ADD PENALTY {doc_id}: articles [{', '.join(art_nums)}] "
            f"(blocking {total_blocks} near-miss queries total)"
        )

    # Check already-penalized blockers that still block (penalty too weak?)
    weak_penalties = Counter()
    for nm in near_misses:
        for b in nm["top10_blockers"]:
            if b["already_penalized"] and b["current_penalty"] > 0.15:
                weak_penalties[b["article_id"]] += 1

    for art_id, cnt in weak_penalties.most_common(10):
        if cnt >= 2:
            cur = get_penalty(art_id, load_all_penalties("config/domains"))
            recs.append(
                f"STRENGTHEN PENALTY {art_id}: currently {cur}, blocks {cnt} near-misses — consider lowering to 0.15"
            )

    return recs


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.chdir("/home/hienlh/Projects/vn_legal_rag")

    # Load penalty config
    penalty_config = load_all_penalties("config/domains")
    print("=== Loaded Penalty Config ===")
    for doc, penalties in sorted(penalty_config.items()):
        arts = sorted(penalties.items())
        print(f"  {doc}: {dict(arts)}")
    print()

    # Load primary run
    run1_file = "results/hard400_round3_batch5_run1.json"
    results = load_results(run1_file)
    print(f"Loaded {len(results)} results from {run1_file}")

    # Collect all expected articles
    all_expected = get_all_expected(results)
    print(f"Total unique expected articles in benchmark: {len(all_expected)}")

    # Find near-misses
    near_misses = find_near_misses(results, penalty_config)
    print(f"\n{'='*60}")
    print(f"NEAR-MISS QUERIES: {len(near_misses)}")
    print(f"(Expected article at positions 11-20)")
    print(f"{'='*60}\n")

    # Print each near-miss detail
    for nm in near_misses:
        print(f"  Q#{nm['stt']} [{nm['expected_domain']}] pos={nm['position']} "
              f"expected={nm['expected_article']} via={nm['retriever_source']}")
        print(f"    Category: {nm['category']}")
        print(f"    Question: {nm['question']}...")

        # Show blockers that are unpenalized
        unpenalized_blockers = [
            b for b in nm["top10_blockers"] if not b["already_penalized"]
        ]
        penalized_blockers = [
            b for b in nm["top10_blockers"] if b["already_penalized"]
        ]
        cross_doc = [b for b in nm["top10_blockers"] if not b["same_doc_as_expected"]]

        print(f"    Top-10 blockers: {len(nm['top10_blockers'])} "
              f"({len(unpenalized_blockers)} unpenalized, "
              f"{len(penalized_blockers)} penalized, "
              f"{len(cross_doc)} cross-doc)")

        # Show top unpenalized blockers that are not expected anywhere
        for b in unpenalized_blockers[:5]:
            is_noise = b["article_id"] not in all_expected
            print(f"      rank={b['rank']} {b['article_id']} "
                  f"{'[PURE NOISE]' if is_noise else '[valid elsewhere]'} "
                  f"{'[cross-doc]' if not b['same_doc_as_expected'] else ''}")
        print()

    # Aggregate analysis
    agg = aggregate_analysis(near_misses, all_expected, penalty_config)
    print(f"\n{'='*60}")
    print("AGGREGATE ANALYSIS")
    print(f"{'='*60}")

    print(f"\nPosition distribution (rank -> count):")
    for pos, cnt in sorted(agg["position_distribution"].items()):
        bar = '#' * cnt
        print(f"  {pos:>2}: {cnt:>3} {bar}")

    print(f"\nRetriever source distribution:")
    for src, cnt in sorted(agg["retriever_source_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    print(f"\nDomain distribution:")
    for dom, cnt in sorted(agg["domain_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {dom}: {cnt}")

    print(f"\nTop 20 blocker articles (appear most in top-10 blocking near-misses):")
    for b in agg["top_blocker_articles"][:20]:
        noise = "PURE_NOISE" if b["is_pure_noise"] else "valid_elsewhere"
        pen = f"penalty={b['penalty']}" if b["penalty"] < 1.0 else "no_penalty"
        print(f"  {b['article_id']:>25s} blocks={b['times_blocking']:>2} "
              f"[{noise}] [{pen}] [{b['domain']}]")

    print(f"\nPure noise blockers (never expected, sorted by blocking frequency):")
    for art_id, info in list(agg["pure_noise_blockers"].items())[:20]:
        pen = f"penalty={info['current_penalty']}" if info["already_penalized"] else "NO_PENALTY"
        print(f"  {art_id:>25s} blocks={info['count']:>2} [{pen}] [{info['domain']}]")

    print(f"\nFixable by new penalties: {agg['fixable_by_new_penalties']}/{len(near_misses)} near-misses")
    print(f"  (have at least 1 unpenalized pure-noise blocker in top-10)")

    print(f"\nTop cross-document blockers:")
    for b in agg["top_cross_doc_blockers"][:15]:
        print(f"  {b['article_id']:>25s} cross-doc blocks={b['count']:>2} [{b['domain']}]")

    # Cross-run analysis
    print(f"\n{'='*60}")
    print("CROSS-RUN CONSISTENCY (3 runs)")
    print(f"{'='*60}")
    run_files = [
        f"results/hard400_round3_batch5_run{i}.json" for i in [1, 2, 3]
    ]
    cross = cross_run_analysis(run_files, penalty_config)

    if "note" not in cross:
        print(f"  Total unique near-miss keys: {cross['total_unique_near_miss_keys']}")
        print(f"  Consistent across ALL 3 runs: {cross['consistent_across_all_runs']}")
        print(f"  Partial overlap (2 of 3): {cross['partial_overlap']}")
        print(f"  Unique to single run: {cross['unique_to_single_run']}")

        if cross["consistent_details"]:
            print(f"\n  Consistently near-miss (all 3 runs):")
            for c in cross["consistent_details"]:
                positions = ", ".join(
                    f"{rn.split('_')[-1]}={p}" for rn, p in c["positions"].items()
                )
                print(f"    {c['query_key']:>40s}  positions: {positions}")
    else:
        print(f"  {cross['note']}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    recs = generate_recommendations(agg, near_misses)
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec}")

    if not recs:
        print("  No strong penalty recommendations found.")

    # ---------------------------------------------------------------------------
    # Save JSON output
    # ---------------------------------------------------------------------------
    output = {
        "metadata": {
            "primary_run": run1_file,
            "total_queries": len(results),
            "total_near_misses": len(near_misses),
        },
        "near_misses": near_misses,
        "aggregate": agg,
        "cross_run": cross,
        "recommendations": recs,
    }

    out_json = "results/hard400_near_miss_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved analysis JSON to {out_json}")

    # ---------------------------------------------------------------------------
    # Save markdown report
    # ---------------------------------------------------------------------------
    report_path = "plans/reports/analysis-260401-1350-near-miss-reranking.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Near-Miss Reranking Analysis — Hard-400 Benchmark\n\n")
        f.write(f"**Date:** 2026-04-01  \n")
        f.write(f"**Primary run:** `{run1_file}` (78.9% Hit@10)  \n")
        f.write(f"**Near-misses found:** {len(near_misses)} queries with expected article at rank 11-20  \n\n")

        f.write("## Position Distribution\n\n")
        f.write("| Rank | Count |\n|------|-------|\n")
        for pos, cnt in sorted(agg["position_distribution"].items()):
            f.write(f"| {pos} | {cnt} |\n")

        f.write(f"\n## Retriever Source\n\n")
        for src, cnt in sorted(agg["retriever_source_distribution"].items(), key=lambda x: -x[1]):
            f.write(f"- **{src}**: {cnt}\n")

        f.write(f"\n## Domain Split\n\n")
        for dom, cnt in sorted(agg["domain_distribution"].items(), key=lambda x: -x[1]):
            f.write(f"- **{dom}**: {cnt}\n")

        f.write(f"\n## Cross-Run Consistency\n\n")
        if "note" not in cross:
            f.write(f"- Consistent across all 3 runs: **{cross['consistent_across_all_runs']}**\n")
            f.write(f"- Partial (2/3 runs): {cross['partial_overlap']}\n")
            f.write(f"- Single-run only: {cross['unique_to_single_run']}\n")
        else:
            f.write(f"{cross['note']}\n")

        f.write(f"\n## Top Pure-Noise Blockers (never expected, unpenalized)\n\n")
        f.write("| Article | Blocks | Current Penalty | Domain |\n")
        f.write("|---------|--------|-----------------|--------|\n")
        for art_id, info in list(agg["pure_noise_blockers"].items())[:20]:
            pen_str = str(info["current_penalty"]) if info["already_penalized"] else "none"
            f.write(f"| {art_id} | {info['count']} | {pen_str} | {info['domain']} |\n")

        f.write(f"\n## Fixability\n\n")
        f.write(f"- **{agg['fixable_by_new_penalties']}/{len(near_misses)}** near-misses have unpenalized pure-noise blockers in top-10\n")

        f.write(f"\n## Recommendations\n\n")
        for i, rec in enumerate(recs, 1):
            f.write(f"{i}. {rec}\n")

        if not recs:
            f.write("No strong penalty recommendations at this time.\n")

        # Consistently near-miss queries
        if "note" not in cross and cross.get("consistent_details"):
            f.write(f"\n## Consistently Near-Miss Queries (all 3 runs)\n\n")
            f.write("| Query Key | Positions (run1, run2, run3) |\n")
            f.write("|-----------|------------------------------|\n")
            for c in cross["consistent_details"]:
                pos_str = ", ".join(f"{p}" for p in c["positions"].values())
                f.write(f"| {c['query_key']} | {pos_str} |\n")

    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
