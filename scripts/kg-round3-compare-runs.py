"""Compare multiple eval runs and compute mean/std for KG Round 3 validation."""

import json
import sys
from pathlib import Path


def load_results(filepath: str) -> dict:
    with open(filepath) as f:
        data = json.load(f)
    return data


def analyze_run(results: list[dict]) -> dict:
    traffic_docs = ["168-2024-ND-CP", "36-2024-QH15", "100-2019-ND-CP"]

    total = len([r for r in results if not r.get("skipped")])
    hit10 = sum(1 for r in results if r.get("full_hit@10") == 1 and not r.get("skipped"))
    hit20 = sum(1 for r in results if r.get("full_hit@20") == 1 and not r.get("skipped"))

    ent_total = ent_hit = traf_total = traf_hit = 0
    for r in results:
        if r.get("skipped"):
            continue
        exp = r.get("expected_articles", [])
        is_traffic = any(any(td in e for td in traffic_docs) for e in exp)
        if is_traffic:
            traf_total += 1
            if r.get("full_hit@10") == 1:
                traf_hit += 1
        else:
            ent_total += 1
            if r.get("full_hit@10") == 1:
                ent_hit += 1

    mrr_sum = sum(r.get("full_mrr", 0) for r in results if not r.get("skipped"))

    return {
        "total": total,
        "hit10": hit10,
        "hit10_pct": hit10 / total * 100 if total else 0,
        "hit20": hit20,
        "hit20_pct": hit20 / total * 100 if total else 0,
        "mrr": mrr_sum / total if total else 0,
        "ent_hit": ent_hit,
        "ent_total": ent_total,
        "ent_pct": ent_hit / ent_total * 100 if ent_total else 0,
        "traf_hit": traf_hit,
        "traf_total": traf_total,
        "traf_pct": traf_hit / traf_total * 100 if traf_total else 0,
    }


def main():
    files = sys.argv[1:]
    if not files:
        # Default: look for round3 runs
        files = sorted(Path("results").glob("hard400_round3_full_run*.json"))
        if not files:
            print("No result files found. Pass file paths as arguments.")
            return

    print(f"Comparing {len(files)} runs:\n")

    runs = []
    for f in files:
        f = str(f)
        data = load_results(f)
        results = data["results"]
        stats = analyze_run(results)
        runs.append(stats)
        print(f"  {Path(f).name}: Hit@10={stats['hit10']}/{stats['total']} ({stats['hit10_pct']:.1f}%), "
              f"Ent={stats['ent_hit']}/{stats['ent_total']} ({stats['ent_pct']:.1f}%), "
              f"Traf={stats['traf_hit']}/{stats['traf_total']} ({stats['traf_pct']:.1f}%), "
              f"MRR={stats['mrr']:.3f}")

    if len(runs) >= 2:
        import statistics
        hit10s = [r["hit10_pct"] for r in runs]
        ent_pcts = [r["ent_pct"] for r in runs]
        traf_pcts = [r["traf_pct"] for r in runs]
        mrrs = [r["mrr"] for r in runs]

        print(f"\n=== Summary ({len(runs)} runs) ===")
        print(f"  Hit@10: mean={statistics.mean(hit10s):.1f}% ± {statistics.stdev(hit10s):.1f}pp "
              f"(range {min(hit10s):.1f}%-{max(hit10s):.1f}%)")
        print(f"  Enterprise: mean={statistics.mean(ent_pcts):.1f}% ± {statistics.stdev(ent_pcts):.1f}pp")
        print(f"  Traffic: mean={statistics.mean(traf_pcts):.1f}% ± {statistics.stdev(traf_pcts):.1f}pp")
        print(f"  MRR: mean={statistics.mean(mrrs):.3f} ± {statistics.stdev(mrrs):.3f}")

        print(f"\n  Baseline: 76.8% (307/400), Ent=81.5%, Traf=72.0%, MRR=0.423")
        delta = statistics.mean(hit10s) - 76.8
        print(f"  Delta vs baseline: {delta:+.1f}pp")


if __name__ == "__main__":
    main()
