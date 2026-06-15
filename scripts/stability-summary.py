"""Compute 3-run stability variance for legal benchmark."""
import json
import statistics


def stats(f):
    r = json.load(open(f, "r", encoding="utf-8"))["results"]
    h = sum(1 for x in r if x.get("hit")) / len(r) * 100
    m = sum(x.get("ir_metrics", {}).get("rr", 0) for x in r) / len(r)
    t = sum(1 for x in r if x.get("tree_hit")) / len(r) * 100
    k = sum(1 for x in r if x.get("kg_hit")) / len(r) * 100
    return h, m, t, k


runs = {
    "run3_round2": "results/benchmark_run3_round2.json",
    "Stability A": "results/benchmark_stability_A.json",
    "Stability B": "results/benchmark_stability_B.json",
}

hits, mrrs = [], []
print(f'{"Run":14} {"Hit@10":>8} {"MRR":>8} {"Tree":>7} {"KG":>7}')
for n, f in runs.items():
    h, m, t, k = stats(f)
    hits.append(h)
    mrrs.append(m)
    print(f"{n:14} {h:7.2f}% {m:8.4f} {t:6.1f}% {k:6.1f}%")

print()
print(f"Hit@10  mean={statistics.mean(hits):.2f}%  range={min(hits):.2f}-{max(hits):.2f}  spread={max(hits)-min(hits):.2f}pp  stdev={statistics.stdev(hits):.3f}")
print(f"MRR     mean={statistics.mean(mrrs):.4f}  range={min(mrrs):.4f}-{max(mrrs):.4f}  stdev={statistics.stdev(mrrs):.4f}")
print()
print("Paper target: Hit@10 92.2%, MRR 0.603")
