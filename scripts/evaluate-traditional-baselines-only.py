"""
Traditional baseline evaluation only (no RAG, no LLM proxy needed).

Runs BM25, TF-IDF, Semantic, SimCSE-PhoBERT, Keyword baselines on any domain.
Pure local computation — finishes in 1-2 minutes.

Usage:
    python scripts/evaluate-traditional-baselines-only.py \
        --config config/education.yaml \
        --test-file data/benchmark/edu-master-qa-benchmark.csv \
        -o results/edu_traditional_baselines.json
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import importlib.util

# Import traditional baselines from separate module
_spec = importlib.util.spec_from_file_location(
    "traditional_baseline_retrievers",
    str(project_root / "scripts" / "traditional-baseline-retrievers.py"),
)
_trad_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trad_mod)
init_traditional_baselines = _trad_mod.init_traditional_baselines

K_VALUES = [1, 3, 5, 10, 15, 20, 30]


def extract_expected_article_ids(article_ids: str) -> set:
    if not article_ids or article_ids.strip() == "":
        return set()
    articles = set()
    for ref in article_ids.split(";"):
        ref = ref.strip()
        if not ref:
            continue
        match = re.match(r'(.+?:d\d+)', ref)
        if match:
            articles.add(match.group(1))
    return articles


def calc_ir_metrics(expected: set, ranked: list) -> dict:
    if not expected or not ranked:
        return {"rr": 0.0, **{f"hit@{k}": 0 for k in K_VALUES},
                **{f"recall@{k}": 0.0 for k in K_VALUES}}
    rr = 0.0
    for i, a in enumerate(ranked):
        if a in expected:
            rr = 1.0 / (i + 1)
            break
    metrics = {"rr": rr}
    for k in K_VALUES:
        top_k = set(ranked[:k])
        relevant = len(expected & top_k)
        metrics[f"hit@{k}"] = 1 if relevant > 0 else 0
        metrics[f"recall@{k}"] = relevant / len(expected)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Traditional baselines only (no RAG)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    # Load config for db_path
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = config.get("database", {}).get("path", "data/legal_docs.db")

    print(f"[1/3] Loading articles from {db_path}...")

    # Init embedding provider for semantic baseline
    from vn_legal_rag.utils import create_embedding_provider
    embedding_gen = create_embedding_provider()

    print(f"[2/3] Initializing traditional baselines...")
    traditional = init_traditional_baselines(db_path=db_path, embedding_gen=embedding_gen)
    baseline_names = list(traditional.keys())
    print(f"      Baselines: {baseline_names}")

    # Load test data
    print(f"[3/3] Loading test data from {args.test_file}...")
    with open(args.test_file, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"      Total questions: {len(rows)}")

    # Evaluate
    results = []
    totals = {b: {f"hit@{k}": 0 for k in K_VALUES} for b in baseline_names}
    totals_mrr = {b: 0.0 for b in baseline_names}
    evaluated = 0
    start = time.time()

    header_baselines = " ".join(f"{b[:8]:>8}" for b in baseline_names)
    print(f"\n{'#':>4} {'STT':>5} {header_baselines}")
    print("-" * (12 + 9 * len(baseline_names)))

    for idx, row in enumerate(rows):
        stt = row.get("STT", str(idx + 1))
        question = row.get("Content", "") or row.get("question", "")
        article_ids = row.get("Article_IDs", "") or row.get("article_ids", "")
        expected = extract_expected_article_ids(article_ids)

        if not expected:
            continue

        evaluated += 1
        record = {"stt": stt, "question": question, "expected_articles": sorted(expected)}

        hits_line = []
        for bname, retriever in traditional.items():
            try:
                ranked = retriever.search(question, top_k=30)
                ir = calc_ir_metrics(expected, ranked)
                record[f"{bname}_retrieved"] = ranked[:10]
                for k in K_VALUES:
                    record[f"{bname}_hit@{k}"] = ir[f"hit@{k}"]
                    record[f"{bname}_recall@{k}"] = round(ir[f"recall@{k}"], 4)
                    totals[bname][f"hit@{k}"] += ir[f"hit@{k}"]
                record[f"{bname}_mrr"] = round(ir["rr"], 4)
                totals_mrr[bname] += ir["rr"]
                hits_line.append(f"{'HIT':>8}" if ir["hit@10"] else f"{'MISS':>8}")
            except Exception as e:
                record[f"{bname}_error"] = str(e)
                hits_line.append(f"{'ERR':>8}")

        print(f"{evaluated:>4} {stt:>5} {' '.join(hits_line)}")
        results.append(record)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print(f"Evaluated: {evaluated} questions in {elapsed:.1f}s")
    print(f"\n{'Baseline':<20} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'Hit@20':>7} {'MRR':>6}")
    print("-" * 54)

    summary = {}
    for b in baseline_names:
        h1 = totals[b]["hit@1"] / evaluated * 100 if evaluated else 0
        h5 = totals[b]["hit@5"] / evaluated * 100 if evaluated else 0
        h10 = totals[b]["hit@10"] / evaluated * 100 if evaluated else 0
        h20 = totals[b]["hit@20"] / evaluated * 100 if evaluated else 0
        mrr = totals_mrr[b] / evaluated if evaluated else 0
        print(f"{b:<20} {h1:>5.1f}% {h5:>5.1f}% {h10:>6.1f}% {h20:>6.1f}% {mrr:>.3f}")
        summary[b] = {
            "hit@10": f"{totals[b]['hit@10']}/{evaluated} ({h10:.1f}%)",
            "mrr": round(mrr, 4),
            **{f"hit@{k}_rate": round(totals[b][f"hit@{k}"] / evaluated * 100, 1) for k in K_VALUES},
        }

    # Save
    output = {
        "config": args.config,
        "test_file": args.test_file,
        "db_path": db_path,
        "evaluated": evaluated,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": summary,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
