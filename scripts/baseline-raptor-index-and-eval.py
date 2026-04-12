#!/usr/bin/env python3
"""
RAPTOR baseline using the original library (Sarthi et al., ICLR 2024).

Uses the actual RAPTOR code from baselines/raptor/ with custom models:
  - Summarization: Anthropic Claude via proxy (OpenAI-compatible)
  - Embedding: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)

Phase 1 (Offline): Concatenate all articles → RAPTOR chunks (100 tokens),
  embeds, clusters (UMAP + GMM), summarizes → recursive tree. Cached as pickle.
Phase 2 (Online): Collapsed tree retrieval (cosine sim on all nodes),
  map selected nodes → article IDs → Hit@k, MRR.

Usage:
    # Full pipeline (build tree if needed + evaluate)
    env -u ANTHROPIC_AUTH_TOKEN python scripts/baseline-raptor-index-and-eval.py \\
        --base-url http://localhost:3210/proxy -o results/hard400_raptor_run1.json

    # Evaluate only (tree must already exist)
    python scripts/baseline-raptor-index-and-eval.py --eval-only -o results/raptor_eval.json

    # Quick test
    python scripts/baseline-raptor-index-and-eval.py --limit 10 -o results/raptor_test.json

Dependencies:
    pip install umap-learn tiktoken tenacity scipy
    cd baselines/raptor && pip install -r requirements.txt  # or just the above
"""

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "baselines" / "raptor"))

from dotenv import load_dotenv
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Metrics (same as other baselines)
# ---------------------------------------------------------------------------

K_VALUES = [1, 3, 5, 10, 20, 30]


def extract_expected_article_ids(article_ids: str) -> set:
    """Parse expected article IDs from benchmark CSV."""
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
    """Calculate IR metrics for a single query."""
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


# ---------------------------------------------------------------------------
# Article loading from SQLite
# ---------------------------------------------------------------------------

def load_articles_from_db(db_path: str) -> list:
    """Load all articles from legal_docs.db with doc-qualified IDs."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, article_number, title, content, document_id
        FROM legal_articles ORDER BY document_id, article_number
    """)
    articles = []
    for article_id, art_num, title, content, doc_id in cur.fetchall():
        art_title = title or f"Điều {art_num}"
        text = f"Điều {art_num}. {art_title}\n\n{content or ''}"
        articles.append({
            "article_id": article_id,
            "title": art_title,
            "content": text,
            "document_id": doc_id,
        })
    conn.close()
    return articles


# ---------------------------------------------------------------------------
# Custom RAPTOR models (summarization via proxy, embedding via SBERT)
# ---------------------------------------------------------------------------

from raptor.SummarizationModels import BaseSummarizationModel
from raptor.EmbeddingModels import BaseEmbeddingModel


class ProxySummarizationModel(BaseSummarizationModel):
    """LLM summarization via Anthropic-compatible proxy (SSE stream)."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def summarize(self, context, max_tokens=150):
        import httpx

        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content":
                          f"Write a summary of the following, including as many "
                          f"key details as possible: {context}:"}],
            "max_tokens": max_tokens,
            "stream": True,
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = httpx.post(url, json=body, headers=headers, timeout=120)
                text_parts = []
                for line in response.text.split("\n"):
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text_parts.append(delta.get("text", ""))
                        except json.JSONDecodeError:
                            continue
                result = "".join(text_parts)
                if result:
                    return result
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  [WARN] summarize failed: {e}")
                return context[:200]  # fallback: truncated original
        return context[:200]


class SBERTEmbeddingModel(BaseEmbeddingModel):
    """Local sentence-transformers embedding model."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


# ---------------------------------------------------------------------------
# Leaf node → article ID mapping (bigram overlap)
# ---------------------------------------------------------------------------

def build_leaf_to_article_map(tree, articles: list) -> dict:
    """Map each leaf node to its source article via bigram overlap.

    RAPTOR chunks articles into ~100-token pieces. We match each chunk
    back to its source article by finding the highest bigram overlap.
    """
    # Precompute bigram sets per article
    article_bigrams = {}
    for art in articles:
        words = art["content"].split()
        bigrams = set()
        for i in range(len(words) - 1):
            bigrams.add(f"{words[i]} {words[i+1]}")
        article_bigrams[art["article_id"]] = bigrams

    leaf_map = {}  # leaf_index → article_id
    # Handle both dict (fresh build) and list (checkpoint) formats
    leaf_items = (tree.leaf_nodes.items() if isinstance(tree.leaf_nodes, dict)
                  else [(n.index, n) for n in tree.leaf_nodes])
    for idx, node in leaf_items:
        words = node.text.strip().split()
        if len(words) < 2:
            leaf_map[idx] = None
            continue

        chunk_bigrams = set()
        for i in range(len(words) - 1):
            chunk_bigrams.add(f"{words[i]} {words[i+1]}")

        best_aid = None
        best_score = 0
        for aid, art_bgs in article_bigrams.items():
            score = len(chunk_bigrams & art_bgs)
            if score > best_score:
                best_score = score
                best_aid = aid

        leaf_map[idx] = best_aid

    mapped = sum(1 for v in leaf_map.values() if v is not None)
    print(f"  Leaf→article mapping: {mapped}/{len(leaf_map)} mapped")
    return leaf_map


def build_node_to_articles_map(tree, leaf_map: dict) -> dict:
    """Map every node (leaf + summary) to the set of article IDs it covers.

    Leaf nodes: directly from leaf_map.
    Summary nodes: union of articles from all descendant leaf nodes.
    """
    node_articles = {}

    # Leaf nodes
    for idx, aid in leaf_map.items():
        node_articles[idx] = {aid} if aid else set()

    # Non-leaf nodes: traverse children recursively (bottom-up by layer)
    for layer in range(1, tree.num_layers + 1):
        if layer not in tree.layer_to_nodes:
            continue
        for node in tree.layer_to_nodes[layer]:
            aids = set()
            for child_idx in node.children:
                aids.update(node_articles.get(child_idx, set()))
            node_articles[node.index] = aids

    return node_articles


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(ra, node_articles, embed_model, benchmark_path, output_path,
                   start=1, limit=None, top_k=30):
    """Run evaluation: for each query, retrieve nodes, map to articles, score."""
    from raptor.utils import (get_node_list, get_embeddings,
                              distances_from_embeddings,
                              indices_of_nearest_neighbors_from_distances)

    with open(benchmark_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    tasks_data = []
    for i, row in enumerate(rows):
        if (i + 1) < start:
            continue
        if limit and len(tasks_data) >= limit:
            break
        tasks_data.append(row)

    print(f"\n  Evaluating {len(tasks_data)} questions...")

    # Precompute all node embeddings once (from the tree)
    node_list = get_node_list(ra.tree.all_nodes)
    emb_model_name = list(node_list[0].embeddings.keys())[0]
    all_embeddings = np.array(get_embeddings(node_list, emb_model_name))

    print(f"  Tree: {len(node_list)} nodes, embedding dim={all_embeddings.shape[1]}")
    print(f"  {'#':>4} {'STT':>5} {'Result':>6}  Running Hit@10")
    print("-" * 55)

    records = []
    hits_10 = 0
    tested = 0

    for idx, row in enumerate(tasks_data):
        stt = row.get("STT", "?")
        question = row.get("Content", "") or row.get("question", "")
        article_ids = row.get("Article_IDs", "") or row.get("article_ids", "")
        category = row.get("Category", "")
        expected = extract_expected_article_ids(article_ids)

        record = {
            "stt": stt, "category": category, "question": question,
            "expected_articles": sorted(expected), "num_expected": len(expected),
        }

        if not expected:
            record["skipped"] = True
            records.append(record)
            print(f"[{idx+1:4d}] STT {stt:>5} SKIPPED")
            continue

        record["skipped"] = False

        # Embed query and compute distances to all tree nodes
        query_emb = embed_model.create_embedding(question)
        distances = distances_from_embeddings(query_emb, all_embeddings)
        nearest = indices_of_nearest_neighbors_from_distances(distances)

        # Map top nodes → article IDs (ordered by relevance)
        ranked_articles = []
        seen = set()
        for node_idx in nearest[:top_k * 3]:  # check more nodes to fill top_k
            node = node_list[node_idx]
            aids = node_articles.get(node.index, set())
            for aid in sorted(aids):  # deterministic order within a node
                if aid and aid not in seen:
                    ranked_articles.append(aid)
                    seen.add(aid)
            if len(ranked_articles) >= top_k:
                break

        ir = calc_ir_metrics(expected, ranked_articles[:top_k])
        record["raptor_retrieved"] = ranked_articles[:top_k]
        for k in K_VALUES:
            record[f"raptor_hit@{k}"] = ir[f"hit@{k}"]
            record[f"raptor_recall@{k}"] = round(ir[f"recall@{k}"], 4)
        record["raptor_mrr"] = round(ir["rr"], 4)

        records.append(record)
        tested += 1
        hit10 = record["raptor_hit@10"]
        hits_10 += hit10
        rate = hits_10 / tested * 100
        result_str = "HIT" if hit10 else "MISS"
        print(f"[{idx+1:4d}] STT {stt:>5} {result_str:>6}  {rate:.1f}%")

        if tested % 50 == 0:
            _save_results(output_path, records, benchmark_path,
                          tested, hits_10, "in_progress")

    return records, tested, hits_10


def _save_results(output_path, results, benchmark_path,
                  tested, hits_10, status):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    non_skipped = [r for r in results if not r.get("skipped")]
    n = len(non_skipped)

    summary = {}
    if n > 0:
        summary = {
            "hit@10": (f"{hits_10}/{tested} ({hits_10/tested*100:.1f}%)"
                       if tested else "N/A"),
            "mrr": round(sum(r.get("raptor_mrr", 0) for r in non_skipped) / n, 4),
        }
        for k in K_VALUES:
            avg = sum(r.get(f"raptor_hit@{k}", 0) for r in non_skipped) / n
            summary[f"hit@{k}_rate"] = round(avg * 100, 1)

    data = {
        "metadata": {
            "baseline": "RAPTOR",
            "paper": "Sarthi et al. (ICLR 2024)",
            "implementation": "original library (baselines/raptor/)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_file": benchmark_path,
            "total_questions": len(results),
            "total_tested": tested,
            "status": status,
        },
        "summary": summary,
        "results": sorted(results, key=lambda r: int(r.get("stt", 0))),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAPTOR baseline (original library): build tree + evaluate",
    )
    parser.add_argument("--db-path", default="data/legal_docs.db",
                        help="Legal docs SQLite DB")
    parser.add_argument("--index-dir", default="data/baseline_raptor_index",
                        help="Directory for tree cache")
    parser.add_argument("--test-file",
                        default="data/benchmark/hard-400-qa-benchmark.csv",
                        help="Benchmark CSV file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSON file")
    parser.add_argument("--llm-model", default="claude-sonnet-4-20250514",
                        help="LLM model for summarization")
    parser.add_argument("--embedding-model",
                        default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence-transformers model for embeddings")
    parser.add_argument("--base-url", default=None,
                        help="Anthropic-compatible API base URL")
    parser.add_argument("--start", type=int, default=1,
                        help="Start from row number")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--reindex", action="store_true",
                        help="Force rebuild tree from scratch")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip tree building, evaluate only")

    args = parser.parse_args()

    print("=" * 55)
    print("RAPTOR Baseline — Original Library")
    print("  Sarthi et al. (ICLR 2024)")
    print("=" * 55)

    output_path = (args.output if args.output.endswith(".json")
                   else f"{args.output}.json")
    tree_path = os.path.join(args.index_dir, "raptor_tree.pkl")
    mapping_path = os.path.join(args.index_dir, "node_to_articles.json")

    # API key
    from dotenv import dotenv_values
    env_vals = dotenv_values()
    api_key = (env_vals.get("ANTHROPIC_API_KEY")
               or os.getenv("ANTHROPIC_API_KEY")
               or os.getenv("OPENAI_API_KEY") or "sk-ant-dummy")
    base_url = (args.base_url or os.getenv("ANTHROPIC_BASE_URL")
                or os.getenv("OPENAI_BASE_URL"))

    print(f"  LLM: {args.llm_model} @ {base_url or 'default'}")
    print(f"  Embedding: {args.embedding_model}")

    # Create custom models
    embed_model = SBERTEmbeddingModel(args.embedding_model)
    summarization_model = ProxySummarizationModel(
        args.llm_model, api_key, base_url)

    # Load articles
    articles = load_articles_from_db(args.db_path)
    print(f"  Loaded {len(articles)} articles from {args.db_path}")

    # Import RAPTOR
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

    # ---------------------------------------------------------------
    # Phase 1: Build RAPTOR tree
    # ---------------------------------------------------------------
    if not args.reindex and os.path.exists(tree_path):
        print(f"\n[Phase 1] Loading cached tree from {tree_path}")
        config = RetrievalAugmentationConfig(
            embedding_model=embed_model,
            summarization_model=summarization_model,
            tb_num_layers=5,
            tb_max_tokens=100,
            tb_summarization_length=100,
            tr_top_k=30,
        )
        ra = RetrievalAugmentation(config, tree=tree_path)
        print(f"  Tree loaded: {len(ra.tree.all_nodes)} nodes, "
              f"{ra.tree.num_layers} layers")
    else:
        if args.eval_only:
            print(f"  [ERROR] --eval-only but no tree at {tree_path}")
            sys.exit(1)

        print(f"\n[Phase 1] Building RAPTOR tree")
        print(f"  Config: max_tokens=100, num_layers=5, threshold=0.5")

        os.makedirs(args.index_dir, exist_ok=True)
        # Set checkpoint path so tree is auto-saved after each layer
        os.environ["RAPTOR_CHECKPOINT_PATH"] = tree_path

        config = RetrievalAugmentationConfig(
            embedding_model=embed_model,
            summarization_model=summarization_model,
            tb_num_layers=5,
            tb_max_tokens=100,        # RAPTOR default: 100-token chunks
            tb_summarization_length=100,
            tr_top_k=30,
        )
        ra = RetrievalAugmentation(config)

        # Concatenate all articles into single text
        full_text = "\n\n".join(a["content"] for a in articles)
        print(f"  Total text: {len(full_text):,} chars")

        t0 = time.time()
        ra.add_documents(full_text)
        elapsed = time.time() - t0

        print(f"  Tree built in {elapsed:.1f}s")
        print(f"  Nodes: {len(ra.tree.all_nodes)}, "
              f"Layers: {ra.tree.num_layers}, "
              f"Leaf nodes: {len(ra.tree.leaf_nodes)}")

        # Save final tree
        ra.save(tree_path)
        print(f"  Tree cached to {tree_path}")

    # ---------------------------------------------------------------
    # Build node → article mapping
    # ---------------------------------------------------------------
    if os.path.exists(mapping_path) and not args.reindex:
        print(f"  Loading cached mapping from {mapping_path}")
        with open(mapping_path, "r") as f:
            raw = json.load(f)
        node_articles = {int(k): set(v) for k, v in raw.items()}
    else:
        print("  Building leaf→article mapping (bigram overlap)...")
        t0 = time.time()
        leaf_map = build_leaf_to_article_map(ra.tree, articles)
        node_articles = build_node_to_articles_map(ra.tree, leaf_map)
        elapsed = time.time() - t0
        print(f"  Mapping built in {elapsed:.1f}s")

        # Cache mapping
        raw = {str(k): sorted(v) for k, v in node_articles.items()}
        with open(mapping_path, "w") as f:
            json.dump(raw, f)

    # Stats
    level_counts = defaultdict(int)
    for layer, nodes in ra.tree.layer_to_nodes.items():
        level_counts[layer] = len(nodes)
    print(f"  Nodes per layer: {dict(sorted(level_counts.items()))}")

    # ---------------------------------------------------------------
    # Phase 2: Evaluation
    # ---------------------------------------------------------------
    print(f"\n[Phase 2] Evaluation on {args.test_file}")
    t0 = time.time()
    results, tested, hits_10 = run_evaluation(
        ra, node_articles, embed_model, args.test_file,
        output_path, start=args.start, limit=args.limit,
    )
    elapsed = time.time() - t0

    _save_results(output_path, results, args.test_file,
                  tested, hits_10, "complete")

    # Print summary
    non_skipped = [r for r in results if not r.get("skipped")]
    n = len(non_skipped)

    print("\n" + "=" * 55)
    print("RAPTOR Baseline Results (Original Library)")
    print("=" * 55)
    if n > 0:
        print(f"\n  {'Metric':<12} {'Value':>10}")
        print(f"  {'-'*12} {'-'*10}")
        for k in K_VALUES:
            avg = sum(r.get(f"raptor_hit@{k}", 0) for r in non_skipped) / n
            print(f"  Hit@{k:<7} {avg*100:>9.1f}%")
        avg_mrr = sum(r.get("raptor_mrr", 0) for r in non_skipped) / n
        avg_r10 = sum(r.get("raptor_recall@10", 0) for r in non_skipped) / n
        print(f"  {'MRR':<12} {avg_mrr:>10.4f}")
        print(f"  {'Recall@10':<12} {avg_r10:>10.4f}")
    print(f"\n  Eval time: {elapsed:.1f}s ({elapsed/max(n,1):.1f}s/query)")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
