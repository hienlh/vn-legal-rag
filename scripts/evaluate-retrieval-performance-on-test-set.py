#!/usr/bin/env python3
"""
Evaluate retrieval performance on test set.

Uses YAML config for GraphRAG initialization (unlike run-full-training-test.py
which uses CLI args directly).

Metrics, output format, ablation analysis, and execution features are identical
to run-full-training-test.py.

Usage:
    # Full evaluation (config-based)
    python scripts/evaluate-retrieval-performance-on-test-set.py

    # With custom config
    python scripts/evaluate-retrieval-performance-on-test-set.py --config config/custom.yaml

    # Limit test questions
    python scripts/evaluate-retrieval-performance-on-test-set.py --limit 50 --verbose

    # Start from specific row
    python scripts/evaluate-retrieval-performance-on-test-set.py --start 100

    # Export results
    python scripts/evaluate-retrieval-performance-on-test-set.py --output results/eval_results.json

    # Run with parallel workers
    python scripts/evaluate-retrieval-performance-on-test-set.py --workers 4

    # Re-test specific failed questions
    python scripts/evaluate-retrieval-performance-on-test-set.py --stt-list failed_stts.json

    # Merge results with existing
    python scripts/evaluate-retrieval-performance-on-test-set.py --stt-list failed.json --merge-with results/prev.json -o results/merged.json

    # Ablation study
    python scripts/evaluate-retrieval-performance-on-test-set.py --ablation no_tree
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress noisy warnings BEFORE importing vn_legal_rag modules
logging.getLogger("vn_legal_rag").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Temporarily suppress stderr during imports
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()

from dotenv import load_dotenv
load_dotenv()

from vn_legal_rag.online import LegalGraphRAG, create_legal_graphrag
from vn_legal_rag.types import NodeType, UnifiedForest
from vn_legal_rag.utils import load_config, load_kg, load_summaries, build_forest_from_db

# Restore stderr
sys.stderr = _stderr

try:
    from vn_legal_rag import AblationConfig, get_paper_ablation_configs
except ImportError:
    AblationConfig = None
    get_paper_ablation_configs = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article extraction helpers (matching run-full-training-test.py exactly)
# ---------------------------------------------------------------------------

def extract_expected_article_ids(article_ids: str) -> set:
    """Extract document-qualified article IDs from Article_IDs column.

    Returns set of strings like {"59-2020-QH14:d206", "59-2020-QH14:d47"}.
    Strips clause info (:k1, :dd2 etc.) to normalize to article level.
    """
    if not article_ids or article_ids.strip() == "":
        return set()
    articles = set()
    for ref in article_ids.split(";"):
        ref = ref.strip()
        if not ref:
            continue
        # Extract doc_id:dN part, stripping clause/point suffixes
        match = re.match(r'(.+?:d\d+)', ref)
        if match:
            articles.add(match.group(1))
    return articles


def extract_doc_id_from_article_id(article_id: str) -> str:
    """Extract document ID from qualified article ID.

    "59-2020-QH14:d206" -> "59-2020-QH14"
    """
    if ':d' in article_id:
        return article_id.rsplit(':d', 1)[0]
    return "unknown"


def extract_tree_articles(result) -> set:
    """Extract document-qualified article IDs from tree search result.

    Uses node.node_id which is already in "doc_id:dN" format (e.g., "59-2020-QH14:d206").
    """
    articles = set()
    if result.tree_search_result:
        for node in result.tree_search_result.target_nodes:
            if node.node_type == NodeType.ARTICLE and node.node_id:
                articles.add(node.node_id)
    return articles


def extract_tree_articles_ranked(result) -> list:
    """Extract document-qualified article IDs from tree search result in ranked order."""
    articles = []
    seen = set()
    if result.tree_search_result:
        for node in result.tree_search_result.target_nodes:
            if node.node_type == NodeType.ARTICLE and node.node_id:
                if node.node_id not in seen:
                    articles.append(node.node_id)
                    seen.add(node.node_id)
    return articles


def extract_kg_articles(result) -> set:
    """Extract document-qualified article IDs from KG/citations.

    Uses cite["source_id"] which is in "doc_id:dN" format.
    Falls back to bare "d{num}" from citation_string if source_id unavailable.
    """
    articles = set()
    if result.citations:
        for cite in result.citations:
            source_id = cite.get("source_id", "")
            if source_id and ":d" in source_id:
                # Normalize: strip clause info if present
                match = re.match(r'(.+?:d\d+)', source_id)
                if match:
                    articles.add(match.group(1))
                    continue
            # Fallback: extract bare article number from citation_string
            cite_str = cite.get("citation_string", "")
            num_match = re.search(r"Điều (\d+)", cite_str)
            if num_match:
                articles.add(f"unknown:d{num_match.group(1)}")
    return articles


def extract_kg_articles_ranked(result) -> list:
    """Extract document-qualified article IDs from KG/citations in order."""
    articles = []
    seen = set()
    if result.citations:
        for cite in result.citations:
            source_id = cite.get("source_id", "")
            article_id = None
            if source_id and ":d" in source_id:
                match = re.match(r'(.+?:d\d+)', source_id)
                if match:
                    article_id = match.group(1)
            if not article_id:
                cite_str = cite.get("citation_string", "")
                num_match = re.search(r"Điều (\d+)", cite_str)
                if num_match:
                    article_id = f"unknown:d{num_match.group(1)}"
            if article_id and article_id not in seen:
                articles.append(article_id)
                seen.add(article_id)
    return articles


def extract_retrieved_articles_ranked(result) -> list:
    """Extract all article numbers in ranked order (tree first, then KG)."""
    tree_ranked = extract_tree_articles_ranked(result)
    kg_ranked = extract_kg_articles_ranked(result)
    seen = set(tree_ranked)
    merged = list(tree_ranked)
    for a in kg_ranked:
        if a not in seen:
            merged.append(a)
            seen.add(a)
    return merged


# ---------------------------------------------------------------------------
# Metric helpers (matching run-full-training-test.py exactly)
# ---------------------------------------------------------------------------

def get_tree_confidence(result) -> float:
    if result.tree_search_result:
        return result.tree_search_result.confidence
    return 0.0


def get_tree_reasoning(result) -> str:
    if result.tree_search_result and result.tree_search_result.reasoning_path:
        return " → ".join(result.tree_search_result.reasoning_path[:2])
    return ""


def calculate_metrics(expected: set, retrieved: set) -> dict:
    """Calculate precision, recall, F1 for a single query."""
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": False, "overlap": 0}
    overlap = expected & retrieved
    overlap_count = len(overlap)
    hit = overlap_count > 0
    precision = overlap_count / len(retrieved) if retrieved else 0.0
    recall = overlap_count / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hit": hit, "overlap": overlap_count}


K_VALUES = [1, 3, 5, 10, 20, 30, 40, 50]


def calculate_ir_metrics(expected: set, ranked_retrieved: list) -> dict:
    """Calculate comprehensive IR metrics for a single query."""
    metrics = {}
    if not expected or not ranked_retrieved:
        metrics["rr"] = 0.0
        for k in K_VALUES:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"hit@{k}"] = 0
        return metrics

    # Reciprocal Rank
    rr = 0.0
    for i, article in enumerate(ranked_retrieved):
        if article in expected:
            rr = 1.0 / (i + 1)
            break
    metrics["rr"] = rr

    for k in K_VALUES:
        top_k = ranked_retrieved[:k]
        top_k_set = set(top_k)
        relevant_in_k = len(expected & top_k_set)
        metrics[f"recall@{k}"] = relevant_in_k / len(expected)
        metrics[f"precision@{k}"] = relevant_in_k / k if k > 0 else 0.0
        metrics[f"hit@{k}"] = 1 if relevant_in_k > 0 else 0

        # NDCG@K
        dcg = 0.0
        for i, article in enumerate(top_k):
            if article in expected:
                dcg += 1.0 / math.log2(i + 2)
        idcg = 0.0
        num_relevant = min(len(expected), k)
        for i in range(num_relevant):
            idcg += 1.0 / math.log2(i + 2)
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def get_ablation_data(result) -> dict:
    """Extract all ablation-relevant data from result."""
    meta = result.metadata or {}
    query_analyzed = meta.get("query_analyzed", {})
    retrieval_strategy = meta.get("retrieval_strategy", {})
    ontology_exp = meta.get("ontology_expansion", [])

    tree_conf = get_tree_confidence(result)
    if tree_conf >= 0.7:
        tree_weight, kg_weight = 0.7, 0.3
    elif tree_conf >= 0.5:
        tree_weight, kg_weight = 0.5, 0.5
    else:
        tree_weight, kg_weight = 0.3, 0.7

    return {
        "intent": result.intent.value if result.intent else "",
        "keywords": query_analyzed.get("keywords", []),
        "article_refs_detected": query_analyzed.get("article_refs", []),
        "retrieval_method": retrieval_strategy.get("method", ""),
        "hybrid_alpha": retrieval_strategy.get("hybrid_alpha", 0),
        "max_hops": retrieval_strategy.get("max_hops", 0),
        "use_temporal": retrieval_strategy.get("use_temporal", False),
        "ontology_terms": [e.get("term", "") for e in ontology_exp],
        "ontology_classes": [e.get("class", "") for e in ontology_exp],
        "contexts_count": meta.get("contexts_retrieved", 0),
        "tree_weight": tree_weight,
        "kg_weight": kg_weight,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance on test set (config-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--test-file", default=None,
                        help="Test CSV file (overrides config; default: data/benchmark/legal-qa-benchmark.csv)")
    parser.add_argument("--start", type=int, default=1, help="Start from row number")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--stt-list", help="JSON file with list of STT numbers to test")
    parser.add_argument("--merge-with", help="Merge results with existing JSON file")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--ablation", help="Ablation config name (e.g., no_tree, no_reranker, dual_only)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-completed queries")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-query results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    print("=" * 70)
    print("VN LEGAL RAG - Evaluation (config-based)")
    print("=" * 70)

    # Load config
    print("\n[1/5] Loading config...")
    config = load_config(args.config)
    print(f"      Config: {args.config}")

    # Resolve paths from config
    kg_path = config.get("kg", {}).get("path", "data/kg_enhanced/legal_kg.json")
    chapter_summaries_path = config.get("kg", {}).get("chapter_summaries", "data/kg_enhanced/chapter_summaries.json")
    article_summaries_path = config.get("kg", {}).get("article_summaries", "data/kg_enhanced/article_summaries.json")
    document_summaries_path = config.get("kg", {}).get("document_summaries", "data/kg_enhanced/document_summaries.json")
    domain_groups_path = config.get("kg", {}).get("domain_groups", "data/kg_enhanced/domain_groups.json")
    db_path = config.get("database", {}).get("path", "data/legal_docs.db")
    test_file = args.test_file or config.get("benchmark", {}).get("path", "data/benchmark/legal-qa-benchmark.csv")

    # Load KG
    print("\n[2/5] Loading Knowledge Graph...")
    kg = load_kg(kg_path)
    print(f"      Entities: {len(kg.get('entities', []))}")

    # Load summaries
    print("\n[3/5] Loading summaries...")
    chapter_summaries = load_summaries(chapter_summaries_path) or {}
    article_summaries = load_summaries(article_summaries_path) or {}
    document_summaries_raw = load_summaries(document_summaries_path) or {}
    # Convert document_summaries dict to Loop 0 format (compact)
    document_summaries = []
    if isinstance(document_summaries_raw, dict):
        for doc_id, s in document_summaries_raw.items():
            domain = ", ".join(s.get("domain_keywords", [])[:15]) if s.get("domain_keywords") else ""
            document_summaries.append({
                "doc_id": s.get("doc_id", doc_id),
                "so_hieu": s.get("so_hieu", doc_id),
                "name": s.get("ten_van_ban", doc_id),
                "loai": s.get("loai_van_ban", ""),
                "domain": domain,
                "scope_preview": (s.get("scope", ""))[:200],
            })
    # Load domain groups
    domain_groups = load_summaries(domain_groups_path) or {}
    article_count = len(article_summaries.get("summaries", [])) if "summaries" in article_summaries else len(article_summaries) if isinstance(article_summaries, dict) else len(article_summaries)
    domain_count = len(domain_groups.get("domain_groups", {}))
    print(f"      Chapters: {len(chapter_summaries)}, Articles: {article_count}, Documents: {len(document_summaries)}, Domains: {domain_count}")

    # Build forest
    print("\n[4/5] Building forest from database...")
    forest = build_forest_from_db(db_path, chapter_summaries)
    stats = forest.stats()
    print(f"      Documents: {len(forest.trees)}, Nodes: {stats.total_nodes}")

    # Parse ablation config
    ablation_config = None
    if args.ablation and get_paper_ablation_configs:
        ablation_configs = get_paper_ablation_configs()
        if args.ablation in ablation_configs:
            ablation_config = ablation_configs[args.ablation]
            print(f"\n[!] Using ablation config: {args.ablation}")
            print(f"    enable_tree={ablation_config.enable_tree}, enable_reranker={ablation_config.enable_reranker}")
        else:
            print(f"\n[!] WARNING: Unknown ablation config '{args.ablation}'")
            print(f"    Available: {list(ablation_configs.keys())}")

    # Create GraphRAG via config
    print("\n[5/5] Initializing GraphRAG...")
    from vn_legal_rag.offline import LegalDocumentDB
    from vn_legal_rag.utils import create_llm_provider, create_embedding_provider

    db = LegalDocumentDB(db_path)
    llm_config = config.get("llm", {})
    llm_kwargs = {
        "provider": llm_config.get("provider", "anthropic"),
        "model": llm_config.get("model", "claude-3-5-haiku-20241022"),
    }
    if llm_config.get("use_cache", True):
        llm_kwargs["cache_db"] = llm_config.get("cache_db", "data/llm_cache.db")
    if llm_config.get("base_url"):
        llm_kwargs["base_url"] = llm_config["base_url"]
    llm_provider = create_llm_provider(**llm_kwargs)
    embedding_gen = create_embedding_provider()

    rag = create_legal_graphrag(
        kg=kg,
        forest=forest,
        db=db,
        llm_provider=llm_provider,
        embedding_gen=embedding_gen,
        article_summaries=article_summaries,
        document_summaries=document_summaries,
        domain_groups=domain_groups,
        config=config,
        ablation_config=ablation_config,
    )
    print("      GraphRAG initialized")

    # Load training data
    print("\n" + "=" * 70)
    print("Loading test data...")

    with open(test_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"Total rows: {total_rows}")
    print(f"Starting from: {args.start}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Workers: {args.workers}")
    print("=" * 70)
    print("\nPress Ctrl+C to stop at any time\n")

    # --- Stats tracking (thread-safe) ---
    stats_lock = Lock()
    hits = 0
    misses = 0
    skipped = 0
    category_stats = defaultdict(lambda: {"hits": 0, "total": 0})
    doc_stats = defaultdict(lambda: {"hits": 0, "total": 0})
    results = []
    processed_count = 0
    hits_at_k = {k: 0 for k in [5, 10, 20, 30, 40, 50]}
    json_results = []

    # --- Resume: load existing results if --resume and output file exists ---
    resumed_stts = set()
    if args.resume and args.output:
        output_path_check = args.output if args.output.endswith(".json") else f"{args.output}.json"
        if os.path.exists(output_path_check):
            print(f"\n[RESUME] Loading existing results from {output_path_check}...")
            with open(output_path_check, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_results = existing_data.get("results", [])
            for r in existing_results:
                stt = str(r.get("stt", ""))
                resumed_stts.add(stt)
                json_results.append(r)
                # Reconstruct stats from existing results
                is_hit = r.get("hit", False)
                cat = r.get("category", "")
                if is_hit:
                    hits += 1
                    category_stats[cat]["hits"] += 1
                else:
                    misses += 1
                category_stats[cat]["total"] += 1
                for k in [5, 10, 20, 30, 40, 50]:
                    if r.get("ir_metrics", {}).get(f"hit@{k}", 0) > 0:
                        hits_at_k[k] += 1
                # Reconstruct per-doc stats
                for aid in r.get("expected", []):
                    doc_id = extract_doc_id_from_article_id(aid)
                    doc_stats[doc_id]["total"] += 1
                    if is_hit:
                        doc_stats[doc_id]["hits"] += 1
                # Reconstruct category metric sums
                if "precision_sum" not in category_stats[cat]:
                    category_stats[cat]["precision_sum"] = 0.0
                    category_stats[cat]["recall_sum"] = 0.0
                    category_stats[cat]["f1_sum"] = 0.0
                    category_stats[cat]["rr_sum"] = 0.0
                    for kk in K_VALUES:
                        category_stats[cat][f"ndcg@{kk}_sum"] = 0.0
                category_stats[cat]["precision_sum"] += r.get("metrics", {}).get("precision", 0)
                category_stats[cat]["recall_sum"] += r.get("metrics", {}).get("recall", 0)
                category_stats[cat]["f1_sum"] += r.get("metrics", {}).get("f1", 0)
                category_stats[cat]["rr_sum"] += r.get("ir_metrics", {}).get("rr", 0)
                for kk in K_VALUES:
                    category_stats[cat][f"ndcg@{kk}_sum"] += r.get("ir_metrics", {}).get(f"ndcg@{kk}", 0)
                # Reconstruct results list for summary calculations
                results.append({
                    "stt": stt,
                    "category": cat,
                    "expected": set(r.get("expected", [])),
                    "tree_articles": set(r.get("tree_articles", [])),
                    "kg_articles": set(r.get("kg_articles", [])),
                    "retrieved": set(r.get("retrieved", [])),
                    "ranked_retrieved": r.get("ranked_retrieved", []),
                    "tree_hit": r.get("tree_hit", False),
                    "kg_hit": r.get("kg_hit", False),
                    "hit": is_hit,
                    "precision": r.get("metrics", {}).get("precision", 0),
                    "recall": r.get("metrics", {}).get("recall", 0),
                    "f1": r.get("metrics", {}).get("f1", 0),
                    "ir_metrics": r.get("ir_metrics", {}),
                    "tree_conf": r.get("tree_conf", 0),
                    "query_type": r.get("query_analysis", {}).get("query_type", "unknown"),
                    "intent": r.get("query_analysis", {}).get("intent", ""),
                    "retrieval_method": r.get("retrieval_strategy", {}).get("method", ""),
                    "has_ontology": r.get("ontology", {}).get("has_expansion", False),
                    "has_article_refs": len(r.get("query_analysis", {}).get("article_refs_detected", [])) > 0,
                })
                processed_count += 1
            print(f"[RESUME] Loaded {len(resumed_stts)} existing results, will skip them")
            total_tested = hits + misses
            if total_tested > 0:
                hit_rates_k = {k: (hits_at_k[k] / total_tested * 100) for k in [5, 10, 20, 30, 40, 50]}
                print(f"[RESUME] Current Hit@10: {hit_rates_k[10]:.1f}%, MRR: {sum(r.get('ir_metrics', {}).get('rr', 0) for r in results) / len(results):.4f}")

    def _save_incremental():
        """Save current results to output file incrementally."""
        if not args.output:
            return
        output_path = args.output if args.output.endswith(".json") else f"{args.output}.json"
        total_tested_now = hits + misses
        hit_rate_now = (hits / total_tested_now * 100) if total_tested_now > 0 else 0
        avg_mrr_now = sum(r.get("ir_metrics", {}).get("rr", 0) for r in results) / len(results) if results else 0

        avg_ir_now = {}
        for k in K_VALUES:
            avg_ir_now[f"hit@{k}"] = sum(r.get("ir_metrics", {}).get(f"hit@{k}", 0) for r in results) / len(results) if results else 0
            avg_ir_now[f"recall@{k}"] = sum(r.get("ir_metrics", {}).get(f"recall@{k}", 0) for r in results) / len(results) if results else 0
            avg_ir_now[f"precision@{k}"] = sum(r.get("ir_metrics", {}).get(f"precision@{k}", 0) for r in results) / len(results) if results else 0
            avg_ir_now[f"ndcg@{k}"] = sum(r.get("ir_metrics", {}).get(f"ndcg@{k}", 0) for r in results) / len(results) if results else 0

        summary_now = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tested": total_tested_now,
                "skipped": skipped,
                "config": args.config,
                "test_file": test_file,
                "ablation": args.ablation,
                "incremental": True,
            },
            "overall_metrics": {
                "hit_rate": avg_ir_now.get("hit@10", 0),
                "hit_rate_all": hit_rate_now / 100,
                "precision": sum(r.get("precision", 0) for r in results) / len(results) if results else 0,
                "recall": sum(r.get("recall", 0) for r in results) / len(results) if results else 0,
                "f1": sum(r.get("f1", 0) for r in results) / len(results) if results else 0,
                "mrr": avg_mrr_now,
            },
            "ir_metrics_at_k": {
                str(k): {
                    "hit": avg_ir_now[f"hit@{k}"],
                    "recall": avg_ir_now[f"recall@{k}"],
                    "precision": avg_ir_now[f"precision@{k}"],
                    "ndcg": avg_ir_now[f"ndcg@{k}"],
                }
                for k in K_VALUES
            },
        }

        output_data = {"summary": summary_now, "results": json_results}
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Write to temp file first, then rename for atomicity
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)

    def process_single_question(task_data: dict) -> dict:
        """Process a single question with retry on rate limit."""
        row = task_data["row"]
        row_num = task_data["row_num"]

        stt = row.get("STT", str(row_num))
        category = row.get("Category", "")
        question = row.get("Content", "") or row.get("question", "")
        article_ids = row.get("Article_IDs", "") or row.get("article_ids", "")
        expected = extract_expected_article_ids(article_ids)

        result_data = {
            "stt": stt,
            "category": category,
            "question": question,
            "expected": expected,
            "skipped": False,
            "error": None,
        }

        if not expected:
            result_data["skipped"] = True
            return result_data

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                result = rag.query(question, adaptive_retrieval=True)
                result_data["time_seconds"] = round(time.time() - t0, 2)
                result_data["tree_articles"] = extract_tree_articles(result)
                result_data["kg_articles"] = extract_kg_articles(result)
                result_data["ranked_retrieved"] = extract_retrieved_articles_ranked(result)
                result_data["tree_conf"] = get_tree_confidence(result)
                result_data["tree_reasoning"] = get_tree_reasoning(result)
                result_data["query_type"] = result.query_type.value if result.query_type else "unknown"
                result_data["ablation"] = get_ablation_data(result)
                return result_data
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str or "429" in error_str or "quota" in error_str:
                    delay = base_delay * (2 ** attempt)
                    print(f"       [Rate limit] STT {stt} - Waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    result_data["error"] = str(e)
                    return result_data

        result_data["error"] = "Max retries exceeded due to rate limit"
        return result_data

    def update_stats_and_print(result_data: dict):
        """Update global stats and print result (thread-safe)."""
        nonlocal hits, misses, skipped, processed_count

        stt = result_data["stt"]
        category = result_data["category"]
        expected = result_data["expected"]

        with stats_lock:
            processed_count += 1
            local_processed = processed_count

            if result_data["skipped"]:
                skipped += 1
                print(f"[{local_processed}] STT {stt} - SKIPPED (no expected articles)")
                return

            if result_data.get("error"):
                misses += 1
                category_stats[category]["total"] += 1
                print(f"[{local_processed}] STT {stt} - ERROR: {result_data['error']}")
                return

            tree_articles = result_data["tree_articles"]
            kg_articles = result_data["kg_articles"]
            retrieved = tree_articles | kg_articles
            ranked_retrieved = result_data.get("ranked_retrieved", [])
            tree_conf = result_data["tree_conf"]
            tree_reasoning = result_data["tree_reasoning"]
            query_type = result_data["query_type"]
            ablation = result_data["ablation"]

            tree_hit = bool(expected & tree_articles)
            kg_hit = bool(expected & kg_articles)
            is_hit = bool(expected & set(ranked_retrieved[:10]))

            metrics = calculate_metrics(expected, retrieved)
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1"]

            ir_metrics = calculate_ir_metrics(expected, ranked_retrieved)

            if is_hit:
                hits += 1
                category_stats[category]["hits"] += 1
                status = "HIT"
            else:
                misses += 1
                status = "MISS"

            for k in [5, 10, 20, 30, 40, 50]:
                if bool(expected & set(ranked_retrieved[:k])):
                    hits_at_k[k] += 1

            category_stats[category]["total"] += 1

            # Per-document tracking
            expected_docs = set(extract_doc_id_from_article_id(aid) for aid in expected)
            for doc_id in expected_docs:
                doc_stats[doc_id]["total"] += 1
                if is_hit:
                    doc_stats[doc_id]["hits"] += 1
            if "precision_sum" not in category_stats[category]:
                category_stats[category]["precision_sum"] = 0.0
                category_stats[category]["recall_sum"] = 0.0
                category_stats[category]["f1_sum"] = 0.0
                category_stats[category]["rr_sum"] = 0.0
                for k in K_VALUES:
                    category_stats[category][f"ndcg@{k}_sum"] = 0.0
            category_stats[category]["precision_sum"] += precision
            category_stats[category]["recall_sum"] += recall
            category_stats[category]["f1_sum"] += f1
            category_stats[category]["rr_sum"] += ir_metrics["rr"]
            for k in K_VALUES:
                category_stats[category][f"ndcg@{k}_sum"] += ir_metrics[f"ndcg@{k}"]

            total_tested = hits + misses
            hit_rates_k = {k: (hits_at_k[k] / total_tested * 100) if total_tested > 0 else 0 for k in [5, 10, 20, 30, 40, 50]}

            tree_status = "T✓" if tree_hit else "T✗"
            kg_status = "K✓" if kg_hit else "K✗"
            hit_str = f"@5:{hit_rates_k[5]:4.0f}% @10:{hit_rates_k[10]:4.0f}% @20:{hit_rates_k[20]:4.0f}% @30:{hit_rates_k[30]:4.0f}% @40:{hit_rates_k[40]:4.0f}%"

            elapsed = result_data.get("time_seconds", 0)
            print(f"[{local_processed:4d}] STT {stt:4s} | {status:4s} | {tree_status} {kg_status} | {elapsed:5.1f}s | {hit_str}")

            result_record = {
                "stt": stt,
                "category": category,
                "question": result_data["question"],
                "expected": sorted(expected),
                "tree_articles": sorted(tree_articles),
                "kg_articles": sorted(kg_articles),
                "retrieved": sorted(retrieved),
                "ranked_retrieved": ranked_retrieved,
                "tree_conf": tree_conf,
                "tree_weight": ablation["tree_weight"],
                "kg_weight": ablation["kg_weight"],
                "tree_hit": tree_hit,
                "kg_hit": kg_hit,
                "hit": is_hit,
                "metrics": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
                "ir_metrics": ir_metrics,
                "query_analysis": {
                    "query_type": query_type,
                    "intent": ablation["intent"],
                    "keywords": ablation["keywords"],
                    "article_refs_detected": ablation["article_refs_detected"],
                },
                "retrieval_strategy": {
                    "method": ablation["retrieval_method"],
                    "hybrid_alpha": ablation["hybrid_alpha"],
                    "max_hops": ablation["max_hops"],
                },
                "ontology": {
                    "terms": ablation["ontology_terms"],
                    "classes": ablation["ontology_classes"],
                    "has_expansion": len(ablation["ontology_terms"]) > 0,
                },
                "contexts_count": ablation["contexts_count"],
                "tree_reasoning": tree_reasoning,
                "time_seconds": result_data.get("time_seconds", 0),
            }
            json_results.append(result_record)
            _save_incremental()

            results.append({
                "stt": stt,
                "category": category,
                "expected": expected,
                "tree_articles": tree_articles,
                "kg_articles": kg_articles,
                "retrieved": retrieved,
                "ranked_retrieved": ranked_retrieved,
                "tree_hit": tree_hit,
                "kg_hit": kg_hit,
                "hit": is_hit,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ir_metrics": ir_metrics,
                "tree_conf": tree_conf,
                "query_type": query_type,
                "intent": ablation["intent"],
                "retrieval_method": ablation["retrieval_method"],
                "has_ontology": len(ablation["ontology_terms"]) > 0,
                "has_article_refs": len(ablation["article_refs_detected"]) > 0,
            })

    # Load STT list if provided
    stt_filter = None
    if args.stt_list:
        print(f"\nLoading STT list from {args.stt_list}...")
        with open(args.stt_list, "r") as f:
            stt_data = json.load(f)
            if isinstance(stt_data, list):
                stt_filter = set(str(s) for s in stt_data)
            elif isinstance(stt_data, dict) and 'failed' in stt_data:
                stt_filter = set(str(r['stt']) for r in stt_data['failed'])
            else:
                stt_filter = set(str(s) for s in stt_data.get('stt_list', []))
        print(f"Will test only {len(stt_filter)} specific questions")

    # Prepare tasks
    tasks = []
    for i, row in enumerate(rows):
        row_num = i + 1
        stt = row.get("STT", str(row_num))
        if stt_filter and str(stt) not in stt_filter:
            continue
        if row_num < args.start:
            continue
        if resumed_stts and str(stt) in resumed_stts:
            continue
        if args.limit and len(tasks) >= args.limit:
            break
        tasks.append({"row": row, "row_num": row_num})

    try:
        if args.workers == 1:
            for task in tasks:
                result_data = process_single_question(task)
                update_stats_and_print(result_data)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_single_question, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        result_data = future.result()
                        update_stats_and_print(result_data)
                    except Exception as e:
                        task = futures[future]
                        stt = task["row"].get("STT", "?")
                        print(f"[?] STT {stt} - EXECUTOR ERROR: {e}")
                        with stats_lock:
                            misses += 1
    except KeyboardInterrupt:
        print("\n\n*** INTERRUPTED BY USER ***")

    # -----------------------------------------------------------------------
    # Final summary (identical to run-full-training-test.py)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (ABLATION)")
    print("=" * 70)

    total_tested = hits + misses
    hit_rate = (hits / total_tested * 100) if total_tested > 0 else 0

    avg_precision = sum(r.get("precision", 0) for r in results) / len(results) if results else 0
    avg_recall = sum(r.get("recall", 0) for r in results) / len(results) if results else 0
    avg_f1 = sum(r.get("f1", 0) for r in results) / len(results) if results else 0

    tree_hits = sum(1 for r in results if r.get("tree_hit"))
    kg_hits = sum(1 for r in results if r.get("kg_hit"))
    both_hits = sum(1 for r in results if r.get("tree_hit") and r.get("kg_hit"))
    tree_only = sum(1 for r in results if r.get("tree_hit") and not r.get("kg_hit"))
    kg_only = sum(1 for r in results if r.get("kg_hit") and not r.get("tree_hit"))
    neither = sum(1 for r in results if not r.get("tree_hit") and not r.get("kg_hit"))

    if results:
        avg_mrr = sum(r.get("ir_metrics", {}).get("rr", 0) for r in results) / len(results)
        avg_ir = {}
        for k in K_VALUES:
            avg_ir[f"recall@{k}"] = sum(r.get("ir_metrics", {}).get(f"recall@{k}", 0) for r in results) / len(results)
            avg_ir[f"precision@{k}"] = sum(r.get("ir_metrics", {}).get(f"precision@{k}", 0) for r in results) / len(results)
            avg_ir[f"ndcg@{k}"] = sum(r.get("ir_metrics", {}).get(f"ndcg@{k}", 0) for r in results) / len(results)
            avg_ir[f"hit@{k}"] = sum(r.get("ir_metrics", {}).get(f"hit@{k}", 0) for r in results) / len(results)
    else:
        avg_mrr = 0
        avg_ir = {f"{m}@{k}": 0 for k in K_VALUES for m in ["recall", "precision", "ndcg", "hit"]}

    hit_at_10 = avg_ir.get("hit@10", 0) * 100 if results else 0

    print(f"\n--- Overall Metrics ---")
    print(f"Hit@10:    {hit_at_10:.1f}% (primary metric - top 10 contexts for RAG)")
    print(f"Hit@all:   {hits}/{total_tested} ({hit_rate:.1f}%) (unbounded)")
    print(f"Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"Recall:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(f"F1 Score:  {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print(f"MRR:       {avg_mrr:.4f}")
    print(f"Skipped:   {skipped}")

    print(f"\n--- IR Metrics @ K ---")
    print(f"{'K':>3} {'Hit@K':>10} {'Recall@K':>10} {'Prec@K':>10} {'NDCG@K':>10}")
    print(f"{'-'*3} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for k in K_VALUES:
        print(f"{k:>3} {avg_ir[f'hit@{k}']*100:>9.1f}% {avg_ir[f'recall@{k}']:>10.4f} {avg_ir[f'precision@{k}']:>10.4f} {avg_ir[f'ndcg@{k}']:>10.4f}")

    print(f"\n--- ABLATION: Component-wise IR Metrics ---")
    if results:
        tree_metrics = [calculate_metrics(r["expected"], r["tree_articles"]) for r in results]
        tree_avg_prec = sum(m["precision"] for m in tree_metrics) / len(tree_metrics)
        tree_avg_recall = sum(m["recall"] for m in tree_metrics) / len(tree_metrics)
        tree_avg_f1 = sum(m["f1"] for m in tree_metrics) / len(tree_metrics)

        kg_metrics = [calculate_metrics(r["expected"], r["kg_articles"]) for r in results]
        kg_avg_prec = sum(m["precision"] for m in kg_metrics) / len(kg_metrics)
        kg_avg_recall = sum(m["recall"] for m in kg_metrics) / len(kg_metrics)
        kg_avg_f1 = sum(m["f1"] for m in kg_metrics) / len(kg_metrics)

        tree_ir = [calculate_ir_metrics(r["expected"], list(r["tree_articles"])) for r in results]
        tree_mrr = sum(m["rr"] for m in tree_ir) / len(tree_ir)

        kg_ir = [calculate_ir_metrics(r["expected"], list(r["kg_articles"])) for r in results]
        kg_mrr = sum(m["rr"] for m in kg_ir) / len(kg_ir)

        print(f"\n{'Component':<10} {'Hit%':>7} {'Prec':>8} {'Recall':>8} {'F1':>8} {'MRR':>8}")
        print(f"{'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'Tree':<10} {tree_hits/len(results)*100:>6.1f}% {tree_avg_prec:>8.4f} {tree_avg_recall:>8.4f} {tree_avg_f1:>8.4f} {tree_mrr:>8.4f}")
        print(f"{'KG':<10} {kg_hits/len(results)*100:>6.1f}% {kg_avg_prec:>8.4f} {kg_avg_recall:>8.4f} {kg_avg_f1:>8.4f} {kg_mrr:>8.4f}")
        print(f"{'Merged':<10} {hit_rate:>6.1f}% {avg_precision:>8.4f} {avg_recall:>8.4f} {avg_f1:>8.4f} {avg_mrr:>8.4f}")

        print(f"\n--- ABLATION: IR@K by Component ---")
        for k in K_VALUES:
            tree_recall_k = sum(m[f"recall@{k}"] for m in tree_ir) / len(tree_ir)
            tree_ndcg_k = sum(m[f"ndcg@{k}"] for m in tree_ir) / len(tree_ir)
            kg_recall_k = sum(m[f"recall@{k}"] for m in kg_ir) / len(kg_ir)
            kg_ndcg_k = sum(m[f"ndcg@{k}"] for m in kg_ir) / len(kg_ir)
            print(f"  @{k}: Tree R={tree_recall_k:.3f} NDCG={tree_ndcg_k:.3f} | KG R={kg_recall_k:.3f} NDCG={kg_ndcg_k:.3f} | Merged R={avg_ir[f'recall@{k}']:.3f} NDCG={avg_ir[f'ndcg@{k}']:.3f}")

        print(f"\n--- Agreement Analysis ---")
        print(f"  Both hit:   {both_hits:4d} ({both_hits/len(results)*100:.1f}%)")
        print(f"  Tree only:  {tree_only:4d} ({tree_only/len(results)*100:.1f}%)")
        print(f"  KG only:    {kg_only:4d} ({kg_only/len(results)*100:.1f}%)")
        print(f"  Neither:    {neither:4d} ({neither/len(results)*100:.1f}%)")

    # Ablation helper functions
    def calc_group_metrics(group: list) -> dict:
        if not group:
            return {"n": 0, "hit": 0, "prec": 0, "recall": 0, "f1": 0, "mrr": 0, "ndcg5": 0}
        n = len(group)
        h = sum(1 for r in group if r.get("hit"))
        prec = sum(r.get("precision", 0) for r in group) / n
        rec = sum(r.get("recall", 0) for r in group) / n
        f1 = sum(r.get("f1", 0) for r in group) / n
        mrr = sum(r.get("ir_metrics", {}).get("rr", 0) for r in group) / n
        ndcg5 = sum(r.get("ir_metrics", {}).get("ndcg@5", 0) for r in group) / n
        return {"n": n, "hit": h/n, "prec": prec, "recall": rec, "f1": f1, "mrr": mrr, "ndcg5": ndcg5}

    def print_ablation_table(title: str, groups: dict):
        print(f"\n--- ABLATION: {title} ---")
        print(f"  {'Group':<25} {'N':>5} {'Hit%':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'MRR':>7} {'NDCG@5':>7}")
        print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for name, group in groups.items():
            m = calc_group_metrics(group)
            if m["n"] > 0:
                print(f"  {name:<25} {m['n']:>5} {m['hit']*100:>6.1f}% {m['prec']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} {m['mrr']:>7.3f} {m['ndcg5']:>7.3f}")

    # ABLATION 1: By Query Type
    query_type_groups = defaultdict(list)
    for r in results:
        query_type_groups[r.get("query_type", "unknown")].append(r)
    print_ablation_table("By Query Type", dict(query_type_groups))

    # ABLATION 2: By Intent
    intent_groups = defaultdict(list)
    for r in results:
        intent_groups[r.get("intent", "unknown")].append(r)
    print_ablation_table("By Intent", dict(intent_groups))

    # ABLATION 3: By Retrieval Method
    method_groups = defaultdict(list)
    for r in results:
        method_groups[r.get("retrieval_method", "unknown")].append(r)
    print_ablation_table("By Retrieval Method", dict(method_groups))

    # ABLATION 4: Ontology Expansion Impact
    print_ablation_table("Ontology Expansion", {
        "With Ontology": [r for r in results if r.get("has_ontology")],
        "Without Ontology": [r for r in results if not r.get("has_ontology")],
    })

    # ABLATION 5: Article Reference Detection
    print_ablation_table("Article Reference Detection", {
        "Has Article Refs": [r for r in results if r.get("has_article_refs")],
        "No Article Refs": [r for r in results if not r.get("has_article_refs")],
    })

    # ABLATION 6: By Tree Confidence Level
    print_ablation_table("Tree Confidence Level", {
        "High (>=0.7)": [r for r in results if r.get("tree_conf", 0) >= 0.7],
        "Medium (0.5-0.7)": [r for r in results if 0.5 <= r.get("tree_conf", 0) < 0.7],
        "Low (<0.5)": [r for r in results if r.get("tree_conf", 0) < 0.5],
    })

    # ABLATION 7: Tree Hit vs KG Hit Analysis
    print_ablation_table("Component Hit Pattern", {
        "Both Hit": [r for r in results if r.get("tree_hit") and r.get("kg_hit")],
        "Tree Only": [r for r in results if r.get("tree_hit") and not r.get("kg_hit")],
        "KG Only": [r for r in results if r.get("kg_hit") and not r.get("tree_hit")],
        "Neither": [r for r in results if not r.get("tree_hit") and not r.get("kg_hit")],
    })

    # ABLATION 8: By Number of Expected Articles
    print_ablation_table("Expected Answer Count", {
        "Single Answer": [r for r in results if len(r.get("expected", set())) == 1],
        "2-3 Answers": [r for r in results if 2 <= len(r.get("expected", set())) <= 3],
        "4+ Answers": [r for r in results if len(r.get("expected", set())) >= 4],
    })

    # ABLATION 9: By Retrieved Count
    print_ablation_table("Retrieval Volume", {
        "0 Retrieved": [r for r in results if len(r.get("retrieved", set())) == 0],
        "1-3 Retrieved": [r for r in results if 1 <= len(r.get("retrieved", set())) <= 3],
        "4-10 Retrieved": [r for r in results if 4 <= len(r.get("retrieved", set())) <= 10],
        "10+ Retrieved": [r for r in results if len(r.get("retrieved", set())) > 10],
    })

    # By Category
    print("\n--- By Category ---")
    print(f"  {'Category':<40} {'Hit':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'MRR':>7} {'NDCG@5':>7}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cat, cstats in sorted(category_stats.items()):
        cat_rate = (cstats["hits"] / cstats["total"] * 100) if cstats["total"] > 0 else 0
        cat_prec = (cstats.get("precision_sum", 0) / cstats["total"]) if cstats["total"] > 0 else 0
        cat_recall = (cstats.get("recall_sum", 0) / cstats["total"]) if cstats["total"] > 0 else 0
        cat_f1 = (cstats.get("f1_sum", 0) / cstats["total"]) if cstats["total"] > 0 else 0
        cat_mrr = (cstats.get("rr_sum", 0) / cstats["total"]) if cstats["total"] > 0 else 0
        cat_ndcg5 = (cstats.get("ndcg@5_sum", 0) / cstats["total"]) if cstats["total"] > 0 else 0
        cat_short = cat[:40] + "..." if len(cat) > 40 else cat
        print(f"  {cat_short:<40} {cat_rate:>6.0f}% {cat_prec:>7.2f} {cat_recall:>7.2f} {cat_f1:>7.2f} {cat_mrr:>7.2f} {cat_ndcg5:>7.2f}")

    # Per-Document Breakdown
    if doc_stats:
        print("\n--- Per-Document Breakdown ---")
        print(f"  {'Document ID':<25} {'Queries':>8} {'Hits':>6} {'Hit%':>7}")
        print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*7}")
        for doc_id, dstats in sorted(doc_stats.items()):
            doc_rate = (dstats["hits"] / dstats["total"] * 100) if dstats["total"] > 0 else 0
            print(f"  {doc_id:<25} {dstats['total']:>8} {dstats['hits']:>6} {doc_rate:>6.1f}%")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results to JSON (identical format to run-full-training-test.py)
    # -----------------------------------------------------------------------
    if args.output:
        summary = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tested": total_tested,
                "skipped": skipped,
                "config": args.config,
                "test_file": test_file,
                "ablation": args.ablation,
            },
            "overall_metrics": {
                "hit_rate": avg_ir.get("hit@10", 0),
                "hit_rate_all": hit_rate / 100,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "mrr": avg_mrr,
            },
            "ir_metrics_at_k": {
                str(k): {
                    "hit": avg_ir[f"hit@{k}"],
                    "recall": avg_ir[f"recall@{k}"],
                    "precision": avg_ir[f"precision@{k}"],
                    "ndcg": avg_ir[f"ndcg@{k}"],
                }
                for k in K_VALUES
            },
            "component_breakdown": {
                "tree_hits": tree_hits,
                "kg_hits": kg_hits,
                "both_hit": both_hits,
                "tree_only": tree_only,
                "kg_only": kg_only,
                "neither": neither,
            },
            "per_document": {
                doc_id: {
                    "total": dstats["total"],
                    "hits": dstats["hits"],
                    "hit_rate": dstats["hits"] / dstats["total"] if dstats["total"] > 0 else 0,
                }
                for doc_id, dstats in doc_stats.items()
            },
        }

        # Merge with existing results if specified
        final_results = json_results
        if args.merge_with and os.path.exists(args.merge_with):
            print(f"\nMerging with existing results from {args.merge_with}...")
            with open(args.merge_with, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_results = existing.get("results", [])

            new_results_map = {str(r['stt']): r for r in json_results}

            merged = []
            replaced = 0
            for r in existing_results:
                stt = str(r['stt'])
                if stt in new_results_map:
                    merged.append(new_results_map[stt])
                    replaced += 1
                else:
                    merged.append(r)

            final_results = merged
            print(f"  Replaced {replaced} results, kept {len(merged) - replaced} existing")

            # Recalculate summary metrics from merged results
            total_tested = len(final_results)
            hits = sum(1 for r in final_results if r.get('hit', False))
            misses = total_tested - hits
            hit_rate = (hits / total_tested * 100) if total_tested > 0 else 0

            avg_precision = sum(r.get("metrics", {}).get("precision", 0) for r in final_results) / total_tested if total_tested else 0
            avg_recall = sum(r.get("metrics", {}).get("recall", 0) for r in final_results) / total_tested if total_tested else 0
            avg_f1 = sum(r.get("metrics", {}).get("f1", 0) for r in final_results) / total_tested if total_tested else 0
            avg_mrr = sum(r.get("ir_metrics", {}).get("rr", 0) for r in final_results) / total_tested if total_tested else 0

            tree_hits = sum(1 for r in final_results if r.get("tree_hit"))
            kg_hits = sum(1 for r in final_results if r.get("kg_hit"))
            both_hits = sum(1 for r in final_results if r.get("tree_hit") and r.get("kg_hit"))
            tree_only = sum(1 for r in final_results if r.get("tree_hit") and not r.get("kg_hit"))
            kg_only = sum(1 for r in final_results if r.get("kg_hit") and not r.get("tree_hit"))
            neither = sum(1 for r in final_results if not r.get("tree_hit") and not r.get("kg_hit"))

            merged_ir = {}
            for k in K_VALUES:
                merged_ir[f"hit@{k}"] = sum(r.get("ir_metrics", {}).get(f"hit@{k}", 0) for r in final_results) / total_tested
                merged_ir[f"recall@{k}"] = sum(r.get("ir_metrics", {}).get(f"recall@{k}", 0) for r in final_results) / total_tested
                merged_ir[f"precision@{k}"] = sum(r.get("ir_metrics", {}).get(f"precision@{k}", 0) for r in final_results) / total_tested
                merged_ir[f"ndcg@{k}"] = sum(r.get("ir_metrics", {}).get(f"ndcg@{k}", 0) for r in final_results) / total_tested

            summary["metadata"]["total_tested"] = total_tested
            summary["metadata"]["merged_from"] = args.merge_with
            summary["overall_metrics"] = {
                "hit_rate": merged_ir.get("hit@10", 0),
                "hit_rate_all": hit_rate / 100,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "mrr": avg_mrr,
            }
            summary["ir_metrics_at_k"] = {
                str(k): {
                    "hit": merged_ir[f"hit@{k}"],
                    "recall": merged_ir[f"recall@{k}"],
                    "precision": merged_ir[f"precision@{k}"],
                    "ndcg": merged_ir[f"ndcg@{k}"],
                }
                for k in K_VALUES
            }
            summary["component_breakdown"] = {
                "tree_hits": tree_hits,
                "kg_hits": kg_hits,
                "both_hit": both_hits,
                "tree_only": tree_only,
                "kg_only": kg_only,
                "neither": neither,
            }

            print(f"  Merged metrics: Hit@10 {merged_ir.get('hit@10', 0)*100:.1f}%, MRR {avg_mrr:.4f}, Precision {avg_precision:.4f}")

        output_data = {
            "summary": summary,
            "results": final_results,
        }

        output_path = args.output if args.output.endswith(".json") else f"{args.output}.json"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")
        print(f"  - {len(final_results)} query results")
        print(f"  - Summary with {len(summary)} sections")


if __name__ == "__main__":
    main()
