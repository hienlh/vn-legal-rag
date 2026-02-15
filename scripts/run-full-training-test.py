#!/usr/bin/env python3
"""
Full training data test with real-time statistics.

Runs VN Legal RAG Enhanced on all training questions and shows
running statistics after each question.

Default: Multi-document mode with 3-loop tree traversal (Loop 0 + Loop 1 + Loop 2)

Usage:
    # Multi-document mode (default)
    python scripts/run-full-training-test.py

    # Single-document mode (original behavior)
    python scripts/run-full-training-test.py --single-document
    python scripts/run-full-training-test.py --document 59-2020-QH14

    # Start from specific row
    python scripts/run-full-training-test.py --start 100

    # Limit number of questions
    python scripts/run-full-training-test.py --limit 50

    # Save results to JSON
    python scripts/run-full-training-test.py --output results.json

    # Run with parallel workers
    python scripts/run-full-training-test.py --workers 4
"""

import argparse
import csv
import json
import logging
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

from vn_legal_rag import (
    LegalDocumentDB,
    LegalGraphRAG,
    LegalDocumentModel,
    AblationConfig,
    get_paper_ablation_configs,
)
from vn_legal_rag.types import (
    TreeNode,
    TreeIndex,
    NodeType,
    UnifiedForest,
)
from vn_legal_rag.utils import ProgressTracker

# Restore stderr and disable progress tracker
sys.stderr = _stderr
ProgressTracker.get_instance().disable()


def extract_article_numbers(article_ids: str) -> set:
    """Extract article numbers from Article_IDs column."""
    if not article_ids or article_ids.strip() == "":
        return set()

    articles = set()
    # Format: 59-2020-QH14:d206:k1 or 59-2020-QH14:d206
    pattern = r":d(\d+)"

    for match in re.finditer(pattern, article_ids):
        articles.add(int(match.group(1)))

    return articles


def load_chapter_summaries(path: Path) -> dict:
    """Load chapter summaries from JSON.

    Supports two formats:
    - Legacy dict: {chapter_id: {description: ...}, ...}
    - New list format: {summaries: [{chapter_id: ..., description: ...}, ...]}
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # New format: {summaries: [...], processed_chapters: [...], stats: {...}}
    if isinstance(data, dict) and "summaries" in data and isinstance(data["summaries"], list):
        return {
            item.get("chapter_id", ""): item.get("description", "")
            for item in data["summaries"]
            if item.get("chapter_id")
        }

    # Legacy format: {chapter_id: {description: ...}}
    return {
        chapter_id: summary.get("description", "") if isinstance(summary, dict) else str(summary)
        for chapter_id, summary in data.items()
    }


def load_article_summaries(path: Path) -> dict:
    """Load article summaries from JSON for loop 2 selection."""
    if not path.exists():
        print(f"  WARNING: Article summaries not found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_document_summaries(path: Path) -> list:
    """Load document summaries from JSON for loop 0 selection."""
    if not path.exists():
        print(f"  WARNING: Document summaries not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tree_from_db(db: LegalDocumentDB, doc_id: str, chapter_summaries: dict) -> TreeIndex:
    """Build tree index from database."""
    with db.SessionLocal() as session:
        doc = session.query(LegalDocumentModel).filter(
            LegalDocumentModel.id == doc_id
        ).first()

        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        sub_nodes = []

        for chapter in sorted(doc.chapters, key=lambda c: c.position):
            chapter_children = []

            for article in sorted(chapter.articles, key=lambda a: a.position):
                article_node = TreeNode(
                    node_id=article.id,
                    node_type=NodeType.ARTICLE,
                    name=article.title or f"Điều {article.article_number}",
                    description="",
                    content=article.raw_text[:500] if article.raw_text else "",
                    metadata={"article_number": article.article_number},
                    sub_nodes=[],
                )
                chapter_children.append(article_node)

            for section in sorted(chapter.sections, key=lambda s: s.position):
                section_articles = []
                for article in sorted(section.articles, key=lambda a: a.position):
                    article_node = TreeNode(
                        node_id=article.id,
                        node_type=NodeType.ARTICLE,
                        name=article.title or f"Điều {article.article_number}",
                        description="",
                        content=article.raw_text[:500] if article.raw_text else "",
                        metadata={"article_number": article.article_number},
                        sub_nodes=[],
                    )
                    section_articles.append(article_node)

                if section_articles:
                    section_node = TreeNode(
                        node_id=section.id,
                        node_type=NodeType.SECTION,
                        name=section.title or f"Mục {section.section_number}",
                        description="",
                        content="",
                        metadata={"section_number": section.section_number},
                        sub_nodes=section_articles,
                    )
                    chapter_children.append(section_node)

            if chapter_children:
                all_nums = []
                for child in chapter_children:
                    if child.node_type == NodeType.ARTICLE:
                        all_nums.append(child.metadata.get("article_number", 0))
                    elif child.node_type == NodeType.SECTION:
                        for a in child.sub_nodes:
                            all_nums.append(a.metadata.get("article_number", 0))

                article_range = ""
                if all_nums:
                    all_nums.sort()
                    article_range = f"Điều {all_nums[0]}-{all_nums[-1]}"

                description = chapter_summaries.get(chapter.id, article_range)

                chapter_node = TreeNode(
                    node_id=chapter.id,
                    node_type=NodeType.CHAPTER,
                    name=chapter.title or f"Chương {chapter.chapter_number}",
                    description=description,
                    content="",
                    metadata={
                        "chapter_number": chapter.chapter_number,
                        "article_range": article_range,
                    },
                    sub_nodes=chapter_children,
                )
                sub_nodes.append(chapter_node)

        root = TreeNode(
            node_id=doc.id,
            node_type=NodeType.DOCUMENT,
            name=doc.title,
            description=f"{doc.loai_van_ban} số {doc.so_hieu}",
            content="",
            metadata={"so_hieu": doc.so_hieu},
            sub_nodes=sub_nodes,
        )

        return TreeIndex(root=root, doc_id=doc.id)


def build_forest_from_db(db: LegalDocumentDB, chapter_summaries: dict, doc_ids: list = None) -> UnifiedForest:
    """Build forest with multiple documents from database.

    Args:
        db: Database connection
        chapter_summaries: Chapter keywords for all documents
        doc_ids: List of document IDs to include (None = all)

    Returns:
        UnifiedForest with all documents
    """
    forest = UnifiedForest()

    with db.SessionLocal() as session:
        query = session.query(LegalDocumentModel)
        if doc_ids:
            query = query.filter(LegalDocumentModel.id.in_(doc_ids))
        docs = query.all()

        for doc in docs:
            try:
                tree = build_tree_from_db(db, doc.id, chapter_summaries)
                forest.add_tree(tree)
            except Exception as e:
                print(f"  WARNING: Failed to build tree for {doc.id}: {e}")

    return forest


def extract_tree_articles(result) -> set:
    """Extract article numbers from tree search result only."""
    articles = set()
    if result.tree_search_result:
        for node in result.tree_search_result.target_nodes:
            if node.node_type == NodeType.ARTICLE:
                num = node.metadata.get("article_number")
                if num:
                    articles.add(int(num) if isinstance(num, str) else num)
    return articles


def extract_tree_articles_ranked(result) -> list:
    """Extract article numbers from tree search result in ranked order."""
    articles = []
    seen = set()
    if result.tree_search_result:
        for node in result.tree_search_result.target_nodes:
            if node.node_type == NodeType.ARTICLE:
                num = node.metadata.get("article_number")
                if num:
                    num = int(num) if isinstance(num, str) else num
                    if num not in seen:
                        articles.append(num)
                        seen.add(num)
    return articles


def extract_kg_articles(result) -> set:
    """Extract article numbers from KG/citations only."""
    articles = set()
    if result.citations:
        for cite in result.citations:
            cite_str = cite.get("citation_string", "")
            match = re.search(r"Điều (\d+)", cite_str)
            if match:
                articles.add(int(match.group(1)))
    return articles


def extract_kg_articles_ranked(result) -> list:
    """Extract article numbers from KG/citations in order."""
    articles = []
    seen = set()
    if result.citations:
        for cite in result.citations:
            cite_str = cite.get("citation_string", "")
            match = re.search(r"Điều (\d+)", cite_str)
            if match:
                num = int(match.group(1))
                if num not in seen:
                    articles.append(num)
                    seen.add(num)
    return articles


def extract_retrieved_articles(result) -> set:
    """Extract all article numbers from GraphRAG result."""
    return extract_tree_articles(result) | extract_kg_articles(result)


def extract_retrieved_articles_ranked(result) -> list:
    """Extract all article numbers in ranked order (tree first, then KG)."""
    tree_ranked = extract_tree_articles_ranked(result)
    kg_ranked = extract_kg_articles_ranked(result)
    # Merge: tree first, then KG (deduplicated)
    seen = set(tree_ranked)
    merged = list(tree_ranked)
    for a in kg_ranked:
        if a not in seen:
            merged.append(a)
            seen.add(a)
    return merged


def get_tree_confidence(result) -> float:
    """Get tree search confidence."""
    if result.tree_search_result:
        return result.tree_search_result.confidence
    return 0.0


def get_tree_reasoning(result) -> str:
    """Get tree traversal reasoning path."""
    if result.tree_search_result and result.tree_search_result.reasoning_path:
        return " → ".join(result.tree_search_result.reasoning_path[:2])
    return ""


def calculate_metrics(expected: set, retrieved: set) -> dict:
    """Calculate precision, recall, F1 for a single query.

    Args:
        expected: Set of expected article numbers (ground truth)
        retrieved: Set of retrieved article numbers

    Returns:
        Dict with precision, recall, f1, hit (bool), overlap count
    """
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": False, "overlap": 0}

    overlap = expected & retrieved
    overlap_count = len(overlap)

    # Hit: at least one expected article found
    hit = overlap_count > 0

    # Precision: what fraction of retrieved are relevant
    precision = overlap_count / len(retrieved) if retrieved else 0.0

    # Recall: what fraction of expected were found
    recall = overlap_count / len(expected) if expected else 0.0

    # F1: harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": hit,
        "overlap": overlap_count,
    }


# Default K values for IR metrics
K_VALUES = [1, 3, 5, 10]


def calculate_ir_metrics(expected: set, ranked_retrieved: list) -> dict:
    """Calculate comprehensive IR metrics for a single query.

    Args:
        expected: Set of expected article numbers (ground truth)
        ranked_retrieved: List of retrieved articles in ranked order

    Returns:
        Dict with MRR, Recall@K, Precision@K, NDCG@K, Hit@K for each K
    """
    import math

    metrics = {}

    if not expected or not ranked_retrieved:
        # Return zeros for all metrics
        metrics["rr"] = 0.0  # Reciprocal Rank
        for k in K_VALUES:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"hit@{k}"] = 0
        return metrics

    # Reciprocal Rank: 1/position of first relevant result
    rr = 0.0
    for i, article in enumerate(ranked_retrieved):
        if article in expected:
            rr = 1.0 / (i + 1)
            break
    metrics["rr"] = rr

    # Calculate metrics at each K
    for k in K_VALUES:
        top_k = ranked_retrieved[:k]
        top_k_set = set(top_k)

        # Recall@K: fraction of expected found in top K
        relevant_in_k = len(expected & top_k_set)
        metrics[f"recall@{k}"] = relevant_in_k / len(expected)

        # Precision@K: fraction of top K that are relevant
        metrics[f"precision@{k}"] = relevant_in_k / k if k > 0 else 0.0

        # Hit@K: 1 if any relevant in top K, else 0
        metrics[f"hit@{k}"] = 1 if relevant_in_k > 0 else 0

        # NDCG@K
        dcg = 0.0
        for i, article in enumerate(top_k):
            if article in expected:
                dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0

        # Ideal DCG: all relevant items at top positions
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

    # Calculate merge weight used
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


def main():
    parser = argparse.ArgumentParser(description="Full training test with real-time stats")
    parser.add_argument("--db", default="data/legal_docs.db", help="Database path")
    parser.add_argument("--kg", default="data/kg_enhanced/legal_kg.json", help="KG path")
    parser.add_argument("--summaries", default="data/kg_enhanced/chapter_summaries.json",
                        help="Chapter summaries JSON")
    parser.add_argument("--article-summaries", default="data/kg_enhanced/article_summaries.json",
                        help="Article summaries JSON for loop 2")
    parser.add_argument("--document-summaries", default="data/kg_enhanced/document_summaries_loop0.json",
                        help="Document summaries JSON for loop 0 (multi-doc)")
    parser.add_argument("--training", default="data/training/training_with_ids.csv", help="Training CSV")
    parser.add_argument("--document", default=None,
                        help="Single document ID (disables multi-document mode)")
    parser.add_argument("--single-document", action="store_true",
                        help="Use single-document mode with --document (default: multi-document)")
    parser.add_argument("--provider", default="anthropic", help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="LLM model (default: claude-3-5-haiku)")
    parser.add_argument("--base-url", default="http://127.0.0.1:3456", help="LLM base URL (default: claude-max-proxy)")
    parser.add_argument("--cache-db", default="data/llm_cache.db", help="LLM cache database (default: data/llm_cache.db)")
    parser.add_argument("--start", type=int, default=1, help="Start from row number")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--stt-list", help="JSON file with list of STT numbers to test (for re-testing failed questions)")
    parser.add_argument("--merge-with", help="Merge results with existing JSON file (for incremental testing)")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--ablation", help="Ablation config name (e.g., no_tree, no_reranker, dual_only)")

    args = parser.parse_args()

    # Validate: Anthropic provider only supports 1 worker due to rate limits
    if args.provider == "anthropic" and args.workers > 1:
        print("ERROR: Anthropic provider only supports 1 worker due to rate limits.")
        print("       Use --workers 1 or switch to a different provider (e.g., --provider openai).")
        sys.exit(1)

    print("=" * 70)
    print("VN LEGAL RAG ENHANCED - Full Training Test")
    print("=" * 70)

    # Initialize
    print("\n[1/4] Loading Knowledge Graph...")
    with open(args.kg, "r", encoding="utf-8") as f:
        kg = json.load(f)
    print(f"      Entities: {len(kg.get('entities', []))}")

    print("\n[2/5] Loading chapter summaries...")
    chapter_summaries = load_chapter_summaries(Path(args.summaries))
    print(f"      Chapters: {len(chapter_summaries)}")

    print("\n[3/6] Loading article summaries...")
    article_summaries = load_article_summaries(Path(args.article_summaries))
    # Display actual count (summaries list, not top-level keys)
    article_count = len(article_summaries.get("summaries", [])) if isinstance(article_summaries, dict) else len(article_summaries)
    print(f"      Articles: {article_count}")

    print("\n[4/6] Loading document summaries...")
    document_summaries = load_document_summaries(Path(args.document_summaries))
    print(f"      Documents: {len(document_summaries)}")

    print("\n[5/6] Building tree index...")
    db = LegalDocumentDB(args.db)

    # Default: multi-document mode (Loop 0 enabled)
    # Use --single-document or --document to switch to single-document mode
    use_multi_doc = not args.single_document and args.document is None

    if use_multi_doc:
        # Multi-document mode: build forest with all documents
        print("      Mode: Multi-document (Loop 0 enabled)")
        forest = build_forest_from_db(db, chapter_summaries)
    else:
        # Single-document mode: build tree for specific document
        doc_id = args.document or "59-2020-QH14"
        print(f"      Mode: Single-document ({doc_id})")
        tree = build_tree_from_db(db, doc_id, chapter_summaries)
        forest = UnifiedForest()
        forest.add_tree(tree)

    stats = forest.stats()
    print(f"      Documents: {len(forest.trees)}, Nodes: {stats.total_nodes}")

    loop_mode = "3-loop (Loop 0 + Loop 1 + Loop 2)" if use_multi_doc else "2-loop (Loop 1 + Loop 2)"

    # Parse ablation config if provided
    ablation_config = None
    if args.ablation:
        ablation_configs = get_paper_ablation_configs()
        if args.ablation in ablation_configs:
            ablation_config = ablation_configs[args.ablation]
            print(f"\n[!] Using ablation config: {args.ablation}")
            print(f"    enable_tree={ablation_config.enable_tree}, enable_reranker={ablation_config.enable_reranker}")
        else:
            print(f"\n[!] WARNING: Unknown ablation config '{args.ablation}'")
            print(f"    Available: {list(ablation_configs.keys())}")

    print(f"\n[6/6] Initializing GraphRAG with {loop_mode} tree traversal...")
    rag = LegalGraphRAG(
        kg=kg,
        db_path=args.db,
        llm_provider=args.provider,
        llm_model=args.model,
        llm_base_url=args.base_url,
        llm_cache_db=args.cache_db,
        forest=forest,
        article_summaries=article_summaries,
        document_summaries=document_summaries if use_multi_doc else None,
        ablation_config=ablation_config,
    )

    # Load training data
    print("\n" + "=" * 70)
    print("Loading training data...")

    with open(args.training, "r", encoding="utf-8-sig") as f:
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

    # Stats tracking (thread-safe)
    stats_lock = Lock()
    hits = 0
    misses = 0
    skipped = 0
    category_stats = defaultdict(lambda: {"hits": 0, "total": 0})
    results = []
    processed_count = 0

    # JSON output (replaces CSV for better structure)
    json_results = []  # Store all results for JSON output

    def process_single_question(task_data: dict) -> dict:
        """Process a single question with retry on rate limit."""
        row = task_data["row"]
        row_num = task_data["row_num"]

        stt = row.get("STT", str(row_num))
        category = row.get("Category", "")
        question = row.get("Content", "")
        article_ids = row.get("Article_IDs", "")
        expected = extract_article_numbers(article_ids)

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

        # Query with retry on rate limit
        max_retries = 5
        base_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                result = rag.query(question, adaptive_retrieval=True)
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
                # Check for rate limit errors
                if "rate" in error_str or "limit" in error_str or "429" in error_str or "quota" in error_str:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
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

            # Check hits
            tree_hit = bool(expected & tree_articles)
            kg_hit = bool(expected & kg_articles)
            # hit@10: check if any expected article is in top 10 ranked results
            is_hit = bool(expected & set(ranked_retrieved[:10]))

            # Calculate precision, recall, F1
            metrics = calculate_metrics(expected, retrieved)
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1"]

            # Calculate IR metrics (MRR, Recall@K, Precision@K, NDCG@K)
            ir_metrics = calculate_ir_metrics(expected, ranked_retrieved)

            if is_hit:
                hits += 1
                category_stats[category]["hits"] += 1
                status = "HIT"
            else:
                misses += 1
                status = "MISS"

            category_stats[category]["total"] += 1
            # Track precision/recall/f1 per category
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

            # Calculate running rate
            total_tested = hits + misses
            hit_rate = (hits / total_tested * 100) if total_tested > 0 else 0

            # Print detailed result
            cat_short = category[:20] + "..." if len(category) > 20 else category
            tree_status = "T✓" if tree_hit else "T✗"
            kg_status = "K✓" if kg_hit else "K✗"

            print(f"[{local_processed:4d}] STT {stt:4s} | {status:4s} | {tree_status} {kg_status} | Hit Rate@10: {hit_rate:5.1f}% | P:{precision:.2f} R:{recall:.2f} F1:{f1:.2f} RR:{ir_metrics['rr']:.2f} | {cat_short}")
            print(f"       Expected: {sorted(expected)} | Retrieved(ranked): {ranked_retrieved[:5]}")

            # Store full result for JSON output
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
            }
            json_results.append(result_record)

            # Store simplified result for internal stats
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

    # Load STT list if provided (for re-testing failed questions only)
    stt_filter = None
    if args.stt_list:
        print(f"\nLoading STT list from {args.stt_list}...")
        with open(args.stt_list, "r") as f:
            stt_data = json.load(f)
            # Support both list format and dict with 'failed' key
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

        # Filter by STT list if provided
        if stt_filter and str(stt) not in stt_filter:
            continue
        if row_num < args.start:
            continue
        if args.limit and len(tasks) >= args.limit:
            break
        tasks.append({"row": row, "row_num": row_num})

    try:
        if args.workers == 1:
            # Sequential execution (original behavior)
            for task in tasks:
                result_data = process_single_question(task)
                update_stats_and_print(result_data)
        else:
            # Parallel execution
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

    # Final summary with ablation breakdown
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (ABLATION)")
    print("=" * 70)

    total_tested = hits + misses
    hit_rate = (hits / total_tested * 100) if total_tested > 0 else 0

    # Calculate average precision, recall, F1
    avg_precision = sum(r.get("precision", 0) for r in results) / len(results) if results else 0
    avg_recall = sum(r.get("recall", 0) for r in results) / len(results) if results else 0
    avg_f1 = sum(r.get("f1", 0) for r in results) / len(results) if results else 0

    # Calculate Tree vs KG stats
    tree_hits = sum(1 for r in results if r.get("tree_hit"))
    kg_hits = sum(1 for r in results if r.get("kg_hit"))
    both_hits = sum(1 for r in results if r.get("tree_hit") and r.get("kg_hit"))
    tree_only = sum(1 for r in results if r.get("tree_hit") and not r.get("kg_hit"))
    kg_only = sum(1 for r in results if r.get("kg_hit") and not r.get("tree_hit"))
    neither = sum(1 for r in results if not r.get("tree_hit") and not r.get("kg_hit"))

    # Calculate average IR metrics
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

    # Primary metric: Hit@10 (realistic for RAG with top-10 contexts)
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
        # Calculate Tree-only metrics (set-based)
        tree_metrics = [calculate_metrics(r["expected"], r["tree_articles"]) for r in results]
        tree_avg_prec = sum(m["precision"] for m in tree_metrics) / len(tree_metrics)
        tree_avg_recall = sum(m["recall"] for m in tree_metrics) / len(tree_metrics)
        tree_avg_f1 = sum(m["f1"] for m in tree_metrics) / len(tree_metrics)

        # Calculate KG-only metrics (set-based)
        kg_metrics = [calculate_metrics(r["expected"], r["kg_articles"]) for r in results]
        kg_avg_prec = sum(m["precision"] for m in kg_metrics) / len(kg_metrics)
        kg_avg_recall = sum(m["recall"] for m in kg_metrics) / len(kg_metrics)
        kg_avg_f1 = sum(m["f1"] for m in kg_metrics) / len(kg_metrics)

        # Calculate Tree-only IR metrics (ranked)
        tree_ir = [calculate_ir_metrics(r["expected"], list(r["tree_articles"])) for r in results]
        tree_mrr = sum(m["rr"] for m in tree_ir) / len(tree_ir)

        # Calculate KG-only IR metrics (ranked)
        kg_ir = [calculate_ir_metrics(r["expected"], list(r["kg_articles"])) for r in results]
        kg_mrr = sum(m["rr"] for m in kg_ir) / len(kg_ir)

        # Basic metrics table
        print(f"\n{'Component':<10} {'Hit%':>7} {'Prec':>8} {'Recall':>8} {'F1':>8} {'MRR':>8}")
        print(f"{'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'Tree':<10} {tree_hits/len(results)*100:>6.1f}% {tree_avg_prec:>8.4f} {tree_avg_recall:>8.4f} {tree_avg_f1:>8.4f} {tree_mrr:>8.4f}")
        print(f"{'KG':<10} {kg_hits/len(results)*100:>6.1f}% {kg_avg_prec:>8.4f} {kg_avg_recall:>8.4f} {kg_avg_f1:>8.4f} {kg_mrr:>8.4f}")
        print(f"{'Merged':<10} {hit_rate:>6.1f}% {avg_precision:>8.4f} {avg_recall:>8.4f} {avg_f1:>8.4f} {avg_mrr:>8.4f}")

        # IR@K metrics per component
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

    # Helper function for ablation metrics
    def calc_group_metrics(group: list) -> dict:
        """Calculate full metrics for a group of results."""
        if not group:
            return {"n": 0, "hit": 0, "prec": 0, "recall": 0, "f1": 0, "mrr": 0, "ndcg5": 0}
        n = len(group)
        hits = sum(1 for r in group if r.get("hit"))
        prec = sum(r.get("precision", 0) for r in group) / n
        recall = sum(r.get("recall", 0) for r in group) / n
        f1 = sum(r.get("f1", 0) for r in group) / n
        mrr = sum(r.get("ir_metrics", {}).get("rr", 0) for r in group) / n
        ndcg5 = sum(r.get("ir_metrics", {}).get("ndcg@5", 0) for r in group) / n
        return {"n": n, "hit": hits/n, "prec": prec, "recall": recall, "f1": f1, "mrr": mrr, "ndcg5": ndcg5}

    def print_ablation_table(title: str, groups: dict):
        """Print ablation table with full metrics."""
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
        qt = r.get("query_type", "unknown")
        query_type_groups[qt].append(r)
    print_ablation_table("By Query Type", dict(query_type_groups))

    # ABLATION 2: By Intent
    intent_groups = defaultdict(list)
    for r in results:
        intent = r.get("intent", "unknown")
        intent_groups[intent].append(r)
    print_ablation_table("By Intent", dict(intent_groups))

    # ABLATION 3: By Retrieval Method
    method_groups = defaultdict(list)
    for r in results:
        method = r.get("retrieval_method", "unknown")
        method_groups[method].append(r)
    print_ablation_table("By Retrieval Method", dict(method_groups))

    # ABLATION 4: Ontology Expansion Impact
    ontology_groups = {
        "With Ontology": [r for r in results if r.get("has_ontology")],
        "Without Ontology": [r for r in results if not r.get("has_ontology")],
    }
    print_ablation_table("Ontology Expansion", ontology_groups)

    # ABLATION 5: Article Reference Detection
    artref_groups = {
        "Has Article Refs": [r for r in results if r.get("has_article_refs")],
        "No Article Refs": [r for r in results if not r.get("has_article_refs")],
    }
    print_ablation_table("Article Reference Detection", artref_groups)

    # ABLATION 6: By Tree Confidence Level
    conf_groups = {
        "High (>=0.7)": [r for r in results if r.get("tree_conf", 0) >= 0.7],
        "Medium (0.5-0.7)": [r for r in results if 0.5 <= r.get("tree_conf", 0) < 0.7],
        "Low (<0.5)": [r for r in results if r.get("tree_conf", 0) < 0.5],
    }
    print_ablation_table("Tree Confidence Level", conf_groups)

    # ABLATION 7: Tree Hit vs KG Hit Analysis
    hit_groups = {
        "Both Hit": [r for r in results if r.get("tree_hit") and r.get("kg_hit")],
        "Tree Only": [r for r in results if r.get("tree_hit") and not r.get("kg_hit")],
        "KG Only": [r for r in results if r.get("kg_hit") and not r.get("tree_hit")],
        "Neither": [r for r in results if not r.get("tree_hit") and not r.get("kg_hit")],
    }
    print_ablation_table("Component Hit Pattern", hit_groups)

    # ABLATION 8: By Number of Expected Articles (single vs multi)
    expected_groups = {
        "Single Answer": [r for r in results if len(r.get("expected", set())) == 1],
        "2-3 Answers": [r for r in results if 2 <= len(r.get("expected", set())) <= 3],
        "4+ Answers": [r for r in results if len(r.get("expected", set())) >= 4],
    }
    print_ablation_table("Expected Answer Count", expected_groups)

    # ABLATION 9: By Retrieved Count (retrieval volume)
    retrieved_groups = {
        "0 Retrieved": [r for r in results if len(r.get("retrieved", set())) == 0],
        "1-3 Retrieved": [r for r in results if 1 <= len(r.get("retrieved", set())) <= 3],
        "4-10 Retrieved": [r for r in results if 4 <= len(r.get("retrieved", set())) <= 10],
        "10+ Retrieved": [r for r in results if len(r.get("retrieved", set())) > 10],
    }
    print_ablation_table("Retrieval Volume", retrieved_groups)

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

    print("=" * 70)

    # Save results to JSON
    if args.output:
        # Build summary object
        summary = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tested": total_tested,
                "skipped": skipped,
                "document": args.document,
                "provider": args.provider,
                "model": args.model,
                "base_url": args.base_url,
            },
            "overall_metrics": {
                "hit_rate": avg_ir.get("hit@10", 0),  # Primary: Hit@10 for RAG
                "hit_rate_all": hit_rate / 100,  # Unbounded (for reference)
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
        }

        # Merge with existing results if specified
        final_results = json_results
        if args.merge_with and os.path.exists(args.merge_with):
            print(f"\nMerging with existing results from {args.merge_with}...")
            with open(args.merge_with, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_results = existing.get("results", [])

            # Build a map of new results by STT
            new_results_map = {str(r['stt']): r for r in json_results}

            # Replace matching STTs, keep others
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

            # Recalculate summary metrics
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

            # Recalculate IR metrics at K
            merged_ir = {}
            for k in K_VALUES:
                merged_ir[f"hit@{k}"] = sum(r.get("ir_metrics", {}).get(f"hit@{k}", 0) for r in final_results) / total_tested
                merged_ir[f"recall@{k}"] = sum(r.get("ir_metrics", {}).get(f"recall@{k}", 0) for r in final_results) / total_tested
                merged_ir[f"precision@{k}"] = sum(r.get("ir_metrics", {}).get(f"precision@{k}", 0) for r in final_results) / total_tested
                merged_ir[f"ndcg@{k}"] = sum(r.get("ir_metrics", {}).get(f"ndcg@{k}", 0) for r in final_results) / total_tested

            # Update summary
            summary["metadata"]["total_tested"] = total_tested
            summary["metadata"]["merged_from"] = args.merge_with
            summary["overall_metrics"] = {
                "hit_rate": merged_ir.get("hit@10", 0),  # Primary: Hit@10 for RAG
                "hit_rate_all": hit_rate / 100,  # Unbounded (for reference)
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

        # Build full output
        output_data = {
            "summary": summary,
            "results": final_results,
        }

        # Save to JSON
        output_path = args.output if args.output.endswith(".json") else f"{args.output}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")
        print(f"  - {len(final_results)} query results")
        print(f"  - Summary with {len(summary)} sections")


if __name__ == "__main__":
    main()
