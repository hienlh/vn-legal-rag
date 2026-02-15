#!/usr/bin/env python3
"""
Evaluate retrieval performance on test set.

Usage:
    # Full evaluation
    python scripts/evaluate-retrieval-performance-on-test-set.py --test-file data/benchmark/legal-qa-benchmark.csv

    # Limit test questions
    python scripts/evaluate-retrieval-performance-on-test-set.py --limit 50 --verbose

    # Export results
    python scripts/evaluate-retrieval-performance-on-test-set.py --output results/eval_results.json
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.online import LegalGraphRAG, create_legal_graphrag, GraphRAGResponse
from vn_legal_rag.types import UnifiedForest
from vn_legal_rag.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def load_test_data(csv_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load test questions from CSV file.

    Supports two CSV formats:
    1. Simple: question,article_ids,document_id
    2. Training: STT,Category,Content,URL,Câu trả lời,Điều luật tham chiếu,Article_IDs

    Args:
        csv_path: Path to CSV file
        limit: Limit number of questions

    Returns:
        List of test cases with question, expected_ids, document_id
    """
    import re
    test_cases = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            # Support both CSV formats
            question = (
                row.get("Content", "").strip() or
                row.get("question", "").strip()
            )
            article_ids_str = (
                row.get("Article_IDs", "").strip() or
                row.get("article_ids", "").strip()
            )
            stt = row.get("STT", str(i + 1))

            if not question or not article_ids_str:
                continue

            # Parse article IDs from format like "59-2020-QH14:d206:k1;59-2020-QH14:d206:k3"
            # Extract article numbers (e.g., d206 -> 206)
            expected_articles = set()
            for ref in article_ids_str.split(";"):
                ref = ref.strip()
                if not ref:
                    continue
                # Match patterns like :d206 or :d206:k1
                match = re.search(r":d(\d+)", ref)
                if match:
                    expected_articles.add(int(match.group(1)))

            if not expected_articles:
                continue

            test_cases.append({
                "question": question,
                "expected_ids": list(expected_articles),
                "stt": int(stt) if stt.isdigit() else i + 1,
                "index": i,
                "category": row.get("Category", ""),
            })

    logger.info(f"Loaded {len(test_cases)} test questions from {csv_path}")
    return test_cases


def extract_article_number(article_id: str) -> Optional[int]:
    """Extract article number from various formats.

    Handles:
    - '59-2020-QH14:d206' -> 206
    - '01-2021-ND:d7' -> 7
    - 'Điều 7' -> 7
    - 'd206' -> 206
    - '206' -> 206
    """
    import re
    if isinstance(article_id, int):
        return article_id
    article_str = str(article_id)

    # Pattern 1: :d206 or d206
    match = re.search(r":?d(\d+)", article_str)
    if match:
        return int(match.group(1))

    # Pattern 2: Điều 7 or Điều 206
    match = re.search(r"[Đđ]i[ềêe]u\s*(\d+)", article_str)
    if match:
        return int(match.group(1))

    # Pattern 3: Plain number
    if article_str.isdigit():
        return int(article_str)

    return None


def normalize_ids(ids: List) -> set:
    """Normalize article IDs to set of article numbers."""
    result = set()
    for aid in ids:
        num = extract_article_number(aid) if isinstance(aid, str) else aid
        if num is not None:
            result.add(num)
    return result


def calculate_hit_at_k(retrieved: List, expected: List, k: int = 5) -> bool:
    """Check if any expected article is in top-k retrieved."""
    top_k = normalize_ids(retrieved[:k])
    expected_set = normalize_ids(expected)
    return len(top_k & expected_set) > 0


def calculate_recall_at_k(retrieved: List, expected: List, k: int = 5) -> float:
    """Calculate recall@k = |retrieved ∩ expected| / |expected|."""
    if not expected:
        return 0.0

    top_k = normalize_ids(retrieved[:k])
    expected_set = normalize_ids(expected)
    return len(top_k & expected_set) / len(expected_set) if expected_set else 0.0


def calculate_mrr(retrieved: List, expected: List) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR = 1 / rank of first correct answer (0 if no correct answer)
    """
    expected_set = normalize_ids(expected)

    for i, article_id in enumerate(retrieved, 1):
        num = extract_article_number(article_id) if isinstance(article_id, str) else article_id
        if num in expected_set:
            return 1.0 / i

    return 0.0


def evaluate_single_query(
    graphrag: LegalGraphRAG,
    test_case: Dict,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate a single query.

    Args:
        graphrag: GraphRAG instance
        test_case: Test case dict
        verbose: Print detailed results

    Returns:
        Evaluation result dict
    """
    question = test_case["question"]
    expected_ids = test_case["expected_ids"]
    index = test_case["index"]

    if verbose:
        logger.info(f"\n[{index}] Question: {question}")
        logger.info(f"  Expected: {', '.join(str(x) for x in expected_ids)}")

    try:
        # Run query
        response = graphrag.query(query=question, adaptive_retrieval=True)

        # Extract article IDs from citations
        retrieved_ids = []
        for citation in response.citations:
            # Try different fields: source_id like "01-2021-ND:d7" or citation_string like "Điều 7"
            article_id = (
                citation.get("article_id") or
                citation.get("article_number") or
                citation.get("source_id") or
                citation.get("citation_string")
            )
            if article_id:
                num = extract_article_number(article_id)
                if num:
                    retrieved_ids.append(num)

        # Calculate metrics
        hit_at_5 = calculate_hit_at_k(retrieved_ids, expected_ids, k=5)
        hit_at_10 = calculate_hit_at_k(retrieved_ids, expected_ids, k=10)
        recall_at_5 = calculate_recall_at_k(retrieved_ids, expected_ids, k=5)
        recall_at_10 = calculate_recall_at_k(retrieved_ids, expected_ids, k=10)
        mrr = calculate_mrr(retrieved_ids, expected_ids)

        result = {
            "index": index,
            "question": question,
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "query_type": response.query_type.value,
            "hit@5": hit_at_5,
            "hit@10": hit_at_10,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
            "mrr": mrr,
            "success": hit_at_5,  # Success = hit@5
        }

        if verbose:
            logger.info(f"  Retrieved: {', '.join(str(x) for x in retrieved_ids[:5])}")
            logger.info(f"  Hit@5: {hit_at_5}, Recall@5: {recall_at_5:.2%}, MRR: {mrr:.4f}")

        return result

    except Exception as e:
        logger.error(f"[{index}] Query failed: {e}")

        return {
            "index": index,
            "question": question,
            "expected_ids": expected_ids,
            "retrieved_ids": [],
            "query_type": "error",
            "hit@5": False,
            "hit@10": False,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "success": False,
            "error": str(e),
        }


def calculate_aggregate_metrics(results: List[Dict]) -> Dict:
    """Calculate aggregate metrics across all results."""
    total = len(results)
    successful = sum(1 for r in results if r["success"])

    hit_at_5 = sum(1 for r in results if r["hit@5"]) / total if total > 0 else 0
    hit_at_10 = sum(1 for r in results if r["hit@10"]) / total if total > 0 else 0
    avg_recall_at_5 = sum(r["recall@5"] for r in results) / total if total > 0 else 0
    avg_recall_at_10 = sum(r["recall@10"] for r in results) / total if total > 0 else 0
    avg_mrr = sum(r["mrr"] for r in results) / total if total > 0 else 0

    return {
        "total_queries": total,
        "successful_queries": successful,
        "failed_queries": total - successful,
        "hit_rate@5": hit_at_5,
        "hit_rate@10": hit_at_10,
        "recall@5": avg_recall_at_5,
        "recall@10": avg_recall_at_10,
        "mrr": avg_mrr,
    }


def print_evaluation_summary(metrics: Dict, results: List[Dict]) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    print(f"\nTotal Queries: {metrics['total_queries']}")
    print(f"Successful: {metrics['successful_queries']} ({metrics['successful_queries']/metrics['total_queries']:.1%})")
    print(f"Failed: {metrics['failed_queries']} ({metrics['failed_queries']/metrics['total_queries']:.1%})")

    print(f"\n{'='*80}")
    print(f"RETRIEVAL METRICS")
    print(f"{'='*80}")
    print(f"Hit Rate@5:  {metrics['hit_rate@5']:.2%}")
    print(f"Hit Rate@10: {metrics['hit_rate@10']:.2%}")
    print(f"Recall@5:    {metrics['recall@5']:.2%}")
    print(f"Recall@10:   {metrics['recall@10']:.2%}")
    print(f"MRR:         {metrics['mrr']:.4f}")

    # Show failed queries
    failed = [r for r in results if not r["success"]]
    if failed:
        print(f"\n{'='*80}")
        print(f"FAILED QUERIES ({len(failed)})")
        print(f"{'='*80}")
        for r in failed[:10]:  # Show first 10
            print(f"[{r['index']}] {r['question']}")
            expected_str = ', '.join(str(x) for x in r['expected_ids'])
            retrieved_str = ', '.join(str(x) for x in r['retrieved_ids'][:5]) if r['retrieved_ids'] else 'None'
            print(f"  Expected: {expected_str}")
            print(f"  Retrieved: {retrieved_str}")
            if r.get("error"):
                print(f"  Error: {r['error']}")
            print()

        if len(failed) > 10:
            print(f"... and {len(failed) - 10} more failed queries")

    print("="*80 + "\n")


def run_evaluation(
    graphrag: LegalGraphRAG,
    test_cases: List[Dict],
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Run evaluation on all test cases.

    Args:
        graphrag: GraphRAG instance
        test_cases: List of test cases
        verbose: Print detailed results

    Returns:
        Tuple of (results, metrics)
    """
    results = []

    logger.info(f"\nStarting evaluation on {len(test_cases)} test cases...")

    for i, test_case in enumerate(test_cases, 1):
        if not verbose and i % 10 == 0:
            logger.info(f"Progress: {i}/{len(test_cases)} ({i/len(test_cases):.1%})")

        result = evaluate_single_query(graphrag, test_case, verbose=verbose)
        results.append(result)

    # Calculate aggregate metrics
    metrics = calculate_aggregate_metrics(results)

    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test-file",
        default="data/benchmark/legal-qa-benchmark.csv",
        help="Path to test CSV file (default: data/benchmark/legal-qa-benchmark.csv)",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Config file path (default: config/default.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of test questions",
    )
    parser.add_argument(
        "--output",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-query results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # Load config
    config = load_config(args.config)

    logger.info(f"Starting evaluation...")
    logger.info(f"  Test file: {args.test_file}")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Limit: {args.limit or 'None'}")

    try:
        # Load test data
        test_cases = load_test_data(args.test_file, limit=args.limit)

        if not test_cases:
            logger.error("No test cases loaded")
            return 1

        # Load GraphRAG components
        from vn_legal_rag.utils import load_kg, load_summaries, build_forest_from_db

        kg = load_kg(config.get("kg", {}).get("path", "data/kg_enhanced/legal_kg.json"))
        chapter_summaries = load_summaries(
            config.get("kg", {}).get("chapter_summaries", "data/kg_enhanced/chapter_summaries.json")
        )
        article_summaries = load_summaries(
            config.get("kg", {}).get("article_summaries", "data/kg_enhanced/article_summaries.json")
        )

        # Build forest from database
        db_path = config.get("database", {}).get("path", "data/legal_docs.db")
        forest = build_forest_from_db(db_path, chapter_summaries)

        # Create DB instance
        from vn_legal_rag.offline import LegalDocumentDB
        db = LegalDocumentDB(db_path)

        # Create LLM provider
        from vn_legal_rag.utils import create_llm_provider, create_embedding_provider
        llm_config = config.get("llm", {})
        llm_provider = create_llm_provider(
            provider=llm_config.get("provider", "anthropic"),
            model=llm_config.get("model", "claude-3-5-haiku-20241022"),
            cache_db=llm_config.get("cache_db", "data/llm_cache.db") if llm_config.get("use_cache", True) else None,
        )

        # Create embedding provider
        embedding_gen = create_embedding_provider()

        # Create GraphRAG
        logger.info(f"\nInitializing 3-Tier GraphRAG...")
        graphrag = create_legal_graphrag(
            kg=kg,
            forest=forest,
            db=db,
            llm_provider=llm_provider,
            embedding_gen=embedding_gen,
            article_summaries=article_summaries,
            config=config,
        )
        logger.info(f"  ✓ GraphRAG initialized")

        # Run evaluation
        results, metrics = run_evaluation(graphrag, test_cases, verbose=args.verbose)

        # Print summary
        print_evaluation_summary(metrics, results)

        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "test_file": args.test_file,
                    "config": args.config,
                    "total_queries": len(test_cases),
                    "limit": args.limit,
                },
                "metrics": metrics,
                "results": results,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Results saved to: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
