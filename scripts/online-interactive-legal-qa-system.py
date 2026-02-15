#!/usr/bin/env python3
"""
Online Q&A: Interactive legal question answering with 3-Tier GraphRAG.

Usage:
    # Single query
    python scripts/online-interactive-legal-qa-system.py --query "Điều kiện thành lập CTCP?"

    # Interactive mode
    python scripts/online-interactive-legal-qa-system.py --interactive

    # Custom config
    python scripts/online-interactive-legal-qa-system.py --config config/custom.yaml --query "..."
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.online import LegalGraphRAG, create_legal_graphrag, GraphRAGResponse
from vn_legal_rag.types import UnifiedForest
from vn_legal_rag.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def load_graphrag_components(config: dict) -> tuple:
    """
    Load KG, forest, and summaries from config paths.

    Args:
        config: Configuration dict

    Returns:
        Tuple of (kg, forest, chapter_summaries, article_summaries)
    """
    kg_path = config.get("kg", {}).get("path", "data/kg_enhanced/legal_kg.json")
    chapter_summaries_path = config.get("kg", {}).get("chapter_summaries", "data/kg_enhanced/chapter_summaries.json")
    article_summaries_path = config.get("kg", {}).get("article_summaries", "data/kg_enhanced/article_summaries.json")
    forest_path = config.get("kg", {}).get("forest", "data/document_forest.json")

    logger.info(f"Loading components...")
    logger.info(f"  KG: {kg_path}")
    logger.info(f"  Forest: {forest_path}")
    logger.info(f"  Chapter summaries: {chapter_summaries_path}")
    logger.info(f"  Article summaries: {article_summaries_path}")

    # Load KG
    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    logger.info(f"  ✓ Loaded KG: {len(kg.get('entities', []))} entities, {len(kg.get('relations', []))} relations")

    # Load forest
    with open(forest_path, "r", encoding="utf-8") as f:
        forest_data = f.read()
        forest = UnifiedForest.from_json(forest_data)
    logger.info(f"  ✓ Loaded forest: {len(forest.trees)} documents")

    # Load chapter summaries
    chapter_summaries = None
    if Path(chapter_summaries_path).exists():
        with open(chapter_summaries_path, "r", encoding="utf-8") as f:
            chapter_summaries = json.load(f)
        logger.info(f"  ✓ Loaded chapter summaries: {len(chapter_summaries)} chapters")
    else:
        logger.warning(f"  ! Chapter summaries not found, tree navigation will be limited")

    # Load article summaries
    article_summaries = None
    if Path(article_summaries_path).exists():
        with open(article_summaries_path, "r", encoding="utf-8") as f:
            article_summaries = json.load(f)
        logger.info(f"  ✓ Loaded article summaries: {len(article_summaries)} articles")
    else:
        logger.warning(f"  ! Article summaries not found, tree navigation will be limited")

    return kg, forest, chapter_summaries, article_summaries


def format_response(response: GraphRAGResponse, verbose: bool = False) -> str:
    """Format GraphRAG response for display."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Query: {response.query}")
    lines.append(f"{'='*80}")

    lines.append(f"\nQuery Type: {response.query_type.value}")
    lines.append(f"Retrieval Method: 3-Tier (Tree + DualLevel + Semantic Bridge)")

    if verbose:
        lines.append(f"\nExpanded Query:")
        lines.append(f"  Keywords: {', '.join(response.expanded_query.keywords)}")
        lines.append(f"  Concepts: {', '.join(response.expanded_query.concepts)}")
        lines.append(f"  Themes: {', '.join(response.expanded_query.themes)}")

    lines.append(f"\nRetrieved Articles ({len(response.retrieved_articles)}):")
    for i, article_id in enumerate(response.retrieved_articles[:10], 1):
        lines.append(f"  {i}. {article_id}")
    if len(response.retrieved_articles) > 10:
        lines.append(f"  ... and {len(response.retrieved_articles) - 10} more")

    lines.append(f"\n{'='*80}")
    lines.append(f"ANSWER:")
    lines.append(f"{'='*80}")
    lines.append(response.response)

    if response.citations:
        lines.append(f"\n{'='*80}")
        lines.append(f"CITATIONS:")
        lines.append(f"{'='*80}")
        for i, citation in enumerate(response.citations, 1):
            lines.append(f"{i}. {citation.get('citation_string', citation.get('article_id', 'Unknown'))}")
            if verbose and citation.get('text'):
                lines.append(f"   \"{citation['text'][:100]}...\"")

    lines.append(f"{'='*80}\n")

    return "\n".join(lines)


def run_single_query(
    graphrag: LegalGraphRAG,
    query: str,
    verbose: bool = False,
) -> GraphRAGResponse:
    """Run a single query and display results."""
    logger.info(f"\nProcessing query: {query}")

    try:
        response = graphrag.query(query=query, adaptive_retrieval=True)

        # Display response
        print(format_response(response, verbose=verbose))

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise


def run_interactive_mode(
    graphrag: LegalGraphRAG,
    verbose: bool = False,
) -> None:
    """Run interactive Q&A session."""
    print("\n" + "="*80)
    print("Vietnamese Legal Q&A System - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  - Type your question in Vietnamese")
    print("  - Type 'exit' or 'quit' to end session")
    print("  - Type 'help' for command list")
    print("="*80 + "\n")

    session_count = 0

    while True:
        try:
            # Get user input
            query = input("Question: ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ["exit", "quit", "q"]:
                print(f"\nSession ended. Total queries: {session_count}")
                break

            if query.lower() in ["help", "h"]:
                print("\nAvailable commands:")
                print("  exit, quit, q - Exit interactive mode")
                print("  help, h       - Show this help")
                print("  verbose       - Toggle verbose mode")
                print()
                continue

            if query.lower() == "verbose":
                verbose = not verbose
                print(f"\nVerbose mode: {'ON' if verbose else 'OFF'}\n")
                continue

            # Process query
            session_count += 1
            run_single_query(graphrag, query, verbose=verbose)

        except KeyboardInterrupt:
            print(f"\n\nSession interrupted. Total queries: {session_count}")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError processing query: {e}\n")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Run online legal Q&A with 3-Tier GraphRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--query",
        help="Single query to process",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Config file path (default: config/default.yaml)",
    )
    parser.add_argument(
        "--db",
        help="Database path (overrides config)",
    )
    parser.add_argument(
        "--provider",
        help="LLM provider (overrides config)",
    )
    parser.add_argument(
        "--model",
        help="LLM model (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.query and not args.interactive:
        parser.error("Must provide --query or --interactive")

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.db:
        config.setdefault("database", {})["path"] = args.db
    if args.provider:
        config.setdefault("llm", {})["provider"] = args.provider
    if args.model:
        config.setdefault("llm", {})["model"] = args.model

    logger.info(f"Starting online Q&A system...")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  LLM: {config.get('llm', {}).get('provider')}/{config.get('llm', {}).get('model')}")
    logger.info(f"  Database: {config.get('database', {}).get('path')}")

    try:
        # Load components
        kg, forest, chapter_summaries, article_summaries = load_graphrag_components(config)

        # Create GraphRAG instance
        logger.info(f"\nInitializing 3-Tier GraphRAG...")
        graphrag = create_legal_graphrag(
            kg=kg,
            db_path=config.get("database", {}).get("path", "data/legal_docs.db"),
            forest=forest,
            chapter_summaries=chapter_summaries,
            article_summaries=article_summaries,
            llm_provider=config.get("llm", {}).get("provider", "anthropic"),
            llm_model=config.get("llm", {}).get("model", "claude-3-5-haiku-20241022"),
            config=config,
        )
        logger.info(f"  ✓ GraphRAG initialized")

        # Run query or interactive mode
        if args.interactive:
            run_interactive_mode(graphrag, verbose=args.verbose)
        else:
            run_single_query(graphrag, args.query, verbose=args.verbose)

        return 0

    except Exception as e:
        logger.error(f"System failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
