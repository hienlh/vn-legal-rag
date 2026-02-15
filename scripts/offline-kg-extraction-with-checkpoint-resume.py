#!/usr/bin/env python3
"""
Offline pipeline: Extract KG from legal documents.

Usage:
    python scripts/run_offline.py --db data/legal_docs.db --output data/kg_enhanced
    python scripts/run_offline.py --limit 10 --document 59-2020-QH14
    python scripts/run_offline.py --no-resume --output data/kg_fresh
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.offline import (
    LegalDocumentDB,
    UnifiedLegalExtractor,
    IncrementalKGBuilder,
    ExtractionResult,
)
from vn_legal_rag.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load extraction checkpoint from disk."""
    if not checkpoint_path.exists():
        return {
            "processed_ids": [],
            "extraction_results": [],
            "stats": {"successful": 0, "failed": 0, "total_entities": 0, "total_relations": 0},
            "last_saved": None,
        }

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint_path: Path, checkpoint: dict) -> None:
    """Save extraction checkpoint atomically."""
    import tempfile
    from datetime import datetime

    checkpoint["last_saved"] = datetime.now().isoformat()

    # Atomic write: write to temp file then rename
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=checkpoint_path.parent
    ) as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        temp_path = f.name

    Path(temp_path).rename(checkpoint_path)


def run_offline_pipeline(
    db_path: str,
    output_dir: str,
    llm_provider: str,
    llm_model: str,
    document_id: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = True,
) -> dict:
    """
    Run offline KG extraction pipeline.

    Args:
        db_path: Path to legal documents SQLite database
        output_dir: Output directory for KG and checkpoint
        llm_provider: LLM provider (openai, anthropic, gemini)
        llm_model: LLM model name
        document_id: Filter by document ID
        limit: Limit number of articles
        resume: Enable checkpoint resume

    Returns:
        Result dict with kg, stats, checkpoint_path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / "checkpoint.json"
    kg_path = output_path / "legal_kg.json"

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(checkpoint_path) if resume else {
        "processed_ids": [],
        "extraction_results": [],
        "stats": {"successful": 0, "failed": 0, "total_entities": 0, "total_relations": 0},
        "last_saved": None,
    }

    processed_ids = set(checkpoint["processed_ids"])

    logger.info(f"Resume enabled: {resume}")
    if resume and processed_ids:
        logger.info(f"Resuming from checkpoint: {len(processed_ids)} articles already processed")

    # Initialize components
    db = LegalDocumentDB(db_path)
    extractor = UnifiedLegalExtractor(provider=llm_provider, model=llm_model)
    builder = IncrementalKGBuilder()

    # Restore previous extractions into builder
    for result_dict in checkpoint["extraction_results"]:
        result = ExtractionResult(
            entities=result_dict["entities"],
            relations=result_dict["relations"],
            source_id=result_dict["source_id"],
            document_id=result_dict["document_id"],
        )
        builder.add_extraction(result)

    # Fetch articles
    articles = db.get_all_articles(document_id=document_id, limit=limit)

    if not articles:
        logger.warning("No articles found in database")
        return {"kg": None, "stats": checkpoint["stats"], "checkpoint_path": None}

    logger.info(f"Found {len(articles)} articles to process")

    # Process articles
    for i, article in enumerate(articles, 1):
        article_id = article.article_id

        # Skip if already processed
        if article_id in processed_ids:
            logger.debug(f"Skipping already processed article {article_id}")
            continue

        logger.info(f"[{i}/{len(articles)}] Processing {article_id}")

        try:
            # Extract entities and relations
            result = extractor.extract(
                text=article.content,
                source_id=article_id,
                document_id=article.document_id,
            )

            # Add to builder (merges immediately)
            builder.add_extraction(result)

            # Update checkpoint
            checkpoint["processed_ids"].append(article_id)
            checkpoint["extraction_results"].append({
                "entities": result.entities,
                "relations": result.relations,
                "source_id": result.source_id,
                "document_id": result.document_id,
            })
            checkpoint["stats"]["successful"] += 1
            checkpoint["stats"]["total_entities"] += len(result.entities)
            checkpoint["stats"]["total_relations"] += len(result.relations)

            # Save checkpoint after each article
            save_checkpoint(checkpoint_path, checkpoint)

            logger.info(f"  Extracted: {len(result.entities)} entities, {len(result.relations)} relations")

        except Exception as e:
            logger.error(f"Failed to process {article_id}: {e}")
            checkpoint["stats"]["failed"] += 1
            save_checkpoint(checkpoint_path, checkpoint)
            continue

    # Build final KG
    kg_result = builder.build()

    # Save KG
    kg_data = {
        "entities": kg_result.entities,
        "relations": kg_result.relations,
        "stats": kg_result.stats,
    }

    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction complete!")
    logger.info(f"  Total articles processed: {checkpoint['stats']['successful']}")
    logger.info(f"  Failed: {checkpoint['stats']['failed']}")
    logger.info(f"  Unique entities: {len(kg_result.entities)}")
    logger.info(f"  Unique relations: {len(kg_result.relations)}")
    logger.info(f"  Entities merged: {kg_result.stats.get('entities_merged', 0)}")
    logger.info(f"\nOutput files:")
    logger.info(f"  KG: {kg_path}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*60}")

    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Checkpoint cleaned up")

    return {
        "kg": kg_data,
        "stats": checkpoint["stats"],
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run offline KG extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db",
        default="data/legal_docs.db",
        help="Path to legal documents database (default: data/legal_docs.db)",
    )
    parser.add_argument(
        "--output",
        default="data/kg_enhanced",
        help="Output directory (default: data/kg_enhanced)",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Config file path (default: config/default.yaml)",
    )
    parser.add_argument(
        "--document",
        help="Filter by document ID (e.g., 59-2020-QH14)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of articles to process",
    )
    parser.add_argument(
        "--provider",
        help="LLM provider (overrides config)",
    )
    parser.add_argument(
        "--model",
        help="LLM model name (overrides config)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and start fresh",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Load config
    config = load_config(args.config)

    # Get LLM settings (CLI args override config)
    llm_provider = args.provider or config.get("llm", {}).get("provider", "anthropic")
    llm_model = args.model or config.get("llm", {}).get("model", "claude-3-5-haiku-20241022")

    logger.info(f"Starting offline pipeline...")
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  LLM: {llm_provider}/{llm_model}")
    logger.info(f"  Document filter: {args.document or 'None'}")
    logger.info(f"  Limit: {args.limit or 'None'}")
    logger.info(f"  Resume: {not args.no_resume}")

    try:
        result = run_offline_pipeline(
            db_path=args.db,
            output_dir=args.output,
            llm_provider=llm_provider,
            llm_model=llm_model,
            document_id=args.document,
            limit=args.limit,
            resume=not args.no_resume,
        )

        logger.info("Pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
