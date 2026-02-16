#!/usr/bin/env python3
"""
Complete Offline Pipeline for Vietnamese Legal RAG.

Runs all offline phase steps:
1. Extract terminology from legal articles (zero LLM cost)
2. Mine synonyms from Q&A corpus (LLM-based, optional)
3. Extract entities & relations (LLM-based)
4. Validate relations (filter self-refs, normalize types)
5. Resolve entities (dedup, abbreviation expansion)
6. Post-process cross-references
7. Build knowledge graph
8. Link KG to SQLite (bidirectional navigation)
9. Generate summaries (optional)

Usage:
    # Full pipeline
    python scripts/run-complete-offline-pipeline.py --db data/legal_docs.db

    # Specific steps only
    python scripts/run-complete-offline-pipeline.py --steps terminology,extraction,kg

    # Single document
    python scripts/run-complete-offline-pipeline.py --document 59-2020-QH14

    # With Q&A synonym mining
    python scripts/run-complete-offline-pipeline.py --qa-csv data/qa_pairs.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.offline import (
    # Database
    LegalDocumentDB,
    # Extraction
    UnifiedLegalExtractor,
    ExtractionResult,
    # Validation & Resolution
    RelationValidator,
    EntityResolver,
    validate_relations,
    batch_resolve_entities,
    # Terminology & Synonyms
    TerminologyExtractor,
    SynonymMiner,
    # Cross-reference
    CrossRefPostProcessor,
    extract_crossrefs_from_relations,
    # KG Building
    IncrementalKGBuilder,
    # Linking
    KGSQLiteLinker,
    create_linker,
    # Types
    LEGAL_RELATION_TYPES,
)
from vn_legal_rag.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


AVAILABLE_STEPS = [
    "terminology",  # Step 1: Extract terminology (zero LLM)
    "synonyms",     # Step 2: Mine synonyms (LLM)
    "extraction",   # Step 3: Extract entities/relations (LLM)
    "validation",   # Step 4: Validate relations
    "resolution",   # Step 5: Resolve entities
    "crossref",     # Step 6: Post-process cross-references
    "kg",           # Step 7: Build KG
    "linking",      # Step 8: Link KG ↔ SQLite
    "summaries",    # Step 9: Generate summaries
]


def run_terminology_extraction(
    db: LegalDocumentDB,
    document_id: Optional[str] = None,
    config_dir: str = "config/domains",
) -> dict:
    """Step 1: Extract terminology from legal glossary articles."""
    logger.info("=" * 60)
    logger.info("Step 1: Terminology Extraction (zero LLM cost)")
    logger.info("=" * 60)

    extractor = TerminologyExtractor(config_dir=config_dir)
    results = {}

    # Get documents
    documents = db.get_all_documents()
    if document_id:
        documents = [d for d in documents if d.id == document_id]

    for doc in documents:
        articles = db.get_articles_for_document(doc.id)
        article_dicts = [
            {"number": a.article_number, "title": a.title, "content": a.raw_text}
            for a in articles
        ]

        result = extractor.extract_from_document(article_dicts, doc.id)
        results[doc.id] = result

        if result.terms:
            extractor.update_domain_config(doc.id, result)
            logger.info(f"  {doc.id}: {len(result.terms)} terms extracted")

    return results


def run_synonym_mining(
    csv_path: str,
    document_id: str,
    config_dir: str = "config/domains",
) -> dict:
    """Step 2: Mine synonyms from Q&A corpus."""
    logger.info("=" * 60)
    logger.info("Step 2: Synonym Mining from Q&A Corpus")
    logger.info("=" * 60)

    miner = SynonymMiner(config_dir=config_dir)
    result = miner.mine_from_csv(csv_path)
    result = miner.merge_with_seeds(result)

    if result.pairs:
        miner.update_domain_config(document_id, result)
        logger.info(f"  Mined {len(result.pairs)} synonym pairs")

    return {"pairs": len(result.pairs), "source_count": result.source_count}


def run_extraction(
    db: LegalDocumentDB,
    output_dir: Path,
    llm_provider: str,
    llm_model: str,
    document_id: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = True,
) -> List[ExtractionResult]:
    """Step 3: Extract entities and relations using LLM."""
    logger.info("=" * 60)
    logger.info("Step 3: Entity & Relation Extraction (LLM-based)")
    logger.info("=" * 60)

    checkpoint_path = output_dir / "extraction_checkpoint.json"

    # Load checkpoint
    checkpoint = {"processed_ids": [], "results": []}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        logger.info(f"  Resuming: {len(checkpoint['processed_ids'])} already processed")

    processed_ids = set(checkpoint["processed_ids"])

    # Initialize extractor
    extractor = UnifiedLegalExtractor(provider=llm_provider, model=llm_model)

    # Get articles
    articles = db.get_all_articles(document_id=document_id, limit=limit)
    logger.info(f"  Found {len(articles)} articles")

    results = []
    for i, article in enumerate(articles, 1):
        if article.article_id in processed_ids:
            continue

        logger.info(f"  [{i}/{len(articles)}] {article.article_id}")

        try:
            result = extractor.extract(
                text=article.content,
                source_id=article.article_id,
                document_id=article.document_id,
            )
            results.append(result)

            # Save checkpoint
            checkpoint["processed_ids"].append(article.article_id)
            checkpoint["results"].append({
                "entities": result.entities,
                "relations": result.relations,
                "source_id": result.source_id,
                "document_id": result.document_id,
            })

            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"    Failed: {e}")

    logger.info(f"  Extracted from {len(results)} articles")
    return results


def run_validation(
    extraction_results: List[ExtractionResult],
) -> List[ExtractionResult]:
    """Step 4: Validate relations."""
    logger.info("=" * 60)
    logger.info("Step 4: Relation Validation")
    logger.info("=" * 60)

    validator = RelationValidator(
        defined_types=LEGAL_RELATION_TYPES,
        enable_semantic_validation=True,
        require_evidence=False,  # Evidence optional for now
    )

    validated_results = []
    total_before = 0
    total_after = 0

    for result in extraction_results:
        total_before += len(result.relations)
        validated_relations = validator.validate(result.relations)
        total_after += len(validated_relations)

        validated_results.append(ExtractionResult(
            entities=result.entities,
            relations=validated_relations,
            source_id=result.source_id,
            document_id=result.document_id,
        ))

    logger.info(f"  Relations: {total_before} → {total_after} (filtered {total_before - total_after})")
    return validated_results


def run_entity_resolution(
    extraction_results: List[ExtractionResult],
) -> List[ExtractionResult]:
    """Step 5: Resolve and deduplicate entities."""
    logger.info("=" * 60)
    logger.info("Step 5: Entity Resolution & Deduplication")
    logger.info("=" * 60)

    resolver = EntityResolver(scope_by_document=True)
    resolved_results = []

    for result in extraction_results:
        resolved_entities = []
        for entity in result.entities:
            entity_id = resolver.resolve(
                entity.get("text", ""),
                entity.get("label", "UNKNOWN"),
                result.document_id,
            )
            entity["entity_id"] = entity_id

            metadata = resolver.get_entity_metadata(entity.get("text", ""))
            if metadata.expanded_text:
                entity["expanded_text"] = metadata.expanded_text
            if metadata.is_abbreviation:
                entity["is_abbreviation"] = True

            resolved_entities.append(entity)

        resolved_results.append(ExtractionResult(
            entities=resolved_entities,
            relations=result.relations,
            source_id=result.source_id,
            document_id=result.document_id,
        ))

    resolver.log_stats()
    return resolved_results


def run_crossref_postprocessing(
    extraction_results: List[ExtractionResult],
    db: Optional[LegalDocumentDB] = None,
) -> List[dict]:
    """Step 6: Post-process cross-references from relations."""
    logger.info("=" * 60)
    logger.info("Step 6: Cross-Reference Post-Processing")
    logger.info("=" * 60)

    processor = CrossRefPostProcessor(db)
    all_crossrefs = []

    for result in extraction_results:
        crossrefs = extract_crossrefs_from_relations(
            relations=result.relations,
            source_article_id=result.source_id,
            current_document_id=result.document_id,
            db=db,
        )
        all_crossrefs.extend(crossrefs)

    logger.info(f"  Extracted {len(all_crossrefs)} cross-references")
    return all_crossrefs


def run_kg_building(
    extraction_results: List[ExtractionResult],
    output_dir: Path,
) -> dict:
    """Step 7: Build knowledge graph."""
    logger.info("=" * 60)
    logger.info("Step 7: Knowledge Graph Building")
    logger.info("=" * 60)

    builder = IncrementalKGBuilder()

    for result in extraction_results:
        builder.add_extraction(result)

    kg_result = builder.build()

    # Save KG
    kg_path = output_dir / "legal_kg.json"
    kg_data = {
        "entities": kg_result.entities,
        "relations": kg_result.relations,
        "stats": kg_result.stats,
    }

    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)

    logger.info(f"  Entities: {len(kg_result.entities)}")
    logger.info(f"  Relations: {len(kg_result.relations)}")
    logger.info(f"  Saved to: {kg_path}")

    return kg_data


def run_kg_linking(
    kg_path: Path,
    db_path: str,
) -> dict:
    """Step 8: Link KG to SQLite for bidirectional navigation."""
    logger.info("=" * 60)
    logger.info("Step 8: KG ↔ SQLite Bidirectional Linking")
    logger.info("=" * 60)

    linker = create_linker(str(kg_path), db_path)

    # Test linking
    stats = {
        "kg_entities": len(linker.kg.get("entities", [])),
        "kg_relations": len(linker.kg.get("relationships", [])),
    }

    logger.info(f"  Linked KG with {stats['kg_entities']} entities")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run complete offline pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--db", default="data/legal_docs.db", help="Database path")
    parser.add_argument("--output", default="data/kg_enhanced", help="Output directory")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    parser.add_argument("--document", help="Filter by document ID")
    parser.add_argument("--limit", type=int, help="Limit articles")
    parser.add_argument("--qa-csv", help="Q&A CSV for synonym mining")
    parser.add_argument(
        "--steps",
        default="all",
        help=f"Steps to run (comma-separated): {','.join(AVAILABLE_STEPS)} or 'all'",
    )
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Parse steps
    if args.steps == "all":
        steps = AVAILABLE_STEPS
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    # Load config
    config = load_config(args.config)
    llm_provider = config.get("llm", {}).get("provider", "anthropic")
    llm_model = config.get("llm", {}).get("model", "claude-3-5-haiku-20241022")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("COMPLETE OFFLINE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Database: {args.db}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Steps: {', '.join(steps)}")
    logger.info(f"LLM: {llm_provider}/{llm_model}")

    # Initialize database
    db = LegalDocumentDB(args.db)

    extraction_results = []

    # Step 1: Terminology
    if "terminology" in steps:
        run_terminology_extraction(db, args.document)

    # Step 2: Synonyms
    if "synonyms" in steps and args.qa_csv:
        doc_id = args.document or "59-2020-QH14"
        run_synonym_mining(args.qa_csv, doc_id)

    # Step 3: Extraction
    if "extraction" in steps:
        extraction_results = run_extraction(
            db, output_dir, llm_provider, llm_model,
            args.document, args.limit, not args.no_resume,
        )

    # Load from checkpoint if skipping extraction
    if not extraction_results:
        checkpoint_path = output_dir / "extraction_checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            extraction_results = [
                ExtractionResult(**r) for r in checkpoint.get("results", [])
            ]
            logger.info(f"Loaded {len(extraction_results)} results from checkpoint")

    # Step 4: Validation
    if "validation" in steps and extraction_results:
        extraction_results = run_validation(extraction_results)

    # Step 5: Resolution
    if "resolution" in steps and extraction_results:
        extraction_results = run_entity_resolution(extraction_results)

    # Step 6: Cross-references
    if "crossref" in steps and extraction_results:
        crossrefs = run_crossref_postprocessing(extraction_results, db)
        # Save crossrefs
        crossref_path = output_dir / "crossrefs.json"
        with open(crossref_path, "w", encoding="utf-8") as f:
            json.dump(crossrefs, f, ensure_ascii=False, indent=2)

    # Step 7: KG Building
    if "kg" in steps and extraction_results:
        kg_data = run_kg_building(extraction_results, output_dir)

    # Step 8: Linking
    if "linking" in steps:
        kg_path = output_dir / "legal_kg.json"
        if kg_path.exists():
            run_kg_linking(kg_path, args.db)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
