"""CLI entry point for education PDF extraction pipeline.

Usage:
    # Extract all documents
    python -m edu_pdf_extractor.extract --all

    # Extract specific document
    python -m edu_pdf_extractor.extract --doc 270-QD-DHCNTT

    # Extract with specific LLM provider
    python -m edu_pdf_extractor.extract --all --provider anthropic

    # Spot-check mode: print random articles
    python -m edu_pdf_extractor.extract --doc 218-QD-DHQG --spot-check
"""

import argparse
import logging
import random
import sys
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from .config import (
    DATA_DIR,
    DOCUMENTS,
    LLM_PROVIDER,
    STRUCTURED_DIR,
    TEXT_DIR,
    get_doc_by_id,
    get_source_pdf_path,
)

# Import from kebab-case modules
_images_mod = import_module(".pdf-to-images", "edu_pdf_extractor")
is_text_pdf = _images_mod.is_text_pdf
pdf_to_images = _images_mod.pdf_to_images

_text_mod = import_module(".text-pdf-extractor", "edu_pdf_extractor")
extract_text_pdf = _text_mod.extract_text_pdf

_ocr_mod = import_module(".llm-vision-ocr", "edu_pdf_extractor")
ocr_document = _ocr_mod.ocr_document

_parser_mod = import_module(".hierarchy-parser", "edu_pdf_extractor")
parse_text = _parser_mod.parse_text
ParsedDocument = _parser_mod.ParsedDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_document(doc_config: dict, provider: str = LLM_PROVIDER) -> ParsedDocument:
    """Run full extraction pipeline for a single document.

    Steps:
    1. Check if text or scan PDF
    2. Extract text (pdfplumber or LLM vision OCR)
    3. Save raw text
    4. Parse into structured hierarchy
    5. Save JSON
    """
    doc_id = doc_config["id"]
    pdf_path = get_source_pdf_path(doc_config)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"{'='*60}")
    logger.info(f"Extracting: {doc_id} ({doc_config['so_hieu']})")
    logger.info(f"Source: {pdf_path.name} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 1: Determine extraction method
    text_output = TEXT_DIR / f"{doc_id}.txt"

    if text_output.exists():
        logger.info(f"Text cache found: {text_output}")
        text = text_output.read_text(encoding="utf-8")
    elif doc_config["format"] == "text" or is_text_pdf(pdf_path):
        # Text PDF -> pdfplumber
        text = extract_text_pdf(pdf_path)
        _save_text(text, text_output)
    else:
        # Scan PDF -> LLM vision OCR
        image_paths = pdf_to_images(pdf_path, doc_id)
        text = ocr_document(image_paths, provider=provider)
        _save_text(text, text_output)

    # Step 2: Parse hierarchy
    parsed = parse_text(text, doc_config)

    # Step 3: Save structured JSON
    json_path = STRUCTURED_DIR / f"{doc_id}.json"
    parsed.save_json(json_path)

    return parsed


def _save_text(text: str, path: Path) -> None:
    """Save extracted text to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info(f"Saved text: {path} ({len(text)} chars)")


def spot_check(doc: ParsedDocument, n: int = 3) -> None:
    """Print random articles for manual verification."""
    all_articles = doc._all_articles()
    if not all_articles:
        print(f"  No articles found in {doc.doc_id}")
        return

    samples = random.sample(all_articles, min(n, len(all_articles)))
    print(f"\n--- Spot Check: {doc.doc_id} ({n} random articles) ---")
    for art in samples:
        print(f"\n  Dieu {art.number}: {art.title}")
        preview = art.content[:300] + ("..." if len(art.content) > 300 else "")
        print(f"  Content: {preview}")
        print(f"  Clauses: {len(art.clauses)}")


def print_summary(results: list[tuple[dict, ParsedDocument]]) -> None:
    """Print summary table of all extracted documents."""
    print(f"\n{'='*70}")
    print(f"{'Doc ID':<20} {'So Hieu':<18} {'Chapters':>8} {'Articles':>8} {'Clauses':>8}")
    print(f"{'-'*70}")

    total_articles = 0
    total_clauses = 0
    for doc_config, parsed in results:
        arts = parsed.total_articles()
        cls = parsed.total_clauses()
        total_articles += arts
        total_clauses += cls
        print(
            f"{parsed.doc_id:<20} {parsed.so_hieu:<18} "
            f"{len(parsed.chapters):>8} {arts:>8} {cls:>8}"
        )

    print(f"{'-'*70}")
    print(f"{'TOTAL':<20} {'':<18} {'':<8} {total_articles:>8} {total_clauses:>8}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Extract education regulation PDFs")
    parser.add_argument("--all", action="store_true", help="Extract all documents")
    parser.add_argument("--doc", type=str, help="Extract specific document by ID")
    parser.add_argument(
        "--provider", type=str, default=LLM_PROVIDER,
        choices=["gemini", "anthropic"],
        help="LLM provider for vision OCR"
    )
    parser.add_argument("--spot-check", action="store_true", help="Print random articles")
    args = parser.parse_args()

    if not args.all and not args.doc:
        parser.print_help()
        sys.exit(1)

    # Ensure output dirs exist
    for d in [TEXT_DIR, STRUCTURED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    docs_to_process = []
    if args.all:
        docs_to_process = DOCUMENTS
    elif args.doc:
        doc = get_doc_by_id(args.doc)
        if not doc:
            print(f"Unknown document ID: {args.doc}")
            print(f"Available: {[d['id'] for d in DOCUMENTS]}")
            sys.exit(1)
        docs_to_process = [doc]

    results = []
    for doc_config in docs_to_process:
        try:
            parsed = extract_document(doc_config, provider=args.provider)
            results.append((doc_config, parsed))

            if args.spot_check:
                spot_check(parsed)
        except Exception as e:
            logger.error(f"Failed to extract {doc_config['id']}: {e}")
            raise

    print_summary(results)


if __name__ == "__main__":
    main()
