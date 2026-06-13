#!/usr/bin/env python3
"""
Ingest a single legal document from TVPL into the SQLite database.

Scrapes the document, parses hierarchical structure, and inserts into DB.

Usage:
    # Scrape from URL
    python scripts/ingest-single-document-from-tvpl.py \
        --url "https://thuvienphapluat.vn/van-ban/..." \
        --db data/legal_docs.db

    # Dry run (scrape + parse, no DB insert)
    python scripts/ingest-single-document-from-tvpl.py \
        --url "https://thuvienphapluat.vn/van-ban/..." \
        --dry-run

    # Override document ID
    python scripts/ingest-single-document-from-tvpl.py \
        --url "https://thuvienphapluat.vn/van-ban/..." \
        --doc-id "36-2024-QH15"
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.offline import (
    LegalDocumentDB,
    TVPLScraper,
    LegalDocumentModel,
    LegalChapterModel,
    LegalSectionModel,
    LegalArticleModel,
    LegalClauseModel,
    LegalPointModel,
    make_document_id,
    make_chapter_id,
    make_section_id,
    make_article_id,
    make_clause_id,
    make_point_id,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def insert_scraped_document(db: LegalDocumentDB, doc, doc_id_override: str = None):
    """Convert scraped LegalDocument dataclass to DB models and insert.

    Args:
        db: Database manager instance
        doc: Scraped LegalDocument dataclass
        doc_id_override: Override the document ID (default: derived from so_hieu)
    """
    doc_id = doc_id_override or make_document_id(doc.so_hieu)

    with db.SessionLocal() as session:
        existing = session.get(LegalDocumentModel, doc_id)
        if existing:
            logger.warning(f"Document {doc_id} already exists. Skipping.")
            return False

        doc_model = LegalDocumentModel(
            id=doc_id,
            so_hieu=doc.so_hieu,
            title=doc.title,
            loai_van_ban=doc.loai_van_ban,
            co_quan_ban_hanh=doc.co_quan_ban_hanh,
            nguoi_ky=doc.nguoi_ky,
            ngay_ban_hanh=doc.ngay_ban_hanh,
            ngay_hieu_luc=doc.ngay_hieu_luc,
            tinh_trang=doc.tinh_trang,
            raw_text=doc.raw_text,
            source_url=doc.url,
        )
        session.add(doc_model)

        article_position = 0
        seen_ids = set()

        for ch_idx, chapter in enumerate(doc.chapters):
            # Use 1-based position as chapter number if Roman numeral collides
            chapter_id = make_chapter_id(doc_id, chapter.number)
            if chapter_id in seen_ids:
                chapter_id = f"{doc_id}:c{ch_idx + 1}"
            seen_ids.add(chapter_id)

            chapter_model = LegalChapterModel(
                id=chapter_id,
                document_id=doc_id,
                chapter_number=chapter.number,
                title=chapter.title,
                position=ch_idx + 1,
            )
            session.add(chapter_model)

            for article in chapter.articles:
                article_position += 1
                _insert_article(session, doc_id, chapter_id, None, article, article_position, seen_ids)

            for sec_idx, section in enumerate(chapter.sections):
                section_id = make_section_id(doc_id, chapter.number, section.number)
                if section_id in seen_ids:
                    section_id = f"{chapter_id}:m{sec_idx + 1}"
                seen_ids.add(section_id)

                section_model = LegalSectionModel(
                    id=section_id,
                    chapter_id=chapter_id,
                    section_number=section.number,
                    title=section.title,
                    position=sec_idx + 1,
                )
                session.add(section_model)

                for article in section.articles:
                    article_position += 1
                    _insert_article(session, doc_id, chapter_id, section_id, article, article_position, seen_ids)

        for article in doc.articles:
            article_position += 1
            _insert_article(session, doc_id, None, None, article, article_position, seen_ids)

        session.commit()
        logger.info(f"Inserted document {doc_id} with {article_position} articles")
        return True


def _insert_article(session, doc_id, chapter_id, section_id, article, position, seen_ids=None):
    """Insert a single article with its clauses and points."""
    if seen_ids is None:
        seen_ids = set()

    article_id = make_article_id(doc_id, article.number)
    if article_id in seen_ids:
        logger.warning(f"Duplicate article ID {article_id}, appending position suffix")
        article_id = f"{article_id}_{position}"
    seen_ids.add(article_id)

    raw_text = article.content or ""
    article_model = LegalArticleModel(
        id=article_id,
        document_id=doc_id,
        chapter_id=chapter_id,
        section_id=section_id,
        article_number=article.number,
        title=article.title,
        content=article.content,
        raw_text=raw_text,
        position=position,
    )
    session.add(article_model)

    for clause in article.clauses:
        clause_id = make_clause_id(article_id, clause.number)
        if clause_id in seen_ids:
            clause_id = f"{clause_id}_{id(clause) % 10000}"
        seen_ids.add(clause_id)

        clause_model = LegalClauseModel(
            id=clause_id,
            article_id=article_id,
            clause_number=clause.number,
            content=clause.content,
        )
        session.add(clause_model)

        for point in clause.points:
            point_id = make_point_id(clause_id, point.letter)
            if point_id in seen_ids:
                point_id = f"{point_id}_{id(point) % 10000}"
            seen_ids.add(point_id)

            point_model = LegalPointModel(
                id=point_id,
                clause_id=clause_id,
                point_letter=point.letter,
                content=point.content,
            )
            session.add(point_model)


async def scrape_and_ingest(url: str, db_path: str, doc_id_override: str = None, dry_run: bool = False):
    """Scrape a TVPL URL and ingest into database."""
    logger.info(f"Scraping: {url}")

    scraper = TVPLScraper(headless=True, save_raw_html=True)
    docs = await scraper.scrape_batch([url])

    if not docs:
        logger.error("No documents scraped. Check URL and try again.")
        return

    doc = docs[0]
    logger.info(f"Scraped: {doc.title}")
    logger.info(f"  so_hieu: {doc.so_hieu}")
    logger.info(f"  Chapters: {len(doc.chapters)}")
    total_articles = sum(
        len(ch.articles) + sum(len(s.articles) for s in ch.sections)
        for ch in doc.chapters
    ) + len(doc.articles)
    logger.info(f"  Articles: {total_articles}")
    logger.info(f"  Scrape errors: {len(doc.scrape_errors)}")
    for err in doc.scrape_errors:
        logger.warning(f"    - {err}")

    if dry_run:
        logger.info("Dry run - not inserting into DB")
        return

    db = LegalDocumentDB(db_path)
    doc_id = doc_id_override or make_document_id(doc.so_hieu)

    success = insert_scraped_document(db, doc, doc_id_override=doc_id)
    if success:
        # Verify insertion
        stats = db.count_stats()
        logger.info(f"DB stats after ingestion: {stats}")

        articles = db.get_articles_for_document(doc_id)
        logger.info(f"Articles for {doc_id}: {len(articles)}")


def main():
    parser = argparse.ArgumentParser(description="Ingest legal document from TVPL")
    parser.add_argument("--url", required=True, help="TVPL URL to scrape")
    parser.add_argument("--db", default="data/legal_docs.db", help="Database path")
    parser.add_argument("--doc-id", default=None, help="Override document ID")
    parser.add_argument("--dry-run", action="store_true", help="Scrape only, no DB insert")
    args = parser.parse_args()

    asyncio.run(scrape_and_ingest(args.url, args.db, args.doc_id, args.dry_run))


if __name__ == "__main__":
    main()
