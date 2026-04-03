#!/usr/bin/env python3
"""
Load education regulation documents from structured JSON into SQLite DB.

Reads from data/education/structured/*.json and inserts into
data/education/edu_docs.db using the same schema as legal documents.

Usage:
    python scripts/load-education-documents.py
    python scripts/load-education-documents.py --db data/education/edu_docs.db
    python scripts/load-education-documents.py --doc 270-QD-DHCNTT
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vn_legal_rag.offline import (
    LegalDocumentDB,
    LegalDocumentModel,
    LegalChapterModel,
    LegalArticleModel,
    LegalClauseModel,
    LegalPointModel,
    make_chapter_id,
    make_article_id,
    make_clause_id,
    make_point_id,
)

STRUCTURED_DIR = Path("data/education/structured")
DEFAULT_DB = "data/education/edu_docs.db"

# Document metadata from edu_pdf_extractor/config.py
DOC_METADATA = {
    "218-QD-DHQG": {
        "so_hieu": "218/QĐ-ĐHQG",
        "title": "Quy chế sửa đổi bổ sung quy định GVHD",
        "co_quan": "ĐHQG-HCM",
        "ngay_ban_hanh": "2024-03-15",
        "loai_van_ban": "Quyết định",
    },
    "270-QD-DHCNTT": {
        "so_hieu": "270/QĐ-ĐHCNTT",
        "title": "Quy chế đào tạo trình độ thạc sĩ UIT",
        "co_quan": "UIT",
        "ngay_ban_hanh": "2022-04-25",
        "loai_van_ban": "Quyết định",
    },
    "160-QD-DHQG": {
        "so_hieu": "160/QĐ-ĐHQG",
        "title": "Quy chế đào tạo trình độ thạc sĩ ĐHQG-HCM (khoá 2021)",
        "co_quan": "ĐHQG-HCM",
        "ngay_ban_hanh": "2017-03-24",
        "loai_van_ban": "Quyết định",
    },
    "1393-QD-DHQG": {
        "so_hieu": "1393/QĐ-ĐHQG",
        "title": "Quy chế đào tạo trình độ thạc sĩ ĐHQG-HCM (khoá 2022+)",
        "co_quan": "ĐHQG-HCM",
        "ngay_ban_hanh": "2021-11-03",
        "loai_van_ban": "Quyết định",
    },
    "851-QD-DHCNTT": {
        "so_hieu": "851/QĐ-ĐHCNTT",
        "title": "Quy định liêm chính học thuật UIT",
        "co_quan": "UIT",
        "ngay_ban_hanh": "2024-08-14",
        "loai_van_ban": "Quyết định",
    },
}


def load_document(db: LegalDocumentDB, json_path: Path) -> dict:
    """Load one structured JSON into the database. Returns stats."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data["doc_id"]
    meta = DOC_METADATA.get(doc_id, {})
    ngay = meta.get("ngay_ban_hanh", "")
    ngay_date = date.fromisoformat(ngay) if ngay else None

    stats = {"chapters": 0, "articles": 0, "clauses": 0, "points": 0}

    with db.SessionLocal() as session, session.no_autoflush:
        # Check if document already exists
        existing = session.get(LegalDocumentModel, doc_id)
        if existing:
            print(f"  [skip] {doc_id} already exists")
            return stats

        # Create document
        doc_model = LegalDocumentModel(
            id=doc_id,
            so_hieu=meta.get("so_hieu", data.get("so_hieu", "")),
            title=meta.get("title", data.get("title", "")),
            loai_van_ban=meta.get("loai_van_ban", "Quyết định"),
            co_quan_ban_hanh=meta.get("co_quan", data.get("co_quan", "")),
            ngay_ban_hanh=ngay_date,
        )
        session.add(doc_model)

        article_pos = 0
        seen_ids = set()

        # Handle documents with chapters
        for ch_idx, chapter in enumerate(data.get("chapters", [])):
            ch_num = chapter.get("number", str(ch_idx + 1))
            ch_id = make_chapter_id(doc_id, ch_num)

            if ch_id in seen_ids:
                continue
            seen_ids.add(ch_id)

            ch_model = LegalChapterModel(
                id=ch_id,
                document_id=doc_id,
                chapter_number=str(ch_num),
                title=chapter.get("title", ""),
                position=ch_idx,
            )
            session.add(ch_model)
            stats["chapters"] += 1

            for art in chapter.get("articles", []):
                art_num = art["number"]
                art_id = make_article_id(doc_id, art_num)

                # Handle duplicate article numbers (OCR parsing artifacts)
                if art_id in seen_ids:
                    # Merge content into existing article
                    existing = session.get(LegalArticleModel, art_id)
                    if existing:
                        existing.content += "\n" + art.get("content", "")
                        existing.raw_text = existing.content
                        print(f"  [merge] duplicate {art_id} → appended to existing")
                    continue
                seen_ids.add(art_id)

                art_model = LegalArticleModel(
                    id=art_id,
                    document_id=doc_id,
                    chapter_id=ch_id,
                    article_number=art_num,
                    title=art.get("title", ""),
                    content=art.get("content", ""),
                    raw_text=art.get("content", ""),
                    position=article_pos,
                )
                session.add(art_model)
                stats["articles"] += 1
                article_pos += 1

                _load_clauses(session, art_id, art.get("clauses", []), stats, seen_ids)

        # Handle documents with top-level articles (no chapters, e.g. QD 218)
        for art in data.get("articles", []):
            art_num = art["number"]
            art_id = make_article_id(doc_id, art_num)

            if art_id in seen_ids:
                existing = session.get(LegalArticleModel, art_id)
                if existing:
                    existing.content += "\n" + art.get("content", "")
                    existing.raw_text = existing.content
                    print(f"  [merge] duplicate {art_id} → appended to existing")
                continue
            seen_ids.add(art_id)

            art_model = LegalArticleModel(
                id=art_id,
                document_id=doc_id,
                article_number=art_num,
                title=art.get("title", ""),
                content=art.get("content", ""),
                raw_text=art.get("content", ""),
                position=article_pos,
            )
            session.add(art_model)
            stats["articles"] += 1
            article_pos += 1

            _load_clauses(session, art_id, art.get("clauses", []), stats, seen_ids)

        session.commit()

    return stats


def _load_clauses(session, art_id: str, clauses: list, stats: dict, seen_ids: set):
    """Load clauses and points for an article."""
    for cl_idx, clause in enumerate(clauses):
        cl_num = clause.get("number", cl_idx + 1)
        cl_id = make_clause_id(art_id, cl_num)

        if cl_id in seen_ids:
            continue
        seen_ids.add(cl_id)

        cl_model = LegalClauseModel(
            id=cl_id,
            article_id=art_id,
            clause_number=cl_num,
            content=clause.get("content", ""),
            raw_text=clause.get("content", ""),
            position=cl_idx,
        )
        session.add(cl_model)
        stats["clauses"] += 1

        for pt_idx, point in enumerate(clause.get("points", [])):
            pt_letter = point.get("letter", chr(ord("a") + pt_idx))
            pt_id = make_point_id(cl_id, pt_letter)

            if pt_id in seen_ids:
                continue
            seen_ids.add(pt_id)

            pt_model = LegalPointModel(
                id=pt_id,
                clause_id=cl_id,
                point_letter=pt_letter,
                content=point.get("content", ""),
                raw_text=point.get("content", ""),
                position=pt_idx,
            )
            session.add(pt_model)
            stats["points"] += 1


def main():
    parser = argparse.ArgumentParser(description="Load education docs into SQLite")
    parser.add_argument("--db", default=DEFAULT_DB, help="Database path")
    parser.add_argument("--doc", help="Load single document by ID")
    args = parser.parse_args()

    db = LegalDocumentDB(args.db)
    print(f"Database: {args.db}")

    json_files = sorted(STRUCTURED_DIR.glob("*.json"))
    if args.doc:
        json_files = [f for f in json_files if f.stem == args.doc]

    if not json_files:
        print("No JSON files found!")
        return 1

    total = {"chapters": 0, "articles": 0, "clauses": 0, "points": 0}

    for jf in json_files:
        print(f"\nLoading {jf.stem}...")
        stats = load_document(db, jf)
        for k in total:
            total[k] += stats[k]
        print(f"  chapters={stats['chapters']}, articles={stats['articles']}, "
              f"clauses={stats['clauses']}, points={stats['points']}")

    print(f"\n{'='*50}")
    print(f"TOTAL: {len(json_files)} documents")
    print(f"  Chapters: {total['chapters']}")
    print(f"  Articles: {total['articles']}")
    print(f"  Clauses:  {total['clauses']}")
    print(f"  Points:   {total['points']}")

    # Verify
    db_stats = db.count_stats()
    print(f"\nDB verification: {db_stats}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
