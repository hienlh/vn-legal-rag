#!/usr/bin/env python3
"""
Batch ingest all 13 legal documents from TVPL into legal_docs.db.

Scrapes each document from thuvienphapluat.vn and inserts into SQLite database
with full hierarchical structure (document → chapter → section → article → clause → point).

Usage:
    python scripts/batch-ingest-all-legal-documents.py
    python scripts/batch-ingest-all-legal-documents.py --dry-run
    python scripts/batch-ingest-all-legal-documents.py --only 59-2020-QH14,01-2021-ND
"""

import argparse
import asyncio
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from vn_legal_rag.offline import LegalDocumentDB, TVPLScraper

_ingest_mod = importlib.import_module("ingest-single-document-from-tvpl")
insert_scraped_document = _ingest_mod.insert_scraped_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LEGAL_DOCUMENTS = [
    # Enterprise law domain (5 core docs used in benchmark)
    {
        "id": "59-2020-QH14",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Law-59-2020-QH14-Enterprises-451799.aspx",
        "name": "Luật Doanh nghiệp 2020",
    },
    {
        "id": "01-2021-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-01-2021-ND-CP-dang-ky-doanh-nghiep-283247.aspx",
        "name": "NĐ 01/2021 - Đăng ký doanh nghiệp",
    },
    {
        "id": "47-2021-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-47-2021-ND-CP-huong-dan-Luat-Doanh-nghiep-470561.aspx",
        "name": "NĐ 47/2021 - Hướng dẫn Luật DN",
    },
    {
        "id": "23-2022-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-23-2022-ND-CP-thanh-lap-doanh-nghiep-do-Nha-nuoc-nam-giu-100-von-dieu-le-509241.aspx",
        "name": "NĐ 23/2022 - DN nhà nước 100% vốn",
    },
    {
        "id": "65-2022-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-65-2022-ND-CP-sua-doi-Nghi-dinh-153-2020-ND-CP-chao-ban-giao-dich-trai-phieu-doanh-nghiep-529835.aspx",
        "name": "NĐ 65/2022 - Trái phiếu DN",
    },
    {
        "id": "16-2023-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-16-2023-ND-CP-to-chuc-quan-ly-doanh-nghiep-truc-tiep-phuc-vu-quoc-phong-an-ninh-564517.aspx",
        "name": "NĐ 16/2023 - DN quốc phòng an ninh",
    },
    {
        "id": "89-2024-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-89-2024-ND-CP-chuyen-doi-cong-ty-nha-nuoc-thanh-cong-ty-trach-nhiem-huu-han-549549.aspx",
        "name": "NĐ 89/2024 - Chuyển đổi công ty NN",
    },
    {
        "id": "44-2025-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-44-2025-ND-CP-quan-ly-lao-dong-tien-luong-tien-thuong-trong-doanh-nghiep-nha-nuoc-624938.aspx",
        "name": "NĐ 44/2025 - Tiền lương DN nhà nước",
    },
    {
        "id": "168-2025-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-168-2025-ND-CP-dang-ky-doanh-nghiep-623074.aspx",
        "name": "NĐ 168/2025 - Đăng ký DN (thay NĐ 01)",
    },
    {
        "id": "248-2025-ND",
        "url": "https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Nghi-dinh-248-2025-ND-CP-che-do-tien-luong-Kiem-soat-vien-trong-doanh-nghiep-nha-nuoc-673078.aspx",
        "name": "NĐ 248/2025 - Lương kiểm soát viên",
    },
    # Traffic law domain (3 docs)
    {
        "id": "36-2024-QH15",
        "url": "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-trat-tu-an-toan-giao-thong-duong-bo-2024-so-36-2024-QH15-444251.aspx",
        "name": "Luật TTATGT đường bộ 2024",
    },
    {
        "id": "168-2024-ND-CP",
        "url": "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-168-2024-ND-CP-xu-phat-vi-pham-hanh-chinh-an-toan-giao-thong-duong-bo-619502.aspx",
        "name": "NĐ 168/2024 - Xử phạt giao thông",
    },
    {
        "id": "100-2019-ND-CP",
        "url": "https://thuvienphapluat.vn/van-ban/Vi-pham-hanh-chinh/Nghi-dinh-100-2019-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-giao-thong-duong-bo-va-duong-sat-426369.aspx",
        "name": "NĐ 100/2019 - Xử phạt GT (cũ)",
    },
]


async def main():
    parser = argparse.ArgumentParser(description="Batch ingest legal documents from TVPL")
    parser.add_argument("--db", default="data/legal_docs.db", help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Scrape only, no DB insert")
    parser.add_argument("--only", type=str, help="Comma-separated doc IDs to ingest")
    parser.add_argument("--skip", type=str, help="Comma-separated doc IDs to skip")
    args = parser.parse_args()

    docs_to_ingest = LEGAL_DOCUMENTS
    if args.only:
        only_ids = set(args.only.split(","))
        docs_to_ingest = [d for d in docs_to_ingest if d["id"] in only_ids]
    if args.skip:
        skip_ids = set(args.skip.split(","))
        docs_to_ingest = [d for d in docs_to_ingest if d["id"] not in skip_ids]

    logger.info(f"Will ingest {len(docs_to_ingest)} documents into {args.db}")

    db = LegalDocumentDB(args.db)
    scraper = TVPLScraper(headless=True, save_raw_html=False)

    success = 0
    failed = []

    for i, doc_info in enumerate(docs_to_ingest, 1):
        doc_id = doc_info["id"]
        url = doc_info["url"]
        name = doc_info["name"]

        logger.info(f"\n[{i}/{len(docs_to_ingest)}] {doc_id} - {name}")
        logger.info(f"  URL: {url}")

        try:
            docs = await scraper.scrape_batch([url])
            if not docs:
                logger.error(f"  FAILED: No content scraped")
                failed.append(doc_id)
                continue

            doc = docs[0]
            total_articles = sum(
                len(ch.articles) + sum(len(s.articles) for s in ch.sections)
                for ch in doc.chapters
            ) + len(doc.articles)

            logger.info(f"  Scraped: {len(doc.chapters)} chapters, {total_articles} articles")

            if doc.scrape_errors:
                for err in doc.scrape_errors:
                    logger.warning(f"  Warning: {err}")

            if args.dry_run:
                logger.info(f"  [DRY RUN] Would insert {doc_id}")
            else:
                result = insert_scraped_document(db, doc, doc_id_override=doc_id)
                if result:
                    logger.info(f"  OK: Inserted {doc_id}")
                    success += 1
                else:
                    logger.warning(f"  SKIPPED: {doc_id} already exists")
                    success += 1

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            failed.append(doc_id)

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {success} success, {len(failed)} failed out of {len(docs_to_ingest)}")
    if failed:
        logger.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    asyncio.run(main())
