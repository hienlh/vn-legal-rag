"""Configuration and document registry for education PDF extraction."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_SOURCE = Path("/Users/hienlh/Projects/Master/luan-van/raw_data")
DATA_DIR = PROJECT_ROOT / "data" / "education"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
TEXT_DIR = DATA_DIR / "extracted_text"
STRUCTURED_DIR = DATA_DIR / "structured"

# LLM settings for vision OCR — reads from .env
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# PDF to image settings
IMAGE_DPI = 300
IMAGE_FORMAT = "png"

# Document registry
DOCUMENTS = [
    {
        "id": "218-QD-DHQG",
        "file": "218_qd_dhqg_quy_che_bo_sung_dao_tao_thac_si_nam_2022.pdf",
        "so_hieu": "218/QĐ-ĐHQG",
        "title": "Quy chế sửa đổi bổ sung quy định Giáo viên hướng dẫn",
        "co_quan": "ĐHQG-HCM",
        "ngay_ban_hanh": "2024-03-15",
        "format": "text",
    },
    {
        "id": "270-QD-DHCNTT",
        "file": "270-qd-dhcntt-25-4-2022.pdf",
        "so_hieu": "270/QĐ-ĐHCNTT",
        "title": "Quy chế đào tạo trình độ thạc sĩ UIT",
        "co_quan": "UIT",
        "ngay_ban_hanh": "2022-04-25",
        "format": "scan",
    },
    {
        "id": "160-QD-DHQG",
        "file": "160.-quy-che-dao-tao-trinh-do-thac-si-nam-2017_cua_dhqg_hcm.pdf",
        "so_hieu": "160/QĐ-ĐHQG",
        "title": "Quy chế đào tạo trình độ thạc sĩ ĐHQG-HCM (khoá 2021)",
        "co_quan": "ĐHQG-HCM",
        "ngay_ban_hanh": "2017-03-24",
        "format": "scan",
    },
    {
        "id": "1393-QD-DHQG",
        "file": "quy_che_dhqg-_thac_si_1393_qd_dhqg_03_11_2021_1.pdf",
        "so_hieu": "1393/QĐ-ĐHQG",
        "title": "Quy chế đào tạo trình độ thạc sĩ ĐHQG-HCM (khoá 2022+)",
        "co_quan": "ĐHQG-HCM",
        "ngay_ban_hanh": "2021-11-03",
        "format": "scan",
    },
    {
        "id": "851-QD-DHCNTT",
        "file": "851-qd-dhcntt-14-08-2024-scan.pdf",
        "so_hieu": "851/QĐ-ĐHCNTT",
        "title": "Quy định liêm chính học thuật UIT",
        "co_quan": "UIT",
        "ngay_ban_hanh": "2024-08-14",
        "format": "scan",
    },
]


def get_doc_by_id(doc_id: str) -> dict | None:
    """Get document config by ID."""
    for doc in DOCUMENTS:
        if doc["id"] == doc_id:
            return doc
    return None


def get_source_pdf_path(doc: dict) -> Path:
    """Get source PDF path from raw_data."""
    return RAW_DATA_SOURCE / doc["file"]
