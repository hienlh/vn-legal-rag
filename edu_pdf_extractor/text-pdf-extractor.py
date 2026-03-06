"""Extract text from text-based PDFs using pdfplumber.

Used for QD 218 (the only text PDF in our dataset).
"""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def extract_text_pdf(pdf_path: str | Path) -> str:
    """Extract text from a text-based PDF using pdfplumber.

    Joins all pages with double newline separator.
    Strips page headers/footers if detected.
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting text from {pdf_path.name} using pdfplumber...")

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Strip common page artifacts
            text = _clean_page_text(text, page_num=i)
            if text.strip():
                pages_text.append(text.strip())

    full_text = "\n\n".join(pages_text)
    logger.info(
        f"Extracted {len(full_text)} chars from {len(pages_text)} pages"
    )
    return full_text


def _clean_page_text(text: str, page_num: int) -> str:
    """Remove common page artifacts: page numbers, repeated headers."""
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()
        # Skip standalone page numbers
        if stripped.isdigit() and len(stripped) <= 3:
            continue
        # Skip common header/footer patterns
        if stripped.startswith("Trang ") and stripped[6:].strip().isdigit():
            continue
        cleaned.append(line)

    return "\n".join(cleaned)
