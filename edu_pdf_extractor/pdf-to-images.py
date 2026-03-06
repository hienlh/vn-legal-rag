"""Convert PDF pages to images for LLM vision OCR.

Uses pdf2image (poppler) to render each page as PNG at configurable DPI.
Skips conversion for text-based PDFs (detected via pdfplumber).
"""

import logging
from pathlib import Path
from typing import List

import pdfplumber
from pdf2image import convert_from_path

from .config import IMAGE_DPI, IMAGE_FORMAT, IMAGES_DIR

logger = logging.getLogger(__name__)

# Minimum chars per page to consider it a text PDF
TEXT_PDF_THRESHOLD = 100


def is_text_pdf(pdf_path: str | Path) -> bool:
    """Auto-detect if PDF contains extractable text or is scanned.

    Checks first 3 pages — if average chars > threshold, it's text-based.
    """
    pdf_path = Path(pdf_path)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = min(3, len(pdf.pages))
            total_chars = 0
            for i in range(pages_to_check):
                text = pdf.pages[i].extract_text() or ""
                total_chars += len(text.strip())
            avg_chars = total_chars / pages_to_check if pages_to_check > 0 else 0
            is_text = avg_chars > TEXT_PDF_THRESHOLD
            logger.info(
                f"{pdf_path.name}: avg {avg_chars:.0f} chars/page "
                f"-> {'text' if is_text else 'scan'} PDF"
            )
            return is_text
    except Exception as e:
        logger.warning(f"Error checking PDF type for {pdf_path}: {e}")
        return False


def pdf_to_images(pdf_path: str | Path, doc_id: str, dpi: int = IMAGE_DPI) -> List[Path]:
    """Convert PDF to page images.

    Args:
        pdf_path: Path to the PDF file
        doc_id: Document ID for output directory naming
        dpi: Resolution for rendering (default 300)

    Returns:
        List of paths to generated page images, sorted by page number
    """
    pdf_path = Path(pdf_path)
    output_dir = IMAGES_DIR / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if images already exist (cache)
    existing = sorted(output_dir.glob(f"page_*.{IMAGE_FORMAT}"))
    if existing:
        logger.info(f"{doc_id}: {len(existing)} cached page images found, skipping conversion")
        return existing

    logger.info(f"{doc_id}: Converting {pdf_path.name} to images at {dpi} DPI...")
    images = convert_from_path(str(pdf_path), dpi=dpi)

    image_paths = []
    for i, img in enumerate(images, start=1):
        img_path = output_dir / f"page_{i:03d}.{IMAGE_FORMAT}"
        img.save(str(img_path), IMAGE_FORMAT.upper())
        image_paths.append(img_path)

    logger.info(f"{doc_id}: Generated {len(image_paths)} page images")
    return image_paths


def get_page_count(pdf_path: str | Path) -> int:
    """Get total number of pages in a PDF."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        return len(pdf.pages)
