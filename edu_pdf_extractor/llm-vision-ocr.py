"""LLM Vision OCR: extract text from scanned PDF page images.

Sends page images to Gemini Flash (or Claude) for text extraction.
Caches results per page to avoid re-processing on retry.
Includes retry with exponential backoff for rate limit (429) errors.
Logs all API calls with token usage and estimated cost.
"""

import base64
import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Cost per million tokens (USD) — updated 2026-03
# https://ai.google.dev/pricing, https://docs.anthropic.com/en/docs/about-claude/models
PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}

# Log file path
_LOG_DIR = Path(__file__).parent.parent / "data" / "education"
_LOG_FILE = _LOG_DIR / "api_call_log.csv"


def _log_api_call(
    provider: str, model: str, image_name: str,
    input_tokens: int, output_tokens: int,
    duration_s: float, output_chars: int,
) -> None:
    """Append API call details to CSV log file."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not _LOG_FILE.exists()

    pricing = PRICING.get(model, {"input": 0, "output": 0})
    est_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "image": image_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": f"{duration_s:.1f}",
        "output_chars": output_chars,
        "est_cost_usd": f"{est_cost:.6f}",
    }

    with open(_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(
        f"  API: {input_tokens} in + {output_tokens} out tokens, "
        f"${est_cost:.6f}, {duration_s:.1f}s"
    )

# OCR prompt for LLM vision
OCR_PROMPT = """Trích xuất TOÀN BỘ văn bản từ hình ảnh trang tài liệu pháp quy tiếng Việt này.

Yêu cầu:
- Giữ nguyên định dạng: xuống dòng, thụt lề, đánh số
- Giữ nguyên chính xác các dấu tiếng Việt (ă, â, đ, ê, ô, ơ, ư, dấu sắc, huyền, hỏi, ngã, nặng)
- KHÔNG tóm tắt hay diễn giải - trích xuất nguyên văn
- KHÔNG thêm bất kỳ chú thích hay giải thích nào
- Bỏ qua watermark, header/footer lặp lại, số trang

Chỉ trả về văn bản trích xuất được, không có gì khác."""

# Rate limiting and retry settings
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 5
INITIAL_BACKOFF = 30.0  # Start with 30s for rate limit errors


def _get_cache_path(image_path: Path) -> Path:
    """Get cache file path for a page image."""
    cache_dir = image_path.parent / ".ocr_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{image_path.stem}.txt"


def _load_cached(image_path: Path) -> Optional[str]:
    """Load cached OCR result if exists."""
    cache_path = _get_cache_path(image_path)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def _save_cache(image_path: Path, text: str) -> None:
    """Save OCR result to cache."""
    cache_path = _get_cache_path(image_path)
    cache_path.write_text(text, encoding="utf-8")


def _ocr_with_gemini(image_path: Path, model: str) -> tuple[str, dict]:
    """Extract text from image using Gemini vision API (google-genai SDK).

    Returns (text, usage_dict) where usage_dict has input_tokens and output_tokens.
    """
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

    client = genai.Client(api_key=api_key)

    image_bytes = image_path.read_bytes()
    mime_type = f"image/{image_path.suffix.lstrip('.')}"

    response = client.models.generate_content(
        model=model,
        contents=[
            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            OCR_PROMPT,
        ],
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )

    # Extract token usage from response metadata
    usage = {"input_tokens": 0, "output_tokens": 0}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        usage["input_tokens"] = getattr(um, 'prompt_token_count', 0) or 0
        usage["output_tokens"] = getattr(um, 'candidates_token_count', 0) or 0

    # Gemini 2.5+ thinking models may have None for response.text
    if response.text:
        return response.text.strip(), usage
    # Fallback: extract from candidates parts (may be None for blank pages)
    if response.candidates and response.candidates[0].content:
        parts = response.candidates[0].content.parts or []
        texts = [p.text for p in parts if hasattr(p, 'text') and p.text]
        if texts:
            return "\n".join(texts).strip(), usage
    return "", usage


def _ocr_with_anthropic(image_path: Path, model: str) -> tuple[str, dict]:
    """Extract text from image using Anthropic Claude vision API.

    Returns (text, usage_dict) where usage_dict has input_tokens and output_tokens.
    """
    import anthropic

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    image_bytes = image_path.read_bytes()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    media_type = f"image/{image_path.suffix.lstrip('.')}"

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
    )

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response.content[0].text.strip(), usage


def _call_with_retry(fn, *args, max_retries: int = MAX_RETRIES, **kwargs) -> str:
    """Call function with exponential backoff retry on rate limit errors."""
    backoff = INITIAL_BACKOFF

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower()

            if not is_rate_limit or attempt >= max_retries:
                raise

            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
                f"Waiting {backoff:.0f}s..."
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)  # Cap at 5 minutes


def ocr_page(
    image_path: Path,
    provider: str = "gemini",
    model: Optional[str] = None,
) -> str:
    """Extract text from a single page image using LLM vision.

    Results are cached per page to avoid re-processing.
    Retries automatically on rate limit (429) errors.
    """
    # Check cache first
    cached = _load_cached(image_path)
    if cached is not None:
        logger.debug(f"Cache hit: {image_path.name}")
        return cached

    logger.info(f"OCR: {image_path.name} ({provider})")

    if provider == "gemini":
        from .config import GEMINI_MODEL
        used_model = model or GEMINI_MODEL
        t0 = time.time()
        text, usage = _call_with_retry(_ocr_with_gemini, image_path, used_model)
        duration = time.time() - t0
    elif provider == "anthropic":
        from .config import ANTHROPIC_MODEL
        used_model = model or ANTHROPIC_MODEL
        t0 = time.time()
        text, usage = _call_with_retry(_ocr_with_anthropic, image_path, used_model)
        duration = time.time() - t0
    else:
        raise ValueError(f"Unknown provider: {provider}")

    _log_api_call(
        provider=provider, model=used_model, image_name=image_path.name,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        duration_s=duration, output_chars=len(text),
    )

    # Cache result
    _save_cache(image_path, text)

    # Rate limit between requests
    time.sleep(REQUEST_DELAY_SECONDS)

    return text


def ocr_document(
    image_paths: List[Path],
    provider: str = "gemini",
    model: Optional[str] = None,
) -> str:
    """Extract text from all page images of a document.

    Processes pages sequentially, joining results with double newline.
    Cached pages are loaded instantly.
    """
    sorted_paths = sorted(image_paths)
    cached_count = sum(1 for p in sorted_paths if _load_cached(p) is not None)
    total = len(sorted_paths)

    if cached_count > 0:
        logger.info(f"{cached_count}/{total} pages cached")

    pages_text = []
    for i, img_path in enumerate(sorted_paths, 1):
        is_cached = _load_cached(img_path) is not None
        if not is_cached:
            logger.info(f"Processing page {i}/{total}...")

        text = ocr_page(img_path, provider=provider, model=model)
        if text.strip():
            pages_text.append(text.strip())

    logger.info(f"OCR complete: {len(pages_text)} pages with text")
    return "\n\n".join(pages_text)
