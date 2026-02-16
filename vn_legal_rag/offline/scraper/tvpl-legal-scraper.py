"""
Web scraper for thuvienphapluat.vn (TVPL).

Uses Playwright headless browser to bypass anti-scraping measures.
Includes rate limiting (1 req/2s) and retry logic.

Dependencies (optional):
    pip install playwright beautifulsoup4
    playwright install chromium
"""

import asyncio
import json
import re
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils import get_logger

# Import from kebab-case filenames
_base_mod = import_module(".base-legal-scraper", "vn_legal_rag.offline.scraper")
BaseLegalScraper = _base_mod.BaseLegalScraper
LegalDocument = _base_mod.LegalDocument

_hierarchy_mod = import_module(".legal-hierarchy-extractor", "vn_legal_rag.offline.scraper")
HierarchyExtractor = _hierarchy_mod.HierarchyExtractor

logger = get_logger("tvpl_scraper")


class TVPLScraper(BaseLegalScraper):
    """
    Scraper for thuvienphapluat.vn Vietnamese legal documents.

    Features:
    - Playwright headless browser for JavaScript rendering
    - Rate limiting (configurable, default 1 req/2s)
    - Retry logic with exponential backoff
    - Saves both raw HTML and parsed content
    - Handles partial failures gracefully

    Usage:
        scraper = TVPLScraper()
        doc = await scraper.scrape("https://thuvienphapluat.vn/...")

        # Batch scrape with rate limiting
        docs = await scraper.scrape_batch(urls)
    """

    # CSS selectors for TVPL website (multiple fallbacks)
    # NOTE: .content1 contains actual document text, divNoiDung contains TOC
    CONTENT_SELECTORS = [
        ".content1",  # Primary - actual document content
        ".contentDoc",  # Alternative content div
        "#ctl00_Content_ThongTinVB_divNoiDung .content1",
        "#toanvancontent",
        "[class*='NoiDung']",
    ]

    SELECTORS = {
        "title": "h1.title, .doc-title h1, .title-vb, h1",
        "metadata": ".thuoc-tinh, .doc-info",
        "so_hieu": '[class*="so-hieu"], .sohieu',
        "ngay_ban_hanh": '[class*="ngay-ban-hanh"], .ngaybanhanh',
        "co_quan": '[class*="co-quan"], .coquanbanhanh',
        "nguoi_ky": '[class*="nguoi-ky"], .nguoiky',
        "loai_van_ban": '[class*="loai-van-ban"], .loaivanban',
        "tinh_trang": '[class*="tinh-trang"], .tinhtrang',
        "ngay_hieu_luc": '[class*="ngay-hieu-luc"], .ngayhieuluc',
    }

    # Common user agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        rate_limit_seconds: float = 2.0,
        max_retries: int = 3,
        save_raw_html: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize TVPL scraper.

        Args:
            headless: Run browser in headless mode
            timeout: Page load timeout in milliseconds
            rate_limit_seconds: Delay between requests (default 2s)
            max_retries: Max retry attempts for failed requests
            save_raw_html: Whether to save raw HTML files
            output_dir: Directory for saving scraped data
        """
        self.headless = headless
        self.timeout = timeout
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.save_raw_html = save_raw_html
        self.output_dir = output_dir or Path("./scraped_legal_docs")
        self._last_request_time: Optional[float] = None
        self._user_agent_index = 0

    def _get_next_user_agent(self) -> str:
        """Rotate through user agents."""
        ua = self.USER_AGENTS[self._user_agent_index]
        self._user_agent_index = (self._user_agent_index + 1) % len(self.USER_AGENTS)
        return ua

    async def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = asyncio.get_event_loop().time() - self._last_request_time
            if elapsed < self.rate_limit_seconds:
                wait_time = self.rate_limit_seconds - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        self._last_request_time = asyncio.get_event_loop().time()

    async def fetch(self, url: str) -> str:
        """
        Fetch HTML content using Playwright with retry logic.

        Args:
            url: Target URL

        Returns:
            Raw HTML content

        Raises:
            Exception: If all retries fail
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required. Install with: pip install playwright && playwright install chromium"
            )

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            await self._wait_for_rate_limit()

            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=self.headless)
                    context = await browser.new_context(
                        user_agent=self._get_next_user_agent(),
                        viewport={"width": 1920, "height": 1080},
                        java_script_enabled=True,
                    )
                    page = await context.new_page()

                    try:
                        logger.info(f"Fetching (attempt {attempt + 1}): {url}")
                        await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)

                        # Wait for dynamic content to load
                        await asyncio.sleep(3)

                        # Try to find content with multiple selectors
                        content_found = False
                        for selector in self.CONTENT_SELECTORS:
                            try:
                                el = await page.query_selector(selector)
                                if el:
                                    text = await el.text_content()
                                    if text and len(text) > 1000:
                                        content_found = True
                                        logger.debug(f"Found content with selector: {selector}")
                                        break
                            except Exception:
                                continue

                        if not content_found:
                            raise Exception("Content element not found with any selector")

                        html = await page.content()
                        logger.info(f"Successfully fetched: {url}")
                        return html

                    finally:
                        await browser.close()

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2**attempt) * self.rate_limit_seconds
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

        raise Exception(f"Failed to fetch {url} after {self.max_retries} attempts: {last_error}")

    def parse(self, html: str, url: str) -> LegalDocument:
        """
        Parse TVPL HTML into LegalDocument structure.

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            Parsed LegalDocument with hierarchy
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("BeautifulSoup is required. Install with: pip install beautifulsoup4")

        soup = BeautifulSoup(html, "html.parser")
        errors: List[str] = []

        # Create document with basic info
        doc = LegalDocument(
            url=url,
            so_hieu=self._extract_text(soup, ".so-hieu, .sohieu, [class*='so-hieu']", ""),
            title=self._extract_title(soup),
            raw_html=html if self.save_raw_html else "",
        )

        # Extract metadata
        try:
            doc.loai_van_ban = self._extract_text(
                soup, ".loai-van-ban, .loaivanban, [class*='loai-van-ban']", ""
            )
            doc.co_quan_ban_hanh = self._extract_text(
                soup, ".co-quan-ban-hanh, .coquanbanhanh, [class*='co-quan']", ""
            )
            doc.nguoi_ky = self._extract_text(
                soup, ".nguoi-ky, .nguoiky, [class*='nguoi-ky']", ""
            )
            doc.tinh_trang = self._extract_text(
                soup, ".tinh-trang, .tinhtrang, [class*='tinh-trang']", ""
            )

            # Parse dates
            ngay_ban_hanh = self._extract_text(
                soup, ".ngay-ban-hanh, .ngaybanhanh, [class*='ngay-ban-hanh']", ""
            )
            if ngay_ban_hanh:
                doc.ngay_ban_hanh = self._parse_date(ngay_ban_hanh)

            ngay_hieu_luc = self._extract_text(
                soup, ".ngay-hieu-luc, .ngayhieuluc, [class*='ngay-hieu-luc']", ""
            )
            if ngay_hieu_luc:
                doc.ngay_hieu_luc = self._parse_date(ngay_hieu_luc)

        except Exception as e:
            errors.append(f"Metadata extraction error: {e}")
            logger.warning(f"Error extracting metadata: {e}")

        # Extract document number from URL if not found in page
        if not doc.so_hieu:
            doc.so_hieu = self._extract_so_hieu_from_url(url)

        # Extract and parse content (try multiple selectors)
        try:
            content_el = None
            for selector in self.CONTENT_SELECTORS:
                content_el = soup.select_one(selector)
                if content_el:
                    # Check if has meaningful content
                    paragraphs = content_el.find_all("p")
                    if paragraphs and len(paragraphs) > 10:
                        logger.debug(f"Using content selector: {selector}")
                        break
                    content_el = None

            if content_el:
                doc.raw_text = self._extract_text_from_paragraphs(content_el)

                # Extract hierarchy
                extractor = HierarchyExtractor()
                doc.chapters, doc.articles, doc.appendices = extractor.extract(doc.raw_text)

                # Log extraction stats
                stats = extractor.validate_structure(doc.raw_text)
                logger.info(
                    f"Extracted: {stats['chapters']} chapters, "
                    f"{stats['articles']} articles, "
                    f"{stats['clauses']} clauses, "
                    f"{stats['appendices']} appendices"
                )
            else:
                errors.append("Content element not found with any selector")
                doc.is_complete = False
        except Exception as e:
            errors.append(f"Content extraction error: {e}")
            doc.is_complete = False
            logger.error(f"Error extracting content: {e}")

        # Extract additional metadata from table if present
        doc.metadata = self._extract_metadata_table(soup)
        doc.scrape_errors = errors

        return doc

    def _extract_title(self, soup) -> str:
        """Extract document title from multiple possible locations."""
        # Try common title selectors
        selectors = [
            "h1.title",
            ".doc-title h1",
            ".title-vb",
            "h1",
            ".ten-van-ban",
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True)
                if text:
                    return text
        return ""

    def _extract_text(self, soup, selector: str, default: str) -> str:
        """
        Extract text from CSS selector, trying multiple selectors.

        Args:
            soup: BeautifulSoup object
            selector: CSS selector (comma-separated for multiple)
            default: Default value if not found
        """
        try:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(strip=True)
                # Clean up common prefixes
                prefixes = ["Số hiệu:", "Ngày ban hành:", "Cơ quan ban hành:", "Loại văn bản:"]
                for prefix in prefixes:
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                return text
        except Exception:
            pass
        return default

    def _extract_metadata_table(self, soup) -> Dict[str, Any]:
        """Extract metadata from table format commonly used on TVPL."""
        metadata = {}

        # Find metadata tables
        tables = soup.select(".thuoc-tinh table, .doc-info table, table.thuoc-tinh")
        for table in tables:
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td, th")
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).rstrip(":")
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        metadata[key] = value

        return metadata

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse Vietnamese date formats.

        Handles:
        - dd/mm/yyyy
        - dd-mm-yyyy
        - "ngày X tháng Y năm Z"
        """
        if not date_str:
            return None

        # Pattern 1: dd/mm/yyyy or dd-mm-yyyy
        patterns = [
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", lambda m: (int(m.group(3)), int(m.group(2)), int(m.group(1)))),
            # Pattern 2: "ngày X tháng Y năm Z"
            (
                r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
                lambda m: (int(m.group(3)), int(m.group(2)), int(m.group(1))),
            ),
        ]

        for pattern, extractor in patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    year, month, day = extractor(match)
                    return datetime(year, month, day)
                except ValueError:
                    continue

        return None

    # Web UI elements to strip from content (TVPL-specific)
    WEB_UI_ELEMENTS = ["huongdan", "tooltip"]

    def _extract_text_from_paragraphs(self, content_el) -> str:
        """
        Extract clean text from HTML content using paragraph-based approach.

        Strategy:
        - Extract text from each <p> tag
        - Strip web UI elements (huongdan, tooltip) before extraction
        - Normalize whitespace within each paragraph
        - Join paragraphs with double newlines (\\n\\n) as delimiters

        Args:
            content_el: BeautifulSoup Tag containing document content

        Returns:
            Normalized text with \\n\\n between paragraphs
        """
        paragraphs = content_el.find_all("p")
        cleaned_paragraphs = []

        for p in paragraphs:
            # Strip web UI elements before extracting text
            for ui_element in p.find_all(self.WEB_UI_ELEMENTS):
                ui_element.decompose()

            # Get text and normalize internal whitespace
            text = p.get_text()
            # Replace multiple whitespace (including newlines) with single space
            text = " ".join(text.split())

            if text:
                cleaned_paragraphs.append(text)

        return "\n\n".join(cleaned_paragraphs)

    def _extract_so_hieu_from_url(self, url: str) -> str:
        """Extract document number from TVPL URL pattern."""
        # Pattern: /van-ban/.../Luat-xxx-so-59-2020-QH14-...
        patterns = [
            r"so-(\d+[-/]\d+[-/][A-Z]+\d*)",
            r"(\d+/\d+/[A-Z]+\d*)",
            r"(\d+-\d+-[A-Z]+\d*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1).replace("-", "/")
        return ""

    async def scrape_batch(
        self,
        urls: List[str],
        continue_on_error: bool = True,
    ) -> List[LegalDocument]:
        """
        Scrape multiple URLs with rate limiting.

        Args:
            urls: List of URLs to scrape
            continue_on_error: Continue if individual scrapes fail

        Returns:
            List of LegalDocument objects (may include incomplete docs)
        """
        documents: List[LegalDocument] = []

        for i, url in enumerate(urls):
            logger.info(f"Scraping {i + 1}/{len(urls)}: {url}")

            try:
                doc = await self.scrape(url)
                documents.append(doc)

                # Save if output dir configured
                if self.output_dir:
                    self._save_document(doc)

            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                if continue_on_error:
                    # Create partial document marking failure
                    doc = LegalDocument(
                        url=url,
                        so_hieu="",
                        title="",
                        is_complete=False,
                        scrape_errors=[str(e)],
                    )
                    documents.append(doc)
                else:
                    raise

        return documents

    def _save_document(self, doc: LegalDocument) -> None:
        """Save scraped document to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from so_hieu or URL
        filename = doc.so_hieu.replace("/", "-") if doc.so_hieu else "unknown"
        filename = re.sub(r"[^\w\-]", "_", filename)

        # Save JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {json_path}")

        # Save raw HTML if enabled
        if self.save_raw_html and doc.raw_html:
            html_path = self.output_dir / f"{filename}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(doc.raw_html)
            logger.info(f"Saved HTML: {html_path}")
