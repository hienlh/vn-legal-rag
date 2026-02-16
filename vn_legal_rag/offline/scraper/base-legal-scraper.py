"""
Base classes for legal document scraping.

Defines data classes for Vietnamese legal document hierarchy:
- LegalDocument (Văn bản)
- LegalChapter (Chương)
- LegalSection (Mục)
- LegalArticle (Điều)
- LegalClause (Khoản)
- LegalPoint (Điểm)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentStatus(Enum):
    """Legal document validity status."""

    CON_HIEU_LUC = "Còn hiệu lực"
    HET_HIEU_LUC = "Hết hiệu lực"
    CHUA_CO_HIEU_LUC = "Chưa có hiệu lực"
    NGUNG_HIEU_LUC = "Ngưng hiệu lực"
    UNKNOWN = "Không xác định"


class DocumentType(Enum):
    """Vietnamese legal document types."""

    LUAT = "Luật"
    NGHI_DINH = "Nghị định"
    NGHI_QUYET = "Nghị quyết"
    THONG_TU = "Thông tư"
    QUYET_DINH = "Quyết định"
    CONG_VAN = "Công văn"
    CHI_THI = "Chỉ thị"
    THONG_TU_LIEN_TICH = "Thông tư liên tịch"
    PHAP_LENH = "Pháp lệnh"
    UNKNOWN = "Khác"


@dataclass
class LegalAppendixItem:
    """
    Represents an item within an appendix (Mục trong Phụ lục).

    Example: "1. Sản xuất cung ứng thuốc nổ..."
    """

    number: int
    content: str
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "content": self.content,
            "raw_text": self.raw_text,
        }


@dataclass
class LegalAppendix:
    """
    Represents an appendix (Phụ lục) in Vietnamese legal text.

    Example: "PHỤ LỤC I: DANH MỤC NGÀNH..."
    """

    number: str  # "I", "II", "1", "2", or empty for single appendix
    title: str
    items: List["LegalAppendixItem"] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "title": self.title,
            "items": [i.to_dict() for i in self.items],
            "raw_text": self.raw_text,
        }


@dataclass
class LegalPoint:
    """
    Represents a point (Điểm) in Vietnamese legal text.

    Example: "a) Có dự án đầu tư kinh doanh..."
    """

    letter: str  # a, b, c, d, đ, e...
    content: str
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "letter": self.letter,
            "content": self.content,
            "raw_text": self.raw_text,
        }


@dataclass
class LegalClause:
    """
    Represents a clause (Khoản) in Vietnamese legal text.

    Example: "1. Doanh nghiệp được thành lập..."
    """

    number: int
    content: str
    points: List[LegalPoint] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "content": self.content,
            "points": [p.to_dict() for p in self.points],
            "raw_text": self.raw_text,
        }


@dataclass
class LegalArticle:
    """
    Represents an article (Điều) in Vietnamese legal text.

    Example: "Điều 1. Phạm vi điều chỉnh"
    """

    number: int
    title: str
    content: str
    clauses: List[LegalClause] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "clauses": [c.to_dict() for c in self.clauses],
            "raw_text": self.raw_text,
        }


@dataclass
class LegalSection:
    """
    Represents a section (Mục) within a chapter.

    Example: "Mục 1. THÀNH LẬP DOANH NGHIỆP"
    """

    number: int
    title: str
    articles: List[LegalArticle] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "title": self.title,
            "articles": [a.to_dict() for a in self.articles],
            "raw_text": self.raw_text,
        }


@dataclass
class LegalChapter:
    """
    Represents a chapter (Chương) in Vietnamese legal text.

    Example: "Chương I. QUY ĐỊNH CHUNG"
    """

    number: str  # Roman numeral (I, II, III...) or Arabic
    title: str
    sections: List[LegalSection] = field(default_factory=list)
    articles: List[LegalArticle] = field(default_factory=list)  # Direct articles
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "articles": [a.to_dict() for a in self.articles],
            "raw_text": self.raw_text,
        }


@dataclass
class LegalDocument:
    """
    Complete legal document with hierarchical structure.

    Attributes:
        url: Source URL
        so_hieu: Document number (e.g., "59/2020/QH14")
        title: Full document title
        loai_van_ban: Document type (Luật, Nghị định, etc.)
        co_quan_ban_hanh: Issuing authority
        nguoi_ky: Signatory
        ngay_ban_hanh: Issue date
        ngay_hieu_luc: Effective date
        tinh_trang: Document status
        chapters: List of chapters (Chương)
        articles: Standalone articles (for docs without chapters)
        raw_html: Original HTML content
        raw_text: Extracted text content
        metadata: Additional metadata
    """

    url: str
    so_hieu: str
    title: str
    loai_van_ban: str = ""
    co_quan_ban_hanh: str = ""
    nguoi_ky: str = ""
    ngay_ban_hanh: Optional[datetime] = None
    ngay_hieu_luc: Optional[datetime] = None
    tinh_trang: str = ""
    chapters: List[LegalChapter] = field(default_factory=list)
    articles: List[LegalArticle] = field(default_factory=list)
    appendices: List[LegalAppendix] = field(default_factory=list)
    raw_html: str = ""
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    scrape_errors: List[str] = field(default_factory=list)
    is_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "so_hieu": self.so_hieu,
            "title": self.title,
            "loai_van_ban": self.loai_van_ban,
            "co_quan_ban_hanh": self.co_quan_ban_hanh,
            "nguoi_ky": self.nguoi_ky,
            "ngay_ban_hanh": self.ngay_ban_hanh.isoformat() if self.ngay_ban_hanh else None,
            "ngay_hieu_luc": self.ngay_hieu_luc.isoformat() if self.ngay_hieu_luc else None,
            "tinh_trang": self.tinh_trang,
            "chapters": [c.to_dict() for c in self.chapters],
            "articles": [a.to_dict() for a in self.articles],
            "appendices": [a.to_dict() for a in self.appendices],
            "metadata": self.metadata,
            "scrape_errors": self.scrape_errors,
            "is_complete": self.is_complete,
        }

    @property
    def total_articles(self) -> int:
        """Count total articles including those in chapters."""
        count = len(self.articles)
        for chapter in self.chapters:
            count += len(chapter.articles)
            for section in chapter.sections:
                count += len(section.articles)
        return count

    @property
    def document_type_enum(self) -> DocumentType:
        """Get document type as enum."""
        type_map = {
            "luật": DocumentType.LUAT,
            "nghị định": DocumentType.NGHI_DINH,
            "nghị quyết": DocumentType.NGHI_QUYET,
            "thông tư": DocumentType.THONG_TU,
            "quyết định": DocumentType.QUYET_DINH,
            "công văn": DocumentType.CONG_VAN,
            "chỉ thị": DocumentType.CHI_THI,
            "thông tư liên tịch": DocumentType.THONG_TU_LIEN_TICH,
            "pháp lệnh": DocumentType.PHAP_LENH,
        }
        return type_map.get(self.loai_van_ban.lower(), DocumentType.UNKNOWN)


class BaseLegalScraper(ABC):
    """
    Abstract base class for legal document scrapers.

    Subclasses implement specific website scrapers (e.g., thuvienphapluat.vn).
    """

    @abstractmethod
    async def fetch(self, url: str) -> str:
        """
        Fetch HTML content from URL.

        Args:
            url: Target URL to fetch

        Returns:
            Raw HTML content as string
        """
        pass

    @abstractmethod
    def parse(self, html: str, url: str) -> LegalDocument:
        """
        Parse HTML into LegalDocument structure.

        Args:
            html: Raw HTML content
            url: Source URL for reference

        Returns:
            Parsed LegalDocument with hierarchy
        """
        pass

    async def scrape(self, url: str) -> LegalDocument:
        """
        Fetch and parse a legal document.

        Args:
            url: URL of the legal document

        Returns:
            Complete LegalDocument with all extracted data
        """
        html = await self.fetch(url)
        return self.parse(html, url)
