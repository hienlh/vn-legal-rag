"""
SQLAlchemy models for Vietnamese legal document storage.

Schema: Document > Chương > Mục > Điều > Khoản > Điểm

ID Format:
- Document: "59-2020-QH14"
- Chapter:  "59-2020-QH14:c1"
- Section:  "59-2020-QH14:c1:m2"
- Article:  "59-2020-QH14:d5"
- Clause:   "59-2020-QH14:d5:k1"
- Point:    "59-2020-QH14:d5:k1:a"
"""

import re
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# =============================================================================
# ID Generation Helpers
# =============================================================================


def normalize_so_hieu(so_hieu: str) -> str:
    """
    Normalize số hiệu for use as document ID.

    Examples:
        "59/2020/QH14" → "59-2020-QH14"
        "89/2024/NĐ-CP" → "89-2024-ND-CP"
    """
    normalized = so_hieu.replace("/", "-")
    normalized = normalized.replace("Đ", "D").replace("đ", "d")
    normalized = re.sub(r"[^a-zA-Z0-9\-]", "", normalized)
    return normalized


def make_document_id(so_hieu: str) -> str:
    """Generate document ID from số hiệu."""
    return normalize_so_hieu(so_hieu)


def make_chapter_id(doc_id: str, chapter_number: str | int) -> str:
    """Generate chapter ID."""
    num = _roman_to_int(str(chapter_number)) if isinstance(chapter_number, str) else chapter_number
    return f"{doc_id}:c{num}"


def make_section_id(doc_id: str, chapter_number: str | int, section_number: int) -> str:
    """Generate section (mục) ID."""
    chap_num = _roman_to_int(str(chapter_number)) if isinstance(chapter_number, str) else chapter_number
    return f"{doc_id}:c{chap_num}:m{section_number}"


def make_article_id(doc_id: str, article_number: int) -> str:
    """Generate article (điều) ID."""
    return f"{doc_id}:d{article_number}"


def make_clause_id(article_id: str, clause_number: int) -> str:
    """Generate clause (khoản) ID."""
    return f"{article_id}:k{clause_number}"


def make_point_id(clause_id: str, point_letter: str) -> str:
    """Generate point (điểm) ID."""
    return f"{clause_id}:{point_letter}"


def make_crossref_id(source_article_id: str, target_article_id: Optional[str], index: int) -> str:
    """Generate cross-reference ID."""
    if target_article_id:
        return f"{source_article_id}→{target_article_id}#{index}"
    return f"{source_article_id}→ext#{index}"


def _roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    roman = roman.strip().upper()
    if roman.isdigit():
        return int(roman)

    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result = 0
    prev = 0
    for char in reversed(roman):
        curr = values.get(char, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result if result > 0 else 1


# =============================================================================
# SQLAlchemy Models
# =============================================================================


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class LegalDocumentModel(Base):
    """Legal document (Văn bản pháp luật)."""

    __tablename__ = "legal_documents"

    id = Column(String(100), primary_key=True)
    so_hieu = Column(String(100), nullable=False, unique=True, index=True)
    title = Column(String(500), nullable=False)
    loai_van_ban = Column(String(50))
    co_quan_ban_hanh = Column(String(200))
    nguoi_ky = Column(String(200))
    ngay_ban_hanh = Column(Date)
    ngay_hieu_luc = Column(Date)
    tinh_trang = Column(String(100))
    raw_text = Column(Text)
    kg_node_id = Column(String(100), index=True)
    source_url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    chapters = relationship(
        "LegalChapterModel",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="LegalChapterModel.position",
    )


class LegalChapterModel(Base):
    """Chapter (Chương) within a legal document."""

    __tablename__ = "legal_chapters"

    id = Column(String(120), primary_key=True)
    document_id = Column(
        String(100), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False
    )
    chapter_number = Column(String(20), nullable=False)
    title = Column(String(500))
    raw_text = Column(Text)
    kg_node_id = Column(String(100), index=True)
    position = Column(Integer, default=0)

    document = relationship("LegalDocumentModel", back_populates="chapters")
    sections = relationship(
        "LegalSectionModel",
        back_populates="chapter",
        cascade="all, delete-orphan",
        order_by="LegalSectionModel.position",
    )
    articles = relationship(
        "LegalArticleModel",
        back_populates="chapter",
        cascade="all, delete-orphan",
        order_by="LegalArticleModel.position",
    )


class LegalSectionModel(Base):
    """Section (Mục) within a chapter."""

    __tablename__ = "legal_sections"

    id = Column(String(130), primary_key=True)
    chapter_id = Column(
        String(120), ForeignKey("legal_chapters.id", ondelete="CASCADE"), nullable=False
    )
    section_number = Column(Integer, nullable=False)
    title = Column(String(500))
    raw_text = Column(Text)
    kg_node_id = Column(String(100), index=True)
    position = Column(Integer, default=0)

    chapter = relationship("LegalChapterModel", back_populates="sections")
    articles = relationship(
        "LegalArticleModel",
        back_populates="section",
        cascade="all, delete-orphan",
        order_by="LegalArticleModel.position",
    )


class LegalArticleModel(Base):
    """Article (Điều) - primary legal unit."""

    __tablename__ = "legal_articles"

    id = Column(String(120), primary_key=True)
    document_id = Column(
        String(100), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False
    )
    chapter_id = Column(
        String(120), ForeignKey("legal_chapters.id", ondelete="CASCADE"), nullable=True
    )
    section_id = Column(
        String(130), ForeignKey("legal_sections.id", ondelete="CASCADE"), nullable=True
    )
    article_number = Column(Integer, nullable=False)
    title = Column(String(500))
    content = Column(Text)
    raw_text = Column(Text)
    kg_node_id = Column(String(100), index=True)
    position = Column(Integer, default=0)

    document = relationship("LegalDocumentModel")
    chapter = relationship("LegalChapterModel", back_populates="articles")
    section = relationship("LegalSectionModel", back_populates="articles")
    clauses = relationship(
        "LegalClauseModel",
        back_populates="article",
        cascade="all, delete-orphan",
        order_by="LegalClauseModel.position",
    )

    __table_args__ = (
        Index("idx_article_doc_num", "document_id", "article_number"),
        Index("idx_article_kg", "kg_node_id"),
    )


class LegalClauseModel(Base):
    """Clause (Khoản) within an article."""

    __tablename__ = "legal_clauses"

    id = Column(String(130), primary_key=True)
    article_id = Column(
        String(120), ForeignKey("legal_articles.id", ondelete="CASCADE"), nullable=False
    )
    clause_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    raw_text = Column(Text)
    kg_node_id = Column(String(100), index=True)
    position = Column(Integer, default=0)

    article = relationship("LegalArticleModel", back_populates="clauses")
    points = relationship(
        "LegalPointModel",
        back_populates="clause",
        cascade="all, delete-orphan",
        order_by="LegalPointModel.position",
    )


class LegalPointModel(Base):
    """Point (Điểm) within a clause."""

    __tablename__ = "legal_points"

    id = Column(String(140), primary_key=True)
    clause_id = Column(
        String(130), ForeignKey("legal_clauses.id", ondelete="CASCADE"), nullable=False
    )
    point_letter = Column(String(5), nullable=False)
    content = Column(Text, nullable=False)
    raw_text = Column(Text)
    kg_node_id = Column(String(100), index=True)
    position = Column(Integer, default=0)

    clause = relationship("LegalClauseModel", back_populates="points")


class LegalCrossReferenceModel(Base):
    """Cross-reference between articles."""

    __tablename__ = "legal_cross_references"

    id = Column(String(300), primary_key=True)
    source_article_id = Column(
        String(120), ForeignKey("legal_articles.id", ondelete="CASCADE")
    )
    target_article_id = Column(
        String(120), ForeignKey("legal_articles.id", ondelete="SET NULL"), nullable=True
    )
    target_document_so_hieu = Column(String(100), nullable=True)
    reference_text = Column(Text)
    context_sentence = Column(Text, nullable=True)
    reference_type = Column(String(50), default="THAM_CHIẾU")
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    source_article = relationship(
        "LegalArticleModel", foreign_keys=[source_article_id]
    )
    target_article = relationship(
        "LegalArticleModel", foreign_keys=[target_article_id]
    )


class LegalAbbreviationModel(Base):
    """Vietnamese legal abbreviation dictionary."""

    __tablename__ = "legal_abbreviations"

    id = Column(String(50), primary_key=True)
    abbreviation = Column(String(50), nullable=False, unique=True, index=True)
    full_form = Column(String(300), nullable=True)
    category = Column(String(50), nullable=True)
    corpus_count = Column(Integer, default=0)
    confidence = Column(Integer, default=100)
    detection_reason = Column(String(50))
    sample_context = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_abbrev_category", "category"),
        Index("idx_abbrev_count", "corpus_count"),
    )
