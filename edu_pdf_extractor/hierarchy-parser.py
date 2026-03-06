"""Parse extracted Vietnamese legal text into structured hierarchy.

Ported from vn_legal_rag/offline/scraper/legal-hierarchy-extractor.py
but standalone — no imports from main project.

Hierarchy: Chương > Mục > Điều > Khoản > Điểm
"""

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """Điểm (e.g., a), b), đ))"""
    letter: str
    content: str

    def to_dict(self) -> dict:
        return {"letter": self.letter, "content": self.content}


@dataclass
class Clause:
    """Khoản (e.g., 1., 2.)"""
    number: int
    content: str
    points: List[Point] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "content": self.content,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class Article:
    """Điều"""
    number: int
    title: str
    content: str
    clauses: List[Clause] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "clauses": [c.to_dict() for c in self.clauses],
        }


@dataclass
class Section:
    """Mục"""
    number: int
    title: str
    articles: List[Article] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "articles": [a.to_dict() for a in self.articles],
        }


@dataclass
class Chapter:
    """Chương"""
    number: str  # Roman numeral or Arabic
    title: str
    sections: List[Section] = field(default_factory=list)
    articles: List[Article] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "articles": [a.to_dict() for a in self.articles],
        }


@dataclass
class ParsedDocument:
    """Full parsed document."""
    doc_id: str
    so_hieu: str
    title: str
    co_quan: str
    ngay_ban_hanh: str
    chapters: List[Chapter] = field(default_factory=list)
    articles: List[Article] = field(default_factory=list)  # Standalone articles

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "so_hieu": self.so_hieu,
            "title": self.title,
            "co_quan": self.co_quan,
            "ngay_ban_hanh": self.ngay_ban_hanh,
            "chapters": [c.to_dict() for c in self.chapters],
            "articles": [a.to_dict() for a in self.articles],
        }

    def total_articles(self) -> int:
        count = len(self.articles)
        for ch in self.chapters:
            count += len(ch.articles)
            for sec in ch.sections:
                count += len(sec.articles)
        return count

    def total_clauses(self) -> int:
        count = 0
        for art in self._all_articles():
            count += len(art.clauses)
        return count

    def _all_articles(self) -> List[Article]:
        articles = list(self.articles)
        for ch in self.chapters:
            articles.extend(ch.articles)
            for sec in ch.sections:
                articles.extend(sec.articles)
        return articles

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved structured JSON: {path}")


# Regex patterns for Vietnamese legal structure
PATTERNS = {
    "chapter": re.compile(
        r"(?:^|\n)\s*(?:CHƯƠNG|Chương)\s+([IVXLC]+|\d+)[\.:\s]*\n*([^\n]*)",
        re.MULTILINE | re.IGNORECASE,
    ),
    "section": re.compile(
        r"(?:^|\n)\s*(?:MỤC|Mục)\s+(\d+)[\.:\s]*\n*([^\n]*)",
        re.MULTILINE | re.IGNORECASE,
    ),
    "article": re.compile(
        r"(?:^|\n)\s*(?:ĐIỀU|Điều)\s+(\d+)[\.:\s]*([^\n]*)",
        re.MULTILINE | re.IGNORECASE,
    ),
    "clause": re.compile(r"^\s*(\d+)\.\s*(.*)", re.MULTILINE),
    "point": re.compile(r"^\s*([a-zđ])\)\s*(.*)", re.MULTILINE),
}


def _clean_title(title: str) -> str:
    if not title:
        return ""
    title = " ".join(title.split())
    title = title.lstrip(".:- ")
    return title.strip()


def _extract_points(text: str) -> List[Point]:
    points = []
    lines = text.split("\n")
    current_point: Optional[Point] = None
    current_content: List[str] = []

    for line in lines:
        m = PATTERNS["point"].match(line)
        if m:
            if current_point is not None:
                current_point.content = "\n".join(current_content).strip()
                points.append(current_point)
            current_point = Point(letter=m.group(1), content="")
            first = m.group(2).strip()
            current_content = [first] if first else []
        elif current_point is not None:
            s = line.strip()
            if s:
                current_content.append(s)

    if current_point is not None:
        current_point.content = "\n".join(current_content).strip()
        points.append(current_point)

    return points


def _extract_clauses(text: str) -> List[Clause]:
    clauses = []
    lines = text.split("\n")
    current_clause: Optional[Clause] = None
    current_content: List[str] = []

    for line in lines:
        m = PATTERNS["clause"].match(line)
        if m:
            if current_clause is not None:
                content = "\n".join(current_content).strip()
                current_clause.content = content
                current_clause.points = _extract_points(content)
                clauses.append(current_clause)
            current_clause = Clause(number=int(m.group(1)), content="")
            first = m.group(2).strip()
            current_content = [first] if first else []
        elif current_clause is not None:
            s = line.strip()
            if s:
                current_content.append(s)

    if current_clause is not None:
        content = "\n".join(current_content).strip()
        current_clause.content = content
        current_clause.points = _extract_points(content)
        clauses.append(current_clause)

    return clauses


def _extract_articles(text: str) -> List[Article]:
    articles = []
    matches = list(PATTERNS["article"].finditer(text))

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        articles.append(Article(
            number=int(m.group(1)),
            title=_clean_title(m.group(2)),
            content=content,
            clauses=_extract_clauses(content),
        ))

    return articles


def _extract_sections_and_articles(text: str) -> Tuple[List[Section], List[Article]]:
    sections = []
    direct_articles = []
    section_matches = list(PATTERNS["section"].finditer(text))

    if section_matches:
        pre = text[: section_matches[0].start()]
        if pre.strip():
            direct_articles = _extract_articles(pre)

        for i, m in enumerate(section_matches):
            start = m.end()
            end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
            sec_text = text[start:end]
            sections.append(Section(
                number=int(m.group(1)),
                title=_clean_title(m.group(2)),
                articles=_extract_articles(sec_text),
            ))
    else:
        direct_articles = _extract_articles(text)

    return sections, direct_articles


def parse_text(text: str, doc_config: dict) -> ParsedDocument:
    """Parse extracted text into structured document hierarchy.

    Args:
        text: Full extracted text of the document
        doc_config: Document config dict from config.DOCUMENTS

    Returns:
        ParsedDocument with full hierarchy
    """
    chapters = []
    standalone_articles = []

    chapter_matches = list(PATTERNS["chapter"].finditer(text))

    if chapter_matches:
        for i, m in enumerate(chapter_matches):
            start = m.end()
            end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            ch_text = text[start:end]

            chapter = Chapter(
                number=m.group(1).strip(),
                title=_clean_title(m.group(2)),
            )
            chapter.sections, chapter.articles = _extract_sections_and_articles(ch_text)
            chapters.append(chapter)
    else:
        standalone_articles = _extract_articles(text)

    doc = ParsedDocument(
        doc_id=doc_config["id"],
        so_hieu=doc_config["so_hieu"],
        title=doc_config["title"],
        co_quan=doc_config["co_quan"],
        ngay_ban_hanh=doc_config["ngay_ban_hanh"],
        chapters=chapters,
        articles=standalone_articles,
    )

    logger.info(
        f"{doc.doc_id}: {len(doc.chapters)} chapters, "
        f"{doc.total_articles()} articles, {doc.total_clauses()} clauses"
    )
    return doc
