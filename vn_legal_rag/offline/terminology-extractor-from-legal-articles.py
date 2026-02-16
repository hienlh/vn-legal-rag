"""
Terminology Extractor for Vietnamese Legal Documents.

Auto-extract abbreviations from "Giải thích từ ngữ" articles (usually Điều 3/4).
Zero LLM cost - pure regex-based extraction.

Features:
- Extract term definitions from legal glossary articles
- Auto-populate DomainConfig.abbreviations
- Enable zero-config for new law types

Usage:
    >>> from vn_legal_rag.offline import TerminologyExtractor
    >>> extractor = TerminologyExtractor()
    >>> terms = extractor.extract_from_article(article_text)
    >>> extractor.update_domain_config("59-2020-QH14", terms)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..utils import get_logger


@dataclass
class ExtractedTerm:
    """Extracted terminology from legal text."""

    short_form: str  # Abbreviation or short name
    full_form: str  # Full definition/expansion
    source_article: Optional[str] = None  # e.g., "Điều 3"
    source_clause: Optional[int] = None  # Clause number if applicable
    confidence: float = 1.0


@dataclass
class TerminologyResult:
    """Result of terminology extraction."""

    terms: List[ExtractedTerm] = field(default_factory=list)
    source_document: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def to_abbreviations_dict(self) -> Dict[str, str]:
        """Convert to abbreviations dict for DomainConfig."""
        return {t.short_form: t.full_form for t in self.terms}


class TerminologyExtractor:
    """
    Extract terminology definitions from Vietnamese legal documents.

    Targets "Giải thích từ ngữ" articles which define abbreviations
    and technical terms used throughout the document.
    """

    # Pattern for glossary article titles
    GLOSSARY_TITLE_PATTERNS = [
        re.compile(r"Giải\s+thích\s+từ\s+ngữ", re.IGNORECASE),
        re.compile(r"Từ\s+ngữ\s+sử\s+dụng", re.IGNORECASE),
        re.compile(r"Định\s+nghĩa", re.IGNORECASE),
        re.compile(r"Thuật\s+ngữ", re.IGNORECASE),
    ]

    # Patterns for term definitions
    # Pattern 1: "Term là/means definition"
    DEFINITION_PATTERN_LA = re.compile(
        r"^[\d.)\s]*([A-ZÀ-Ỹa-zà-ỹ][A-ZÀ-Ỹa-zà-ỹ\s,]+?)\s+là\s+(.+?)(?:\.|;|$)",
        re.MULTILINE | re.UNICODE,
    )

    # Pattern 2: "Term (abbreviation) là definition"
    DEFINITION_PATTERN_ABBR = re.compile(
        r"([A-ZÀ-Ỹa-zà-ỹ][A-ZÀ-Ỹa-zà-ỹ\s]+?)\s*\((?:sau\s+đây\s+)?(?:gọi\s+(?:tắt\s+)?là\s+)?[\"']?([A-ZĐa-zđ\s]+)[\"']?\)\s+là\s+(.+?)(?:\.|;|$)",
        re.MULTILINE | re.UNICODE,
    )

    # Pattern 3: Abbreviated form pattern - "gọi tắt là X"
    ABBREVIATION_PATTERN = re.compile(
        r"(.+?)\s*\(?\s*sau\s+đây\s+gọi\s+(?:tắt\s+)?là\s+[\"']?([A-ZĐa-zđ\s]+)[\"']?\s*\)?",
        re.UNICODE,
    )

    # Pattern 4: Numbered definitions "1. Term: definition" or "1. Term là definition"
    NUMBERED_DEFINITION_PATTERN = re.compile(
        r"^(\d+)[.)\s]+([A-ZÀ-Ỹa-zà-ỹ][A-ZÀ-Ỹa-zà-ỹ\s,]+?)(?::|là)\s*(.+?)(?:\.|;|$)",
        re.MULTILINE | re.UNICODE,
    )

    def __init__(self, config_dir: str = "config/domains"):
        """
        Initialize terminology extractor.

        Args:
            config_dir: Directory containing domain config YAML files
        """
        self.config_dir = Path(config_dir)
        self.logger = get_logger("terminology_extractor")

    def is_glossary_article(self, title: str) -> bool:
        """Check if article title indicates a glossary/terminology article."""
        if not title:
            return False
        return any(p.search(title) for p in self.GLOSSARY_TITLE_PATTERNS)

    def extract_from_article(
        self,
        content: str,
        article_title: Optional[str] = None,
        article_number: Optional[int] = None,
    ) -> TerminologyResult:
        """
        Extract terminology from a single article.

        Args:
            content: Article text content
            article_title: Article title (e.g., "Giải thích từ ngữ")
            article_number: Article number (e.g., 3)

        Returns:
            TerminologyResult with extracted terms
        """
        result = TerminologyResult()
        source = f"Điều {article_number}" if article_number else None

        # Check if this looks like a glossary article
        if article_title and not self.is_glossary_article(article_title):
            # Still try to extract if content has definition patterns
            if "là" not in content and "gọi tắt" not in content:
                return result

        # Try all extraction patterns
        terms_found: Dict[str, ExtractedTerm] = {}

        # Pattern 1: Abbreviation with "sau đây gọi tắt là"
        for match in self.ABBREVIATION_PATTERN.finditer(content):
            full_form = match.group(1).strip()
            short_form = match.group(2).strip()
            if self._is_valid_term(short_form, full_form):
                terms_found[short_form] = ExtractedTerm(
                    short_form=short_form,
                    full_form=full_form,
                    source_article=source,
                    confidence=0.95,
                )

        # Pattern 2: "Term (abbreviation) là definition"
        for match in self.DEFINITION_PATTERN_ABBR.finditer(content):
            full_form = match.group(1).strip()
            short_form = match.group(2).strip()
            if self._is_valid_term(short_form, full_form):
                terms_found[short_form] = ExtractedTerm(
                    short_form=short_form,
                    full_form=full_form,
                    source_article=source,
                    confidence=0.9,
                )

        # Pattern 3: Numbered definitions
        for match in self.NUMBERED_DEFINITION_PATTERN.finditer(content):
            clause_num = int(match.group(1))
            term = match.group(2).strip()
            definition = match.group(3).strip()

            # Check for embedded abbreviation
            abbr_match = self.ABBREVIATION_PATTERN.search(definition)
            if abbr_match:
                short_form = abbr_match.group(2).strip()
                if self._is_valid_term(short_form, term):
                    terms_found[short_form] = ExtractedTerm(
                        short_form=short_form,
                        full_form=term,
                        source_article=source,
                        source_clause=clause_num,
                        confidence=0.85,
                    )

        result.terms = list(terms_found.values())
        return result

    def extract_from_document(
        self,
        articles: List[Dict[str, Any]],
        document_id: Optional[str] = None,
    ) -> TerminologyResult:
        """
        Extract terminology from all glossary articles in a document.

        Args:
            articles: List of article dicts with 'number', 'title', 'content'
            document_id: Document identifier

        Returns:
            TerminologyResult with all extracted terms
        """
        result = TerminologyResult(source_document=document_id)

        # Find glossary articles (usually Điều 3 or 4)
        glossary_articles = []
        for article in articles:
            title = article.get("title", "")
            number = article.get("number")

            # Check by title
            if self.is_glossary_article(title):
                glossary_articles.append(article)
            # Or by common position (Điều 3, 4)
            elif number in [3, 4] and "giải thích" in title.lower():
                glossary_articles.append(article)

        if not glossary_articles:
            self.logger.info(f"No glossary articles found in {document_id}")
            return result

        # Extract from each glossary article
        all_terms: Dict[str, ExtractedTerm] = {}
        for article in glossary_articles:
            content = article.get("content", article.get("raw_text", ""))
            article_result = self.extract_from_article(
                content=content,
                article_title=article.get("title"),
                article_number=article.get("number"),
            )

            for term in article_result.terms:
                # Keep highest confidence version
                if (
                    term.short_form not in all_terms
                    or term.confidence > all_terms[term.short_form].confidence
                ):
                    all_terms[term.short_form] = term

        result.terms = list(all_terms.values())
        self.logger.info(
            f"Extracted {len(result.terms)} terms from {document_id}"
        )
        return result

    def update_domain_config(
        self,
        document_id: str,
        result: TerminologyResult,
        merge: bool = True,
    ) -> Path:
        """
        Update domain config YAML with extracted terms.

        Args:
            document_id: Document ID (e.g., "59-2020-QH14")
            result: Extraction result
            merge: If True, merge with existing; if False, replace

        Returns:
            Path to updated config file
        """
        config_path = self.config_dir / f"{document_id}.yaml"

        # Load existing config if merging
        config: Dict[str, Any] = {}
        if merge and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Update abbreviations
        abbreviations = config.get("abbreviations", {})
        for term in result.terms:
            if term.short_form not in abbreviations:
                abbreviations[term.short_form] = term.full_form

        config["abbreviations"] = abbreviations

        # Save
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        self.logger.info(
            f"Updated {config_path} with {len(result.terms)} abbreviations"
        )
        return config_path

    def _is_valid_term(self, short_form: str, full_form: str) -> bool:
        """Validate extracted term pair."""
        if not short_form or not full_form:
            return False
        if len(short_form) > 50 or len(full_form) > 200:
            return False
        if short_form == full_form:
            return False
        # Short form should be shorter than full form
        if len(short_form) >= len(full_form):
            return False
        return True


def extract_terminology_from_db(
    db,
    document_id: str,
    update_config: bool = False,
    config_dir: str = "config/domains",
) -> TerminologyResult:
    """
    Convenience function to extract terminology from database document.

    Args:
        db: LegalDocumentDB instance
        document_id: Document ID
        update_config: Whether to update domain config
        config_dir: Domain config directory

    Returns:
        TerminologyResult
    """
    extractor = TerminologyExtractor(config_dir)

    # Get articles from database
    from .models import LegalArticleModel, LegalDocumentModel

    with db.SessionLocal() as session:
        doc = session.query(LegalDocumentModel).filter(
            LegalDocumentModel.id == document_id
        ).first()

        if not doc:
            return TerminologyResult(errors=[f"Document {document_id} not found"])

        articles = []
        for chapter in doc.chapters:
            for article in chapter.articles:
                articles.append({
                    "number": article.article_number,
                    "title": article.title,
                    "content": article.raw_text,
                })
            for section in chapter.sections:
                for article in section.articles:
                    articles.append({
                        "number": article.article_number,
                        "title": article.title,
                        "content": article.raw_text,
                    })

    result = extractor.extract_from_document(articles, document_id)

    if update_config and result.terms:
        extractor.update_domain_config(document_id, result)

    return result
