"""
Hierarchy extractor for Vietnamese legal documents.

Extracts hierarchical structure using regex patterns:
- Chương (Chapter): "Chương I", "Chương 1"
- Mục (Section): "Mục 1", "MỤC 1"
- Điều (Article): "Điều 1", "ĐIỀU 1"
- Khoản (Clause): "1.", "2."
- Điểm (Point): "a)", "b)", "đ)"
- Phụ lục (Appendix): "Phụ lục I", "PHỤ LỤC"
"""

import re
from importlib import import_module
from typing import List, Optional, Tuple

# Import from kebab-case filename
_base_mod = import_module(".base-legal-scraper", "vn_legal_rag.offline.scraper")
LegalAppendix = _base_mod.LegalAppendix
LegalAppendixItem = _base_mod.LegalAppendixItem
LegalArticle = _base_mod.LegalArticle
LegalChapter = _base_mod.LegalChapter
LegalClause = _base_mod.LegalClause
LegalPoint = _base_mod.LegalPoint
LegalSection = _base_mod.LegalSection


class HierarchyExtractor:
    """Extract hierarchical structure from Vietnamese legal text."""

    # Regex patterns for Vietnamese legal structure
    PATTERNS = {
        # Chapter: "Chương I", "Chương 1", "CHƯƠNG I"
        "chapter": re.compile(
            r"(?:^|\n)\s*(?:CHƯƠNG|Chương)\s+([IVXLC]+|\d+)[\.:\s]*\n*([^\n]*)",
            re.MULTILINE | re.IGNORECASE,
        ),
        # Section: "Mục 1", "MỤC 1"
        "section": re.compile(
            r"(?:^|\n)\s*(?:MỤC|Mục)\s+(\d+)[\.:\s]*\n*([^\n]*)",
            re.MULTILINE | re.IGNORECASE,
        ),
        # Article: "Điều 1", "ĐIỀU 1", "Điều 1."
        "article": re.compile(
            r"(?:^|\n)\s*(?:ĐIỀU|Điều)\s+(\d+)[\.:\s]*([^\n]*)",
            re.MULTILINE | re.IGNORECASE,
        ),
        # Clause: "1. ", "2. " or just "1." at line start (content may be on next line)
        "clause": re.compile(r"^\s*(\d+)\.\s*(.*)", re.MULTILINE),
        # Point: "a) ", "b) ", "đ) " or just "a)" at line start
        "point": re.compile(r"^\s*([a-zđ])\)\s*(.*)", re.MULTILINE),
        # Appendix: "Phụ lục I", "PHỤ LỤC", "Phụ lục:" with optional number/title
        "appendix": re.compile(
            r"(?:^|\n)\s*(?:PHỤ\s*LỤC|Phụ\s*lục)\s*([IVXLC]+|\d+)?[\.:\s]*\n*([^\n]*)",
            re.MULTILINE | re.IGNORECASE,
        ),
    }

    def extract(
        self, text: str
    ) -> Tuple[List[LegalChapter], List[LegalArticle], List[LegalAppendix]]:
        """
        Extract chapters, articles, and appendices from legal text.

        Args:
            text: Full text content of the legal document

        Returns:
            Tuple of (chapters, standalone_articles, appendices)
            - chapters: List of LegalChapter with nested structure
            - standalone_articles: Articles not in any chapter
            - appendices: List of LegalAppendix extracted from document
        """
        # First, split text at appendix markers
        main_text, appendices = self._split_appendices(text)

        chapters: List[LegalChapter] = []
        standalone_articles: List[LegalArticle] = []

        # Find all chapters in main text only
        chapter_matches = list(self.PATTERNS["chapter"].finditer(main_text))

        if chapter_matches:
            for i, match in enumerate(chapter_matches):
                # Determine chapter content boundaries
                start = match.end()
                end = (
                    chapter_matches[i + 1].start()
                    if i + 1 < len(chapter_matches)
                    else len(main_text)
                )
                chapter_text = main_text[start:end]

                # Extract chapter title (may span multiple lines until first article)
                title = self._clean_title(match.group(2))

                # Create chapter with nested content
                chapter = LegalChapter(
                    number=match.group(1).strip(),
                    title=title,
                    raw_text=match.group(0) + chapter_text,
                )

                # Extract sections and articles within chapter
                chapter.sections, chapter.articles = self._extract_sections_and_articles(
                    chapter_text
                )
                chapters.append(chapter)
        else:
            # No chapters found, extract articles directly
            standalone_articles = self._extract_articles(main_text)

        return chapters, standalone_articles, appendices

    def _split_appendices(self, text: str) -> Tuple[str, List[LegalAppendix]]:
        """
        Split text at appendix markers and extract appendices.

        Returns:
            Tuple of (main_text, appendices)
        """
        appendix_matches = list(self.PATTERNS["appendix"].finditer(text))

        if not appendix_matches:
            return text, []

        # Main text is everything before first appendix
        main_text = text[: appendix_matches[0].start()]
        appendices: List[LegalAppendix] = []

        for i, match in enumerate(appendix_matches):
            start = match.end()
            end = (
                appendix_matches[i + 1].start()
                if i + 1 < len(appendix_matches)
                else len(text)
            )
            appendix_text = text[start:end]

            # Extract appendix number
            number = (match.group(1) or "").strip()

            # Extract title from match.group(2) first (inline title after "PHỤ LỤC X")
            inline_title = self._clean_title(match.group(2) or "")

            # If inline title is meaningful, use it; otherwise extract from appendix text
            if inline_title and len(inline_title) > 10 and not inline_title.startswith("("):
                title = inline_title
            else:
                # Fall back to extracting from appendix text body
                title = self._extract_appendix_title(appendix_text)

            # Extract items from appendix (numbered items like "1. ...")
            items = self._extract_appendix_items(appendix_text)

            appendix = LegalAppendix(
                number=number,
                title=title,
                items=items,
                raw_text=match.group(0) + appendix_text,
            )
            appendices.append(appendix)

        return main_text, appendices

    def _extract_appendix_title(self, appendix_text: str) -> str:
        """
        Extract meaningful title from appendix text.

        Skip metadata lines like "(Ban hành kèm theo...)" and find
        the actual document title (usually ALL CAPS document type).
        """
        lines = appendix_text.split("\n")
        title_parts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip metadata lines starting with parentheses
            if line.startswith("(") and "kèm theo" in line.lower():
                continue

            # Skip common header patterns (org names, dates, etc.)
            if any(skip in line for skip in ["CỘNG HÒA XÃ HỘI", "Độc lập", "-------", "---------------"]):
                continue
            if re.match(r"^(Số|ngày|tháng|năm)[:\s/]", line, re.IGNORECASE):
                continue
            if re.match(r"^TÊN (CƠ QUAN|DOANH NGHIỆP)", line):
                continue

            # Skip placeholder lines (mostly ellipsis, dots, or date templates)
            ellipsis_count = line.count("…") + line.count(".")
            if ellipsis_count > 3 and ellipsis_count > len(line) / 10:
                continue

            # Found a meaningful line - likely the title
            if line and len(line) > 3:
                title_parts.append(line)
                # Get next non-empty line if title seems incomplete
                if len(line) < 50 and not line.endswith((".", ":", ")")):
                    continue
                break

        # Join title parts and clean
        title = " ".join(title_parts)
        return self._clean_title(title)[:200]  # Limit length

    def _extract_appendix_items(self, text: str) -> List[LegalAppendixItem]:
        """Extract numbered items from appendix text."""
        items: List[LegalAppendixItem] = []
        lines = text.split("\n")
        current_item: Optional[LegalAppendixItem] = None
        current_content: List[str] = []

        for line in lines:
            # Match numbered items: "1. content" or "1) content"
            item_match = re.match(r"^\s*(\d+)[.)\s]+(.*)$", line)

            if item_match:
                # Save previous item
                if current_item is not None:
                    content = "\n".join(current_content).strip()
                    current_item.content = content
                    current_item.raw_text = f"{current_item.number}. {content}"
                    items.append(current_item)

                # Start new item
                current_item = LegalAppendixItem(
                    number=int(item_match.group(1)),
                    content="",
                )
                first_content = item_match.group(2).strip()
                current_content = [first_content] if first_content else []
            elif current_item is not None:
                stripped = line.strip()
                if stripped:
                    current_content.append(stripped)

        # Save last item
        if current_item is not None:
            content = "\n".join(current_content).strip()
            current_item.content = content
            current_item.raw_text = f"{current_item.number}. {content}"
            items.append(current_item)

        return items

    def _extract_sections_and_articles(
        self, text: str
    ) -> Tuple[List[LegalSection], List[LegalArticle]]:
        """
        Extract sections and direct articles from chapter text.

        Returns:
            Tuple of (sections, direct_articles)
        """
        sections: List[LegalSection] = []
        direct_articles: List[LegalArticle] = []

        section_matches = list(self.PATTERNS["section"].finditer(text))

        if section_matches:
            # Track content before first section for direct articles
            pre_section_text = text[: section_matches[0].start()]
            if pre_section_text.strip():
                direct_articles = self._extract_articles(pre_section_text)

            for i, match in enumerate(section_matches):
                start = match.end()
                end = (
                    section_matches[i + 1].start()
                    if i + 1 < len(section_matches)
                    else len(text)
                )
                section_text = text[start:end]

                section = LegalSection(
                    number=int(match.group(1)),
                    title=self._clean_title(match.group(2)),
                    articles=self._extract_articles(section_text),
                    raw_text=match.group(0) + section_text,
                )
                sections.append(section)
        else:
            # No sections, all articles are direct
            direct_articles = self._extract_articles(text)

        return sections, direct_articles

    def _extract_articles(self, text: str) -> List[LegalArticle]:
        """Extract articles (Điều) from text segment."""
        articles: List[LegalArticle] = []
        article_matches = list(self.PATTERNS["article"].finditer(text))

        for i, match in enumerate(article_matches):
            start = match.end()
            end = (
                article_matches[i + 1].start()
                if i + 1 < len(article_matches)
                else len(text)
            )
            article_text = text[start:end].strip()

            article = LegalArticle(
                number=int(match.group(1)),
                title=self._clean_title(match.group(2)),
                content=article_text,
                clauses=self._extract_clauses(article_text),
                raw_text=match.group(0) + article_text,
            )
            articles.append(article)

        return articles

    def _extract_clauses(self, text: str) -> List[LegalClause]:
        """
        Extract clauses (Khoản) from article text.

        Handles multi-line clause content and nested points.
        """
        clauses: List[LegalClause] = []
        lines = text.split("\n")
        current_clause: Optional[LegalClause] = None
        current_content: List[str] = []

        for line in lines:
            clause_match = self.PATTERNS["clause"].match(line)

            if clause_match:
                # Save previous clause
                if current_clause is not None:
                    content = "\n".join(current_content).strip()
                    current_clause.content = content
                    current_clause.raw_text = f"{current_clause.number}. {content}"
                    current_clause.points = self._extract_points(content)
                    clauses.append(current_clause)

                # Start new clause
                current_clause = LegalClause(
                    number=int(clause_match.group(1)),
                    content="",
                )
                # Content might be empty if on next line
                first_content = clause_match.group(2).strip()
                current_content = [first_content] if first_content else []
            elif current_clause is not None:
                # Add line to current clause content
                stripped = line.strip()
                if stripped:
                    current_content.append(stripped)

        # Save last clause
        if current_clause is not None:
            content = "\n".join(current_content).strip()
            current_clause.content = content
            current_clause.raw_text = f"{current_clause.number}. {content}"
            current_clause.points = self._extract_points(content)
            clauses.append(current_clause)

        return clauses

    def _extract_points(self, text: str) -> List[LegalPoint]:
        """
        Extract points (Điểm) from clause text.

        Handles Vietnamese letters including đ.
        """
        points: List[LegalPoint] = []
        lines = text.split("\n")
        current_point: Optional[LegalPoint] = None
        current_content: List[str] = []

        for line in lines:
            point_match = self.PATTERNS["point"].match(line)

            if point_match:
                if current_point is not None:
                    content = "\n".join(current_content).strip()
                    current_point.content = content
                    current_point.raw_text = f"{current_point.letter}) {content}"
                    points.append(current_point)

                current_point = LegalPoint(
                    letter=point_match.group(1),
                    content="",
                )
                # Content might be empty if on next line
                first_content = point_match.group(2).strip()
                current_content = [first_content] if first_content else []
            elif current_point is not None:
                stripped = line.strip()
                if stripped:
                    current_content.append(stripped)

        if current_point is not None:
            content = "\n".join(current_content).strip()
            current_point.content = content
            current_point.raw_text = f"{current_point.letter}) {content}"
            points.append(current_point)

        return points

    def _clean_title(self, title: str) -> str:
        """Clean and normalize title text."""
        if not title:
            return ""
        # Remove extra whitespace
        title = " ".join(title.split())
        # Remove leading punctuation
        title = title.lstrip(".:- ")
        return title.strip()

    def validate_structure(self, text: str) -> dict:
        """
        Validate extracted structure and return statistics.

        Returns:
            Dictionary with counts and potential issues
        """
        chapters, articles, appendices = self.extract(text)

        total_articles = len(articles)
        total_clauses = 0
        total_points = 0

        for chapter in chapters:
            total_articles += len(chapter.articles)
            for section in chapter.sections:
                total_articles += len(section.articles)
                for article in section.articles:
                    total_clauses += len(article.clauses)
                    for clause in article.clauses:
                        total_points += len(clause.points)

            for article in chapter.articles:
                total_clauses += len(article.clauses)
                for clause in article.clauses:
                    total_points += len(clause.points)

        for article in articles:
            total_clauses += len(article.clauses)
            for clause in article.clauses:
                total_points += len(clause.points)

        total_appendix_items = sum(len(a.items) for a in appendices)

        return {
            "chapters": len(chapters),
            "sections": sum(len(c.sections) for c in chapters),
            "articles": total_articles,
            "clauses": total_clauses,
            "points": total_points,
            "has_chapters": len(chapters) > 0,
            "standalone_articles": len(articles),
            "appendices": len(appendices),
            "appendix_items": total_appendix_items,
        }
