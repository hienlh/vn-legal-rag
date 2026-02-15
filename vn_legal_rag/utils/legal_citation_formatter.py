"""
Legal citation formatter for Vietnamese law.

Formats citations like: "Điều X, Khoản Y, Điểm Z - Luật ABC số 123/2020/QH14"
"""

from typing import Dict, Optional


def format_citation(
    article_number: int,
    clause_number: Optional[int] = None,
    point_letter: Optional[str] = None,
    document_title: Optional[str] = None,
    document_number: Optional[str] = None,
) -> str:
    """
    Format legal citation from components.

    Args:
        article_number: Article number (Điều)
        clause_number: Optional clause number (Khoản)
        point_letter: Optional point letter (Điểm)
        document_title: Optional document title
        document_number: Optional document number

    Returns:
        Formatted citation string

    Examples:
        >>> format_citation(5)
        'Điều 5'
        >>> format_citation(5, 2)
        'Khoản 2, Điều 5'
        >>> format_citation(5, 2, 'a')
        'Điểm a, Khoản 2, Điều 5'
        >>> format_citation(5, document_title="Luật Doanh nghiệp", document_number="59/2020/QH14")
        'Điều 5 - Luật Doanh nghiệp số 59/2020/QH14'
    """
    parts = []

    # Add point, clause, article in reverse order
    if point_letter:
        parts.append(f"Điểm {point_letter}")
    if clause_number:
        parts.append(f"Khoản {clause_number}")
    parts.append(f"Điều {article_number}")

    citation = ", ".join(parts)

    # Add document reference
    if document_title or document_number:
        doc_parts = []
        if document_title:
            doc_parts.append(document_title)
        if document_number:
            doc_parts.append(f"số {document_number}")
        citation += f" - {' '.join(doc_parts)}"

    return citation


def parse_citation(citation: str) -> Dict[str, Optional[str]]:
    """
    Parse citation string into components.

    Args:
        citation: Citation string

    Returns:
        Dictionary with components (article, clause, point, document)

    Examples:
        >>> parse_citation("Điều 5")
        {'article': '5', 'clause': None, 'point': None, 'document': None}
        >>> parse_citation("Khoản 2, Điều 5 - Luật Doanh nghiệp")
        {'article': '5', 'clause': '2', 'point': None, 'document': 'Luật Doanh nghiệp'}
    """
    import re

    result = {
        "article": None,
        "clause": None,
        "point": None,
        "document": None,
    }

    # Split by document separator
    parts = citation.split(" - ")
    main_part = parts[0].strip()
    doc_part = parts[1].strip() if len(parts) > 1 else None

    # Extract article
    article_match = re.search(r"Điều\s+(\d+)", main_part)
    if article_match:
        result["article"] = article_match.group(1)

    # Extract clause
    clause_match = re.search(r"Khoản\s+(\d+)", main_part)
    if clause_match:
        result["clause"] = clause_match.group(1)

    # Extract point
    point_match = re.search(r"Điểm\s+([a-zđ])", main_part)
    if point_match:
        result["point"] = point_match.group(1)

    # Extract document
    if doc_part:
        result["document"] = doc_part

    return result


def format_article_citation(article_number: int, title: Optional[str] = None) -> str:
    """
    Format article citation with optional title.

    Args:
        article_number: Article number
        title: Optional article title

    Returns:
        Formatted article citation
    """
    citation = f"Điều {article_number}"
    if title:
        citation += f". {title}"
    return citation
