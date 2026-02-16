"""
Cross-reference detector for Vietnamese legal documents.

Detects patterns like:
- "theo Điều X Luật số Y"
- "căn cứ Điều X"
- "quy định tại Điều X, Khoản Y"
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CrossReference:
    """Detected cross-reference in legal text."""

    source_article_id: str
    target_article_num: int
    target_clause_num: Optional[int]
    target_point_letter: Optional[str]
    target_so_hieu: Optional[str]  # Target document number
    reference_text: str
    reference_type: str  # Vietnamese types: 'THAM_CHIẾU', 'SỬA_ĐỔI', 'THAY_THẾ'
    confidence: float
    start_pos: int
    end_pos: int
    context_sentence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_article_id": self.source_article_id,
            "target_article_num": self.target_article_num,
            "target_clause_num": self.target_clause_num,
            "target_point_letter": self.target_point_letter,
            "target_so_hieu": self.target_so_hieu,
            "reference_text": self.reference_text,
            "reference_type": self.reference_type,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "context_sentence": self.context_sentence,
        }


def extract_context_sentence(
    text: str, start: int, end: int, max_length: int = 200
) -> str:
    """
    Extract the sentence containing a match position.

    Finds sentence boundaries (., ;, newline) around the match and returns
    the sentence with some context.
    """
    delimiters = ".;:\n"

    # Find sentence start (search backwards for delimiter)
    sent_start = start
    for i in range(start - 1, max(0, start - max_length), -1):
        if text[i] in delimiters:
            sent_start = i + 1
            break
    else:
        sent_start = max(0, start - max_length // 2)

    # Find sentence end (search forwards for delimiter)
    sent_end = end
    for i in range(end, min(len(text), end + max_length)):
        if text[i] in delimiters:
            sent_end = i + 1
            break
    else:
        sent_end = min(len(text), end + max_length // 2)

    # Extract and clean
    context = text[sent_start:sent_end].strip()

    # Truncate if still too long
    if len(context) > max_length:
        match_rel_start = start - sent_start
        context_start = max(0, match_rel_start - max_length // 3)
        context = context[context_start:context_start + max_length]
        if context_start > 0:
            context = "..." + context
        if sent_end - sent_start > max_length:
            context = context + "..."

    return context


class CrossReferenceDetector:
    """
    Detect cross-references in Vietnamese legal text.

    Usage:
        detector = CrossReferenceDetector()
        refs = detector.detect(
            text="Theo Điều 5, Khoản 2 Luật số 59/2020/QH14...",
            source_article_id="uuid-123"
        )
    """

    # Reference patterns with named groups (ordered by specificity)
    PATTERNS = [
        # Full reference: "theo Điều X, Khoản Y, Điểm Z Luật số ABC"
        (
            r"(?:theo|căn\s+cứ|quy\s+định\s+(?:tại)?|áp\s+dụng)\s+"
            r"Điều\s+(?P<article>\d+)"
            r"(?:[,\s]+Khoản\s+(?P<clause>\d+))?"
            r"(?:[,\s]+Điểm\s+(?P<point>[a-zđ]))?"
            r"(?:\s+(?:Luật|Nghị\s+định|Thông\s+tư)\s+(?:số\s+)?(?P<law>\d+/\d{4}/[A-Za-z0-9-]+))?",
            "THAM_CHIẾU",
            0.95,
        ),
        # Amendment: "được sửa đổi bởi Điều X"
        (
            r"(?:được\s+)?(?:sửa\s+đổi|bổ\s+sung)\s+(?:bởi|theo|tại)\s+"
            r"Điều\s+(?P<article>\d+)"
            r"(?:\s+(?:Luật|Nghị\s+định)\s+(?:số\s+)?(?P<law>\d+/\d{4}/[A-Za-z0-9-]+))?",
            "SỬA_ĐỔI",
            0.90,
        ),
        # Supersedes: "thay thế Điều X"
        (
            r"thay\s+thế\s+(?:cho\s+)?"
            r"Điều\s+(?P<article>\d+)"
            r"(?:\s+(?:Luật|Nghị\s+định)\s+(?:số\s+)?(?P<law>\d+/\d{4}/[A-Za-z0-9-]+))?",
            "THAY_THẾ",
            0.90,
        ),
        # Simple reference: "tại Điều X" or "của Điều X"
        (
            r"(?:tại|của|theo)\s+Điều\s+(?P<article>\d+)"
            r"(?:[,\s]+Khoản\s+(?P<clause>\d+))?",
            "THAM_CHIẾU",
            0.75,
        ),
    ]

    def detect(
        self,
        text: str,
        source_article_id: str,
        current_so_hieu: Optional[str] = None,
    ) -> List[CrossReference]:
        """
        Detect cross-references in text.

        Args:
            text: Article or clause text to analyze
            source_article_id: UUID/ID of source article
            current_so_hieu: Current document's số hiệu (for context)

        Returns:
            List of detected CrossReference objects
        """
        references = []
        seen_spans = set()

        for pattern, ref_type, base_confidence in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.UNICODE):
                start, end = match.start(), match.end()

                # Skip overlapping matches
                if any(
                    start < s_end and end > s_start for s_start, s_end in seen_spans
                ):
                    continue

                seen_spans.add((start, end))
                groups = match.groupdict()

                # Extract article number
                article_num = int(groups.get("article") or 0)
                if not article_num:
                    continue

                # Extract optional clause/point
                clause_num = int(groups["clause"]) if groups.get("clause") else None
                point_letter = groups.get("point")

                # Target document
                target_so_hieu = groups.get("law") or None

                # Adjust confidence based on specificity
                confidence = base_confidence
                if target_so_hieu:
                    confidence = min(confidence + 0.05, 1.0)
                if clause_num:
                    confidence = min(confidence + 0.02, 1.0)

                # Extract context sentence
                context = extract_context_sentence(text, start, end)

                ref = CrossReference(
                    source_article_id=source_article_id,
                    target_article_num=article_num,
                    target_clause_num=clause_num,
                    target_point_letter=point_letter,
                    target_so_hieu=target_so_hieu,
                    reference_text=match.group(0).strip(),
                    reference_type=ref_type,
                    confidence=confidence,
                    start_pos=start,
                    end_pos=end,
                    context_sentence=context,
                )
                references.append(ref)

        return sorted(references, key=lambda r: r.start_pos)

    def detect_all(
        self, texts: List[Dict[str, str]]
    ) -> List[CrossReference]:
        """
        Detect cross-references in multiple texts.

        Args:
            texts: List of {"id": str, "content": str, "so_hieu"?: str}

        Returns:
            List of all detected cross-references
        """
        all_refs = []
        for item in texts:
            if item.get("content"):
                refs = self.detect(
                    text=item["content"],
                    source_article_id=item["id"],
                    current_so_hieu=item.get("so_hieu"),
                )
                all_refs.extend(refs)
        return all_refs
