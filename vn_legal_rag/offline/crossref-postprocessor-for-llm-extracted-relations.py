"""
Cross-Reference Post-Processor for Vietnamese Legal Documents.

Converts LLM-extracted relations to structured cross-references.

This module implements post-processing of LLM relations to create
cross-references with proper IDs for the legal_cross_references table.

Flow:
    LLM Relations (THAM_CHIẾU, SỬA_ĐỔI, etc.)
    → Filter crossref predicates
    → Parse article/document references
    → Resolve IDs
    → Create CrossReference records
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import LegalCrossReferenceModel, make_crossref_id
from .relation_types import CROSSREF_PREDICATES, CROSSREF_RELATION_MAPPING


@dataclass
class ParsedReference:
    """Parsed reference from LLM-extracted relation object."""

    article_num: Optional[int] = None
    clause_num: Optional[int] = None
    point_letter: Optional[str] = None
    document_ref: Optional[str] = None  # "Luật Doanh nghiệp", "Nghị định 47/2021"
    document_so_hieu: Optional[str] = None  # Normalized: "59-2020-QH14"
    raw_text: str = ""


@dataclass
class CrossRefCandidate:
    """Candidate cross-reference from LLM relation."""

    source_article_id: str
    predicate: str  # Vietnamese: THAM_CHIẾU, SỬA_ĐỔI, etc.
    crossref_type: str  # English: references, amends, etc.
    parsed_ref: ParsedReference
    confidence: float
    evidence: str = ""


class CrossRefPostProcessor:
    """
    Post-process LLM-extracted relations to create cross-references.

    Example:
        >>> processor = CrossRefPostProcessor(db)
        >>> crossrefs = processor.process_relations(
        ...     relations=[
        ...         {"subject": "Điều 5", "predicate": "THAM_CHIẾU", "object": "Điều 64"},
        ...     ],
        ...     source_article_id="59-2020-QH14:d5",
        ...     current_document_id="59-2020-QH14"
        ... )
        >>> processor.store_crossrefs(crossrefs)
    """

    # Pattern to extract article number from text
    ARTICLE_PATTERN = re.compile(
        r"Điều\s+(\d+)", re.IGNORECASE | re.UNICODE
    )

    # Pattern to extract clause number
    CLAUSE_PATTERN = re.compile(
        r"[Kk]hoản\s+(\d+)", re.IGNORECASE | re.UNICODE
    )

    # Pattern to extract point letter
    POINT_PATTERN = re.compile(
        r"[Đđ]iểm\s+([a-zđ])", re.IGNORECASE | re.UNICODE
    )

    # Pattern to extract document reference (Luật, Nghị định, Thông tư)
    DOCUMENT_PATTERNS = [
        # "Luật Doanh nghiệp" or "Luật số 59/2020/QH14"
        re.compile(
            r"Luật\s+(?:số\s+)?(?:(\d+/\d{4}/[A-Za-z0-9Đđ-]+)|([A-Za-zÀ-ỹ\s]+))",
            re.IGNORECASE | re.UNICODE
        ),
        # "Nghị định số 47/2021/NĐ-CP" or "Nghị định 47/2021"
        re.compile(
            r"Nghị\s+định\s+(?:số\s+)?(\d+/\d{4}(?:/[A-Za-z0-9Đđ-]+)?)",
            re.IGNORECASE | re.UNICODE
        ),
        # "Thông tư số 01/2021/TT-BTC"
        re.compile(
            r"Thông\s+tư\s+(?:số\s+)?(\d+/\d{4}(?:/[A-Za-z0-9Đđ-]+)?)",
            re.IGNORECASE | re.UNICODE
        ),
    ]

    # Common document name aliases for resolution
    DOCUMENT_ALIASES: Dict[str, str] = {
        "Luật Doanh nghiệp": "59-2020-QH14",
        "Luật doanh nghiệp": "59-2020-QH14",
        "LDN": "59-2020-QH14",
        "Luật này": None,  # Refers to current document
        "Nghị định này": None,  # Refers to current document
        "Thông tư này": None,  # Refers to current document
    }

    def __init__(self, db=None):
        """
        Initialize post-processor.

        Args:
            db: LegalDocumentDB instance for ID resolution (optional)
        """
        self.db = db
        self._so_hieu_cache: Dict[str, str] = {}  # document_ref -> so_hieu

    def process_relations(
        self,
        relations: List[Dict[str, Any]],
        source_article_id: str,
        current_document_id: str,
    ) -> List[CrossRefCandidate]:
        """
        Process LLM-extracted relations and extract cross-references.

        Args:
            relations: List of relation dicts from LLM
            source_article_id: ID of the source article (e.g., "59-2020-QH14:d5")
            current_document_id: ID of the current document (e.g., "59-2020-QH14")

        Returns:
            List of CrossRefCandidate objects
        """
        candidates = []

        for rel in relations:
            predicate = rel.get("predicate", "").upper()

            # Check if this is a cross-reference relation
            if predicate not in CROSSREF_PREDICATES:
                continue

            # Get crossref type from mapping
            crossref_type = CROSSREF_RELATION_MAPPING.get(predicate)
            if not crossref_type:
                continue

            # Parse the object to extract reference info
            object_text = rel.get("object", "")
            parsed_ref = self._parse_reference(object_text, current_document_id)

            # Skip if no article number found
            if parsed_ref.article_num is None and not parsed_ref.document_ref:
                continue

            candidate = CrossRefCandidate(
                source_article_id=source_article_id,
                predicate=predicate,
                crossref_type=crossref_type,
                parsed_ref=parsed_ref,
                confidence=rel.get("confidence", 0.8),
                evidence=rel.get("evidence", object_text),
            )
            candidates.append(candidate)

        return candidates

    def _parse_reference(
        self, text: str, current_document_id: str
    ) -> ParsedReference:
        """
        Parse reference text to extract article, clause, point, and document.

        Args:
            text: Reference text (e.g., "Điều 64 Luật Doanh nghiệp")
            current_document_id: Current document ID for self-references

        Returns:
            ParsedReference object
        """
        parsed = ParsedReference(raw_text=text)

        # Extract article number
        article_match = self.ARTICLE_PATTERN.search(text)
        if article_match:
            parsed.article_num = int(article_match.group(1))

        # Extract clause number
        clause_match = self.CLAUSE_PATTERN.search(text)
        if clause_match:
            parsed.clause_num = int(clause_match.group(1))

        # Extract point letter
        point_match = self.POINT_PATTERN.search(text)
        if point_match:
            parsed.point_letter = point_match.group(1).lower()

        # PRIORITY 1: Check known aliases FIRST
        for alias, resolved in self.DOCUMENT_ALIASES.items():
            if alias in text:
                parsed.document_ref = alias
                if resolved is None:
                    # Self-reference (Luật này, Nghị định này, etc.)
                    parsed.document_so_hieu = current_document_id
                else:
                    parsed.document_so_hieu = resolved
                return parsed

        # PRIORITY 2: Extract document reference via regex patterns
        for pattern in self.DOCUMENT_PATTERNS:
            doc_match = pattern.search(text)
            if doc_match:
                groups = doc_match.groups()
                doc_ref = next((g for g in groups if g), None)
                if doc_ref:
                    parsed.document_ref = doc_ref.strip()
                    parsed.document_so_hieu = self._resolve_document_id(
                        parsed.document_ref, current_document_id
                    )
                break

        # PRIORITY 3: If no document specified, assume current document
        if not parsed.document_so_hieu and parsed.article_num:
            parsed.document_so_hieu = current_document_id

        return parsed

    def _resolve_document_id(
        self, document_ref: str, current_document_id: str
    ) -> Optional[str]:
        """Resolve document reference to normalized ID."""
        # Check cache first
        if document_ref in self._so_hieu_cache:
            return self._so_hieu_cache[document_ref]

        # Check aliases
        if document_ref in self.DOCUMENT_ALIASES:
            resolved = self.DOCUMENT_ALIASES[document_ref]
            if resolved is None:
                return current_document_id
            return resolved

        # Try to normalize the reference format
        normalized = self._normalize_so_hieu(document_ref)
        if normalized:
            self._so_hieu_cache[document_ref] = normalized
            return normalized

        # If DB available, try to look up
        if self.db:
            resolved = self._lookup_document_in_db(document_ref)
            if resolved:
                self._so_hieu_cache[document_ref] = resolved
                return resolved

        return None

    def _normalize_so_hieu(self, so_hieu: str) -> Optional[str]:
        """
        Normalize số hiệu to ID format.

        "59/2020/QH14" -> "59-2020-QH14"
        "47/2021/NĐ-CP" -> "47-2021-ND"
        """
        if not so_hieu:
            return None

        # Replace slashes with dashes
        normalized = so_hieu.replace("/", "-")

        # Remove Vietnamese diacritics from suffix
        normalized = normalized.replace("Đ", "D").replace("đ", "d")
        normalized = re.sub(r"-CP$", "", normalized)
        normalized = re.sub(r"-BTC$", "", normalized)

        return normalized

    def _lookup_document_in_db(self, document_ref: str) -> Optional[str]:
        """Look up document in database by reference text."""
        if not self.db:
            return None

        normalized = self._normalize_so_hieu(document_ref)
        if normalized:
            try:
                with self.db.SessionLocal() as session:
                    from .models import LegalDocumentModel
                    doc = session.query(LegalDocumentModel).filter(
                        LegalDocumentModel.id == normalized
                    ).first()
                    if doc:
                        return doc.id
            except Exception:
                pass

        return None

    def resolve_target_article_id(
        self, candidate: CrossRefCandidate
    ) -> Optional[str]:
        """
        Resolve target article ID from parsed reference.

        Args:
            candidate: CrossRefCandidate with parsed reference

        Returns:
            Target article ID or None if unresolved
        """
        parsed = candidate.parsed_ref

        if not parsed.article_num:
            return None

        if not parsed.document_so_hieu:
            return None

        # Build target article ID
        target_id = f"{parsed.document_so_hieu}:d{parsed.article_num}"

        return target_id

    def create_crossref_models(
        self, candidates: List[CrossRefCandidate]
    ) -> List[LegalCrossReferenceModel]:
        """
        Create cross-reference model objects from candidates.

        Args:
            candidates: List of CrossRefCandidate

        Returns:
            List of LegalCrossReferenceModel ready for storage
        """
        models = []

        for idx, candidate in enumerate(candidates):
            target_id = self.resolve_target_article_id(candidate)

            crossref_id = make_crossref_id(
                candidate.source_article_id,
                target_id,
                idx
            )

            model = LegalCrossReferenceModel(
                id=crossref_id,
                source_article_id=candidate.source_article_id,
                target_article_id=target_id,
                target_document_so_hieu=(
                    candidate.parsed_ref.document_so_hieu
                    if candidate.parsed_ref.document_so_hieu != candidate.source_article_id.split(":")[0]
                    else None
                ),
                reference_text=candidate.evidence or candidate.parsed_ref.raw_text,
                reference_type=candidate.crossref_type,
                confidence=candidate.confidence,
            )
            models.append(model)

        return models

    def store_crossrefs(
        self, candidates: List[CrossRefCandidate]
    ) -> int:
        """
        Store cross-references in database.

        Args:
            candidates: List of CrossRefCandidate

        Returns:
            Number of cross-references stored
        """
        if not self.db:
            raise ValueError("Database connection required for storage")

        models = self.create_crossref_models(candidates)

        stored = 0
        with self.db.SessionLocal() as session:
            for model in models:
                # Check for duplicates
                existing = session.query(LegalCrossReferenceModel).filter(
                    LegalCrossReferenceModel.source_article_id == model.source_article_id,
                    LegalCrossReferenceModel.target_article_id == model.target_article_id,
                    LegalCrossReferenceModel.reference_type == model.reference_type,
                ).first()

                if not existing:
                    session.add(model)
                    stored += 1

            session.commit()

        return stored


def extract_crossrefs_from_relations(
    relations: List[Dict[str, Any]],
    source_article_id: str,
    current_document_id: str,
    db=None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract cross-references from LLM relations.

    Args:
        relations: List of relation dicts from LLM
        source_article_id: Source article ID
        current_document_id: Current document ID
        db: Optional LegalDocumentDB for ID resolution

    Returns:
        List of cross-reference dicts ready for storage
    """
    processor = CrossRefPostProcessor(db)
    candidates = processor.process_relations(
        relations, source_article_id, current_document_id
    )

    # Convert to dicts for easier handling
    results = []
    for candidate in candidates:
        target_id = processor.resolve_target_article_id(candidate)
        results.append({
            "source_article_id": candidate.source_article_id,
            "target_article_id": target_id,
            "target_document_so_hieu": candidate.parsed_ref.document_so_hieu,
            "reference_type": candidate.crossref_type,
            "reference_text": candidate.evidence or candidate.parsed_ref.raw_text,
            "confidence": candidate.confidence,
            "predicate": candidate.predicate,
            "parsed": {
                "article_num": candidate.parsed_ref.article_num,
                "clause_num": candidate.parsed_ref.clause_num,
                "point_letter": candidate.parsed_ref.point_letter,
                "document_ref": candidate.parsed_ref.document_ref,
            },
        })

    return results
