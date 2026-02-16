"""
Relation Validator for Vietnamese Legal KG Pipeline.

Post-processing validation pipeline for extracted relations:
1. Self-reference filter (subject != object)
2. Type normalization (UPPERCASE)
3. Auto-persist new relation types to JSON config
4. Semantic type validation (soft warning)
5. Evidence validation (filter invalid, boost exact matches)

Usage:
    >>> from vn_legal_rag.offline import RelationValidator
    >>> validator = RelationValidator(defined_types=LEGAL_RELATION_TYPES)
    >>> validated = validator.validate(relations, source_text)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..utils import get_logger


# Semantic type rules for soft validation
# Subject/Object entity types compatible with each relation type
SEMANTIC_TYPE_RULES: Dict[str, Dict[str, List[str]]] = {
    "CÓ_QUYỀN": {
        "subject_types": ["ORGANIZATION", "PERSON_ROLE"],
        "object_types": ["ACTION", "LEGAL_TERM"],
    },
    "CÓ_NGHĨA_VỤ": {
        "subject_types": ["ORGANIZATION", "PERSON_ROLE"],
        "object_types": ["ACTION", "LEGAL_TERM"],
    },
    "ĐỊNH_NGHĨA_LÀ": {
        "subject_types": ["ORGANIZATION", "LEGAL_TERM", "PERSON_ROLE"],
        "object_types": ["LEGAL_TERM", "ORGANIZATION"],
    },
    "VI_PHẠM": {
        "subject_types": ["ACTION", "ORGANIZATION", "PERSON_ROLE"],
        "object_types": ["LEGAL_TERM", "ACTION"],
    },
    "CHỊU_TRÁCH_NHIỆM": {
        "subject_types": ["ORGANIZATION", "PERSON_ROLE"],
        "object_types": ["ACTION", "LEGAL_TERM"],
    },
    "NGHIÊM_CẤM": {
        "subject_types": ["ORGANIZATION", "PERSON_ROLE"],
        "object_types": ["ACTION"],
    },
    "YÊU_CẦU": {
        "subject_types": ["ORGANIZATION", "PERSON_ROLE", "LEGAL_TERM", "LEGAL_REFERENCE"],
        "object_types": ["LEGAL_TERM", "ACTION", "ORGANIZATION", "MONETARY", "DURATION"],
    },
    "BAO_GỒM": {
        "subject_types": ["ORGANIZATION", "LEGAL_TERM", "LEGAL_REFERENCE"],
        "object_types": ["ORGANIZATION", "LEGAL_TERM", "ACTION", "LEGAL_REFERENCE"],
    },
    "THAM_CHIẾU": {
        "subject_types": ["LEGAL_TERM", "LEGAL_REFERENCE"],
        "object_types": ["LEGAL_TERM", "LEGAL_REFERENCE"],
    },
    "QUY_ĐỊNH_VỀ": {
        "subject_types": ["LEGAL_REFERENCE", "LEGAL_TERM"],
        "object_types": ["ORGANIZATION", "LEGAL_TERM", "ACTION", "PERSON_ROLE"],
    },
    "BAN_HÀNH": {
        "subject_types": ["ORGANIZATION"],
        "object_types": ["LEGAL_REFERENCE", "LEGAL_TERM"],
    },
    "ÁP_DỤNG_TẠI": {
        "subject_types": ["LEGAL_TERM", "LEGAL_REFERENCE", "ORGANIZATION"],
        "object_types": ["LOCATION"],
    },
    "THỰC_HIỆN": {
        "subject_types": ["ORGANIZATION", "PERSON_ROLE"],
        "object_types": ["ACTION", "LEGAL_TERM"],
    },
    "PHỐI_HỢP": {
        "subject_types": ["ORGANIZATION", "PERSON_ROLE"],
        "object_types": ["ORGANIZATION", "PERSON_ROLE"],
    },
}

DEFAULT_DISCOVERED_TYPES_PATH = Path("data/discovered_relation_types.json")


@dataclass
class ValidationStats:
    """Statistics from validation run."""

    total_input: int = 0
    self_references_filtered: int = 0
    types_normalized: int = 0
    new_types_discovered: int = 0
    semantic_warnings: int = 0
    evidence_invalid: int = 0
    evidence_boosted: int = 0
    total_output: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_input": self.total_input,
            "self_references_filtered": self.self_references_filtered,
            "types_normalized": self.types_normalized,
            "new_types_discovered": self.new_types_discovered,
            "semantic_warnings": self.semantic_warnings,
            "evidence_invalid": self.evidence_invalid,
            "evidence_boosted": self.evidence_boosted,
            "total_output": self.total_output,
        }


class RelationValidator:
    """
    Post-processing validator for extracted relations.

    Validation pipeline:
    1. Filter self-references (subject != object)
    2. Normalize types to UPPERCASE
    3. Auto-persist new types to JSON config
    4. Evidence validation (filter/boost based on source text)
    5. Semantic type check (soft validation - warn only)

    Example:
        >>> from vn_legal_rag.offline.relation_types import LEGAL_RELATION_TYPES
        >>> validator = RelationValidator(defined_types=LEGAL_RELATION_TYPES)
        >>> relations = [
        ...     {"subject_text": "Công ty", "predicate": "có_quyền", "object_text": "kinh doanh"},
        ... ]
        >>> validated = validator.validate(relations)
    """

    def __init__(
        self,
        defined_types: List[str],
        discovered_types_path: Optional[Path] = None,
        enable_semantic_validation: bool = True,
        require_evidence: bool = True,
        evidence_fuzzy_threshold: float = 0.7,
    ):
        """
        Initialize validator.

        Args:
            defined_types: List of predefined relation types
            discovered_types_path: Path to JSON file for discovered types
            enable_semantic_validation: Enable semantic type checking (soft)
            require_evidence: Require evidence field for relations
            evidence_fuzzy_threshold: Fuzzy match threshold for evidence
        """
        self.logger = get_logger("relation_validator")
        self.defined_types = set(t.upper() for t in defined_types)
        self.discovered_types_path = discovered_types_path or DEFAULT_DISCOVERED_TYPES_PATH
        self.enable_semantic_validation = enable_semantic_validation
        self.require_evidence = require_evidence
        self.evidence_fuzzy_threshold = evidence_fuzzy_threshold

        # Load previously discovered types
        self.discovered_types = self._load_discovered_types()

        # Stats for reporting
        self.stats = ValidationStats()

    def _load_discovered_types(self) -> Set[str]:
        """Load previously discovered relation types from JSON."""
        if not self.discovered_types_path.exists():
            return set()

        try:
            with open(self.discovered_types_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("relation_types", []))
        except Exception as e:
            self.logger.warning(f"Failed to load discovered types: {e}")
            return set()

    def _save_discovered_types(self):
        """Save discovered relation types to JSON."""
        try:
            self.discovered_types_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.discovered_types_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "relation_types": sorted(list(self.discovered_types)),
                        "count": len(self.discovered_types),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.logger.info(f"Saved {len(self.discovered_types)} discovered types")
        except Exception as e:
            self.logger.error(f"Failed to save discovered types: {e}")

    def validate(
        self,
        relations: List[Dict[str, Any]],
        source_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run full validation pipeline.

        Args:
            relations: List of relation dicts with subject_text, predicate, object_text
            source_text: Original source text for evidence validation

        Returns:
            Validated relations list
        """
        self.stats = ValidationStats(total_input=len(relations))

        # Step 1: Filter self-references
        relations = self.filter_self_references(relations)

        # Step 2: Normalize types to UPPERCASE
        relations = self.normalize_types(relations)

        # Step 3: Persist new types (instead of rejecting)
        relations = self.persist_new_types(relations)

        # Step 4: Evidence validation
        if self.require_evidence and source_text:
            relations = self.validate_evidence(relations, source_text)

        # Step 5: Semantic type validation (soft - warn only)
        if self.enable_semantic_validation:
            self.check_semantic_types(relations)

        self.stats.total_output = len(relations)
        self._log_stats()

        return relations

    def filter_self_references(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove relations where subject == object.

        Comparison is case-insensitive and whitespace-normalized.
        """
        valid = []
        for r in relations:
            subj = self._normalize_text(r.get("subject_text", ""))
            obj = self._normalize_text(r.get("object_text", ""))

            if subj != obj:
                valid.append(r)
            else:
                self.stats.self_references_filtered += 1
                self.logger.debug(
                    f"Filtered self-reference: {subj} -> {r.get('predicate')} -> {obj}"
                )

        return valid

    def normalize_types(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize all predicate types to UPPERCASE.

        Also replaces spaces with underscores.
        """
        for r in relations:
            predicate = r.get("predicate", "")
            original = predicate

            # Normalize: spaces -> underscores, uppercase
            normalized = predicate.upper().replace(" ", "_")
            # Remove multiple underscores
            normalized = re.sub(r"_+", "_", normalized).strip("_")

            if normalized != original:
                self.stats.types_normalized += 1
                self.logger.debug(f"Normalized type: {original} -> {normalized}")

            r["predicate"] = normalized

        return relations

    def persist_new_types(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Auto-persist new relation types to JSON config.

        Instead of rejecting undefined types, save them for future use.
        """
        new_types = set()

        for r in relations:
            pred = r.get("predicate", "").upper()

            # Check if it's a new type
            if pred not in self.defined_types and pred not in self.discovered_types:
                new_types.add(pred)
                self.discovered_types.add(pred)
                self.stats.new_types_discovered += 1
                self.logger.info(f"Discovered new relation type: {pred}")

        # Save if new types were discovered
        if new_types:
            self._save_discovered_types()

        return relations

    def validate_evidence(
        self,
        relations: List[Dict[str, Any]],
        source_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Validate that each relation has evidence grounded in source text.

        - Filter relations without evidence field
        - Filter relations where evidence is not found in source text
        - Boost confidence for exact match evidence

        Args:
            relations: List of relation dicts
            source_text: Original source text

        Returns:
            Relations with valid evidence
        """
        valid = []
        source_lower = source_text.lower()

        for r in relations:
            evidence = r.get("evidence", "")

            # Filter relations without evidence
            if not evidence:
                self.stats.evidence_invalid += 1
                self.logger.debug(
                    f"Filtered relation without evidence: "
                    f"{r.get('subject_text')} -> {r.get('predicate')} -> {r.get('object_text')}"
                )
                continue

            # Check if evidence exists in source text
            evidence_lower = evidence.lower().strip()

            # Exact match (case-insensitive)
            if evidence_lower in source_lower:
                # Boost confidence for exact match
                old_conf = r.get("confidence", 0.8)
                r["confidence"] = min(1.0, old_conf + 0.05)
                self.stats.evidence_boosted += 1
                valid.append(r)
                continue

            # Fuzzy match using substring containment of key terms
            if self._fuzzy_evidence_match(evidence_lower, source_lower):
                valid.append(r)
                continue

            # Evidence not found in source - filter out
            self.stats.evidence_invalid += 1
            self.logger.debug(
                f"Filtered relation with invalid evidence: "
                f"{r.get('subject_text')} -> {r.get('predicate')} -> {r.get('object_text')} "
                f"(evidence: '{evidence[:50]}...')"
            )

        return valid

    def _fuzzy_evidence_match(self, evidence: str, source: str) -> bool:
        """
        Fuzzy match evidence against source text.

        Checks if most significant words from evidence appear in source.
        """
        # Extract significant words (length > 2)
        evidence_words = set(w for w in evidence.split() if len(w) > 2)
        if not evidence_words:
            return False

        # Count how many evidence words appear in source
        matches = sum(1 for w in evidence_words if w in source)
        match_ratio = matches / len(evidence_words)

        return match_ratio >= self.evidence_fuzzy_threshold

    def check_semantic_types(self, relations: List[Dict[str, Any]]):
        """
        Soft validation for semantic type compatibility.

        Logs warnings but does not reject relations.
        Helps identify potential extraction errors for review.
        """
        for r in relations:
            pred = r.get("predicate", "").upper()

            if pred not in SEMANTIC_TYPE_RULES:
                continue

            rules = SEMANTIC_TYPE_RULES[pred]

            # Check subject type compatibility
            subject_type = r.get("subject_type", "")
            if subject_type and subject_type not in rules.get("subject_types", []):
                self.stats.semantic_warnings += 1
                self.logger.debug(
                    f"Semantic warning: {pred} expects subject types "
                    f"{rules['subject_types']}, got {subject_type}"
                )

            # Check object type compatibility
            object_type = r.get("object_type", "")
            if object_type and object_type not in rules.get("object_types", []):
                self.stats.semantic_warnings += 1
                self.logger.debug(
                    f"Semantic warning: {pred} expects object types "
                    f"{rules['object_types']}, got {object_type}"
                )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, trim, collapse whitespace)."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _log_stats(self):
        """Log validation statistics."""
        self.logger.info(
            f"Validation stats: "
            f"input={self.stats.total_input}, "
            f"output={self.stats.total_output}, "
            f"self_refs_filtered={self.stats.self_references_filtered}, "
            f"types_normalized={self.stats.types_normalized}, "
            f"new_types={self.stats.new_types_discovered}, "
            f"evidence_invalid={self.stats.evidence_invalid}, "
            f"evidence_boosted={self.stats.evidence_boosted}"
        )

    def get_all_relation_types(self) -> Set[str]:
        """Get all known relation types (defined + discovered)."""
        return self.defined_types | self.discovered_types

    def reset_stats(self):
        """Reset validation statistics."""
        self.stats = ValidationStats()


def validate_relations(
    relations: List[Dict[str, Any]],
    defined_types: Optional[List[str]] = None,
    source_text: Optional[str] = None,
    require_evidence: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function to validate relations.

    Args:
        relations: List of relation dicts
        defined_types: Optional list of defined relation types
        source_text: Optional source text for evidence validation
        require_evidence: Whether to require evidence field

    Returns:
        Validated relations
    """
    from .relation_types import LEGAL_RELATION_TYPES

    types = defined_types or LEGAL_RELATION_TYPES

    validator = RelationValidator(
        defined_types=types,
        require_evidence=require_evidence,
    )

    return validator.validate(relations, source_text)
