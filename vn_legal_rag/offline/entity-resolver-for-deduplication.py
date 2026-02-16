"""
Entity Resolver for Vietnamese Legal Document Deduplication.

Multi-pass entity resolution pipeline:
1. Exact match cache (text, type, document_id)
2. Synonym resolution (case variants, abbreviations)
3. Abbreviation expansion (TNHH -> Trach nhiem huu han)
4. Document-scoped deduplication

Preserves original text in metadata for traceability.

Usage:
    >>> from vn_legal_rag.offline import EntityResolver
    >>> resolver = EntityResolver()
    >>> entity_id = resolver.resolve("HĐQT", "PERSON_ROLE", "59-2020-QH14")
    >>> metadata = resolver.get_entity_metadata("HĐQT")
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils import get_logger
from .entity_types import LEGAL_ABBREVIATIONS, LEGAL_SYNONYMS


@dataclass
class ResolverStats:
    """Statistics from resolver operations."""

    cache_hits: int = 0
    abbrev_expansions: int = 0
    synonym_resolutions: int = 0
    new_entities: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "abbrev_expansions": self.abbrev_expansions,
            "synonym_resolutions": self.synonym_resolutions,
            "new_entities": self.new_entities,
        }


@dataclass
class EntityMetadata:
    """Metadata for resolved entity."""

    original_text: str
    expanded_text: Optional[str] = None
    is_abbreviation: bool = False
    full_form: Optional[str] = None
    canonical_form: Optional[str] = None


class EntityResolver:
    """
    Resolve entities to canonical IDs with abbreviation handling.

    Resolution pipeline:
    1. Exact match cache (text, type, document_id)
    2. Synonym resolution (handles case variants)
    3. Abbreviation expansion (TNHH -> full form)
    4. Create new entity ID with document scope

    Example:
        >>> resolver = EntityResolver(scope_by_document=True)
        >>> # Same entity resolved to same ID
        >>> id1 = resolver.resolve("HĐQT", "PERSON_ROLE", "59-2020-QH14")
        >>> id2 = resolver.resolve("Hội đồng quản trị", "PERSON_ROLE", "59-2020-QH14")
        >>> assert id1 == id2
    """

    def __init__(
        self,
        abbreviations: Optional[Dict[str, str]] = None,
        synonyms: Optional[Dict[str, List[str]]] = None,
        scope_by_document: bool = True,
    ):
        """
        Initialize entity resolver.

        Args:
            abbreviations: Dict of abbreviation -> full form mappings
            synonyms: Dict of canonical -> variants mappings
            scope_by_document: Whether to scope entity IDs by document
        """
        self.logger = get_logger("entity_resolver")
        self.abbreviations = abbreviations or LEGAL_ABBREVIATIONS
        self.synonyms = synonyms or LEGAL_SYNONYMS
        self.scope_by_document = scope_by_document

        # Cache: (text, type, document_id) -> entity_id
        self.cache: Dict[Tuple[str, str, str], str] = {}

        # Build lookup tables
        self._abbrev_lookup = {k.upper(): v for k, v in self.abbreviations.items()}
        self._full_to_abbrev = {v.upper(): k for k, v in self.abbreviations.items()}

        # Build synonym lookup: variant (lowercase) -> canonical form
        self._synonym_lookup: Dict[str, str] = {}
        for canonical, variants in self.synonyms.items():
            self._synonym_lookup[canonical.lower()] = canonical
            for variant in variants:
                self._synonym_lookup[variant.lower()] = canonical

        # Stats
        self.stats = ResolverStats()

    def resolve(
        self,
        entity_text: str,
        entity_type: str,
        document_id: str,
    ) -> str:
        """
        Resolve entity to canonical ID.

        Resolution pipeline:
        1. Exact match cache
        2. Synonym resolution (case variants)
        3. Abbreviation expansion
        4. Create new entity ID

        Args:
            entity_text: Original entity text
            entity_type: Entity type label
            document_id: Document ID for scoping

        Returns:
            Canonical entity ID
        """
        # Normalize for comparison
        norm_text = self._normalize_for_comparison(entity_text)
        scope_id = document_id if self.scope_by_document else "global"

        # 1. Check exact cache
        cache_key = (norm_text, entity_type, scope_id)
        if cache_key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[cache_key]

        # 2. Try synonym resolution
        canonical = self._get_canonical_form(norm_text)
        if canonical != norm_text:
            canonical_key = (canonical, entity_type, scope_id)
            if canonical_key in self.cache:
                self.stats.synonym_resolutions += 1
                self.cache[cache_key] = self.cache[canonical_key]
                return self.cache[canonical_key]
            # Use canonical form for entity ID
            norm_text = canonical

        # 3. Try abbreviation expansion and check cache
        expanded = self.expand_abbreviations(entity_text)
        if expanded != entity_text:
            norm_expanded = self._normalize_for_comparison(expanded)
            expanded_key = (norm_expanded, entity_type, scope_id)
            if expanded_key in self.cache:
                # Link abbreviation to expanded form's ID
                self.stats.abbrev_expansions += 1
                self.cache[cache_key] = self.cache[expanded_key]
                return self.cache[expanded_key]

        # 4. Create new entity ID
        text_for_id = canonical if canonical != norm_text else entity_text
        entity_id = self._create_entity_id(text_for_id, document_id)
        self.cache[cache_key] = entity_id
        self.stats.new_entities += 1

        # Also cache canonical and expanded forms
        if canonical != norm_text:
            canonical_key = (canonical, entity_type, scope_id)
            self.cache[canonical_key] = entity_id

        if expanded != entity_text:
            norm_expanded = self._normalize_for_comparison(expanded)
            expanded_key = (norm_expanded, entity_type, scope_id)
            self.cache[expanded_key] = entity_id

        return entity_id

    def _get_canonical_form(self, text: str) -> str:
        """
        Get canonical form from synonym dictionary.

        Maps variants to canonical form (lowercase).
        E.g., "Doanh nghiệp" -> "doanh nghiệp"
        """
        text_lower = text.lower().strip()
        return self._synonym_lookup.get(text_lower, text)

    def expand_abbreviations(self, text: str) -> str:
        """
        Expand known abbreviations in text.

        Example: "Công ty TNHH ABC" -> "Công ty Trách nhiệm hữu hạn ABC"
        """
        result = text

        for abbrev, full_form in self.abbreviations.items():
            # Replace whole word only (case-insensitive)
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            result = re.sub(pattern, full_form, result, flags=re.IGNORECASE)

        return result

    def get_abbreviation(self, text: str) -> Optional[str]:
        """
        Get abbreviation for a full form if exists.

        Example: "Trách nhiệm hữu hạn" -> "TNHH"
        """
        norm = text.upper().strip()
        return self._full_to_abbrev.get(norm)

    def is_abbreviation(self, text: str) -> bool:
        """Check if text is a known abbreviation."""
        return text.upper().strip() in self._abbrev_lookup

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison (lowercase, trim, collapse whitespace)."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _create_entity_id(self, entity_text: str, document_id: str) -> str:
        """Create entity ID from text and document."""
        from .incremental_knowledge_graph_builder import slugify_vietnamese

        slug = slugify_vietnamese(entity_text)

        if self.scope_by_document:
            return f"{document_id}:{slug}"
        else:
            return slug

    def get_entity_metadata(self, entity_text: str) -> EntityMetadata:
        """
        Get additional metadata for entity.

        Returns:
            EntityMetadata with original_text, expanded_text, is_abbreviation, etc.
        """
        expanded = self.expand_abbreviations(entity_text)
        is_abbrev = self.is_abbreviation(entity_text)
        canonical = self._get_canonical_form(self._normalize_for_comparison(entity_text))

        return EntityMetadata(
            original_text=entity_text,
            expanded_text=expanded if expanded != entity_text else None,
            is_abbreviation=is_abbrev,
            full_form=self._abbrev_lookup.get(entity_text.upper()) if is_abbrev else None,
            canonical_form=canonical if canonical != entity_text.lower() else None,
        )

    def clear_cache(self):
        """Clear entity cache (useful between pipeline runs)."""
        self.cache.clear()
        self.stats = ResolverStats()

    def log_stats(self):
        """Log resolver statistics."""
        self.logger.info(
            f"EntityResolver stats: "
            f"cache_hits={self.stats.cache_hits}, "
            f"abbrev_expansions={self.stats.abbrev_expansions}, "
            f"synonym_resolutions={self.stats.synonym_resolutions}, "
            f"new_entities={self.stats.new_entities}"
        )

    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def get_known_abbreviations(self) -> Set[str]:
        """Get all known abbreviations."""
        return set(self._abbrev_lookup.keys())

    def add_abbreviation(self, abbrev: str, full_form: str):
        """
        Add a new abbreviation mapping.

        Args:
            abbrev: Abbreviation (e.g., "HĐQT")
            full_form: Full form (e.g., "Hội đồng quản trị")
        """
        self.abbreviations[abbrev] = full_form
        self._abbrev_lookup[abbrev.upper()] = full_form
        self._full_to_abbrev[full_form.upper()] = abbrev

    def add_synonym(self, canonical: str, variants: List[str]):
        """
        Add a new synonym mapping.

        Args:
            canonical: Canonical form
            variants: List of variant forms
        """
        self.synonyms[canonical] = variants
        self._synonym_lookup[canonical.lower()] = canonical
        for variant in variants:
            self._synonym_lookup[variant.lower()] = canonical


def resolve_entity(
    entity_text: str,
    entity_type: str,
    document_id: str,
    resolver: Optional[EntityResolver] = None,
) -> str:
    """
    Convenience function to resolve a single entity.

    Args:
        entity_text: Entity text to resolve
        entity_type: Entity type label
        document_id: Document ID for scoping
        resolver: Optional existing resolver instance

    Returns:
        Canonical entity ID
    """
    if resolver is None:
        resolver = EntityResolver()

    return resolver.resolve(entity_text, entity_type, document_id)


def batch_resolve_entities(
    entities: List[Dict[str, Any]],
    document_id: str,
    text_key: str = "text",
    type_key: str = "label",
) -> List[Dict[str, Any]]:
    """
    Batch resolve entities and add canonical IDs.

    Args:
        entities: List of entity dicts
        document_id: Document ID for scoping
        text_key: Key for entity text in dict
        type_key: Key for entity type in dict

    Returns:
        Entities with added 'entity_id' field
    """
    resolver = EntityResolver()

    for entity in entities:
        text = entity.get(text_key, "")
        etype = entity.get(type_key, "UNKNOWN")

        if text:
            entity["entity_id"] = resolver.resolve(text, etype, document_id)
            metadata = resolver.get_entity_metadata(text)
            if metadata.expanded_text:
                entity["expanded_text"] = metadata.expanded_text
            if metadata.is_abbreviation:
                entity["is_abbreviation"] = True

    resolver.log_stats()
    return entities
