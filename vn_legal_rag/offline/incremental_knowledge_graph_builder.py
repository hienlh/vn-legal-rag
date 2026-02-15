"""
Incremental Knowledge Graph Builder with Built-in Merge

LightRAG-style real-time entity/relation merging during extraction.
Memory efficient - entities merged in-place, not accumulated then deduped.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..utils.simple_logger import get_logger


def slugify_vietnamese(text: str) -> str:
    """
    Convert Vietnamese text to slug format.

    Example: "Công ty cổ phần" -> "cong-ty-co-phan"
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")

    return text


@dataclass
class MergedEntity:
    """Entity with merged metadata from multiple sources."""

    id: str
    name: str
    entity_type: str
    type_str: str
    description: str = ""
    confidence: float = 0.9
    source_ids: Set[str] = field(default_factory=set)
    document_ids: Set[str] = field(default_factory=set)
    aliases: Set[str] = field(default_factory=set)
    occurrence_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge_from(self, other_entity: Dict[str, Any]) -> None:
        """Merge another entity's data into this one."""
        other_conf = other_entity.get("confidence", 0.9)
        if other_conf > self.confidence:
            self.confidence = other_conf

        if source_id := other_entity.get("source_id", ""):
            self.source_ids.add(source_id)

        if doc_id := other_entity.get("document_id", ""):
            self.document_ids.add(doc_id)

        if other_name := other_entity.get("name", ""):
            if other_name != self.name:
                self.aliases.add(other_name)

        if other_desc := other_entity.get("description", ""):
            if len(other_desc) > len(self.description):
                self.description = other_desc

        self.occurrence_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_str,
            "entity_type": self.entity_type,
            "description": self.description,
            "confidence": self.confidence,
            "metadata": {
                "source_ids": list(self.source_ids),
                "document_ids": list(self.document_ids),
                "aliases": list(self.aliases),
                "occurrence_count": self.occurrence_count,
                **self.metadata,
            },
        }


@dataclass
class MergedRelation:
    """Relation with merged metadata."""

    source_id: str
    target_id: str
    relation_type: str
    predicate: str
    evidence: str = ""
    confidence: float = 0.85
    article_ids: Set[str] = field(default_factory=set)
    occurrence_count: int = 1

    def merge_from(self, other_relation: Dict[str, Any]) -> None:
        """Merge another relation's data."""
        if other_evidence := other_relation.get("evidence", ""):
            if other_evidence not in self.evidence:
                self.evidence = f"{self.evidence} | {other_evidence}" if self.evidence else other_evidence

        if other_conf := other_relation.get("confidence", 0.85):
            if other_conf > self.confidence:
                self.confidence = other_conf

        if source_id := other_relation.get("source_id", ""):
            self.article_ids.add(source_id)

        self.occurrence_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.predicate,
            "predicate": self.predicate,
            "relation_type": self.relation_type,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "metadata": {
                "article_ids": list(self.article_ids),
                "occurrence_count": self.occurrence_count,
            },
        }


@dataclass
class IncrementalKGResult:
    """Result of incremental KG building."""

    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    entity_remap: Dict[str, str]
    stats: Dict[str, int]


class IncrementalKGBuilder:
    """
    Incremental Knowledge Graph Builder with built-in merging.

    Merges entities in real-time as they're added (LightRAG-style).
    Memory efficient: O(unique entities) instead of O(all entities).
    """

    def __init__(
        self,
        merge_relations: bool = True,
        track_provenance: bool = True,
    ):
        self.merge_relations = merge_relations
        self.track_provenance = track_provenance
        self.logger = get_logger("incremental_kg_builder")

        self._entities: Dict[str, MergedEntity] = {}
        self._relations: Dict[str, MergedRelation] = {}
        self._entity_remap: Dict[str, str] = {}

        self._total_entities_added = 0
        self._total_relations_added = 0
        self._documents_processed = 0

    def add_extraction(self, result) -> tuple[int, int]:
        """
        Add extraction result to KG, merging with existing entities.

        Args:
            result: ExtractionResult from UnifiedLegalExtractor

        Returns:
            Tuple of (entities_merged, relations_added)
        """
        entities_before = len(self._entities)
        relations_before = len(self._relations)

        for entity in result.entities:
            self._add_entity(entity, result.source_id, result.document_id)

        for relation in result.relations:
            self._add_relation(relation, result.source_id)

        self._documents_processed += 1

        return self._total_entities_added - entities_before, len(self._relations) - relations_before

    def _add_entity(self, entity: Dict[str, Any], source_id: str, document_id: str) -> str:
        """Add or merge an entity. Returns global entity ID (slug)."""
        name = entity.get("name", "")
        if not name:
            return ""

        slug = slugify_vietnamese(name)
        if not slug:
            return ""

        original_id = entity.get("id", f"{document_id}:{slug}")
        self._entity_remap[original_id] = slug

        entity["source_id"] = source_id
        entity["document_id"] = document_id

        self._total_entities_added += 1

        if slug in self._entities:
            self._entities[slug].merge_from(entity)
        else:
            type_str = entity.get("type", "THUẬT_NGỮ")

            merged = MergedEntity(
                id=slug,
                name=name,
                entity_type=type_str,
                type_str=type_str,
                description=entity.get("description", ""),
                confidence=entity.get("confidence", 0.9),
                source_ids={source_id} if source_id else set(),
                document_ids={document_id} if document_id else set(),
            )
            self._entities[slug] = merged

        return slug

    def _add_relation(self, relation: Dict[str, Any], source_id: str) -> Optional[str]:
        """Add or merge a relation. Returns relation key if added/merged."""
        source_text = relation.get("source", "")
        target_text = relation.get("target", "")
        predicate = relation.get("predicate", "LIÊN_QUAN")

        if not source_text or not target_text:
            return None

        source_slug = slugify_vietnamese(source_text)
        target_slug = slugify_vietnamese(target_text)

        if source_slug == target_slug:
            return None

        if source_slug not in self._entities or target_slug not in self._entities:
            return None

        self._total_relations_added += 1

        relation_key = f"{source_slug}|{predicate}|{target_slug}"
        relation["source_id"] = source_id

        if relation_key in self._relations:
            if self.merge_relations:
                self._relations[relation_key].merge_from(relation)
        else:
            merged = MergedRelation(
                source_id=source_slug,
                target_id=target_slug,
                relation_type=predicate,
                predicate=predicate,
                evidence=relation.get("evidence", ""),
                confidence=relation.get("confidence", 0.85),
                article_ids={source_id} if source_id else set(),
            )
            self._relations[relation_key] = merged

        return relation_key

    def build(self) -> IncrementalKGResult:
        """Build final KG result."""
        entities = [e.to_dict() for e in self._entities.values()]
        relations = [r.to_dict() for r in self._relations.values()]

        stats = {
            "total_entities_added": self._total_entities_added,
            "unique_entities": len(entities),
            "entities_merged": self._total_entities_added - len(entities),
            "total_relations_added": self._total_relations_added,
            "unique_relations": len(relations),
            "documents_processed": self._documents_processed,
        }

        self.logger.info(
            f"Built KG: {stats['unique_entities']} entities ({stats['entities_merged']} merged), "
            f"{stats['unique_relations']} relations"
        )

        return IncrementalKGResult(
            entities=entities,
            relations=relations,
            entity_remap=self._entity_remap.copy(),
            stats=stats,
        )

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return {
            "total_entities_added": self._total_entities_added,
            "unique_entities": len(self._entities),
            "entities_merged": self._total_entities_added - len(self._entities),
            "total_relations_added": self._total_relations_added,
            "unique_relations": len(self._relations),
            "documents_processed": self._documents_processed,
        }
