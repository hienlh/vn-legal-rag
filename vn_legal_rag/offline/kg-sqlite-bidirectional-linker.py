"""
Bidirectional KG ↔ SQLite Linker for Vietnamese Legal RAG.

Links Knowledge Graph nodes to SQLite database elements and vice versa.
Enables context retrieval for RAG and provenance tracking.

Features:
- KG nodes ↔ Articles/Clauses/Points in SQLite
- Cross-references ↔ KG edges synchronization
- Context retrieval for RAG pipeline
- Entity provenance tracking
- Bidirectional navigation helpers

Usage:
    >>> from vn_legal_rag.offline import KGSQLiteLinker
    >>> linker = KGSQLiteLinker(kg_path, db_path)
    >>> context = linker.get_context_for_entity("cong_ty_tnhh")
    >>> articles = linker.get_articles_for_kg_node("cong_ty_tnhh")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils import get_logger


@dataclass
class EntityContext:
    """Context retrieved for an entity from linked sources."""

    entity_id: str
    entity_name: str
    entity_type: Optional[str] = None

    # Linked articles
    article_ids: List[str] = field(default_factory=list)
    article_texts: List[str] = field(default_factory=list)

    # Related entities from KG
    related_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-references
    crossrefs: List[Dict[str, Any]] = field(default_factory=list)

    def to_rag_context(self, max_chars: int = 4000) -> str:
        """Format as context string for RAG."""
        parts = []

        # Entity info
        parts.append(f"Entity: {self.entity_name} ({self.entity_type or 'unknown'})")

        # Article texts (truncated)
        total_chars = 0
        for i, text in enumerate(self.article_texts):
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    parts.append(f"\n[Article {i+1}]: {text[:remaining]}...")
                break
            parts.append(f"\n[Article {i+1}]: {text}")
            total_chars += len(text)

        # Related entities
        if self.related_entities:
            related = ", ".join(
                e.get("name", e.get("id", "?"))[:30]
                for e in self.related_entities[:5]
            )
            parts.append(f"\n[Related]: {related}")

        return "\n".join(parts)


@dataclass
class LinkResult:
    """Result of linking operation."""

    success: bool = True
    entities_linked: int = 0
    relations_synced: int = 0
    crossrefs_added: int = 0
    errors: List[str] = field(default_factory=list)


class KGSQLiteLinker:
    """
    Bidirectional linker between KG and SQLite database.

    Provides navigation between:
    - KG entity nodes ↔ SQLite articles/clauses
    - KG relationship edges ↔ SQLite cross-references
    """

    def __init__(
        self,
        kg_path: str,
        db_path: str = "data/legal_docs.db",
    ):
        """
        Initialize linker.

        Args:
            kg_path: Path to KG JSON file
            db_path: Path to SQLite database
        """
        self.kg_path = Path(kg_path)
        self.db_path = db_path
        self.logger = get_logger("kg_sqlite_linker")

        # Load KG
        self.kg = self._load_kg()

        # Build indexes for fast lookup
        self._entity_by_id: Dict[str, Dict] = {}
        self._entity_by_source: Dict[str, List[str]] = {}  # source_id -> [entity_ids]
        self._relations_by_source: Dict[str, List[Dict]] = {}
        self._relations_by_target: Dict[str, List[Dict]] = {}

        self._build_indexes()

    def _load_kg(self) -> Dict[str, Any]:
        """Load KG from disk."""
        if not self.kg_path.exists():
            self.logger.warning(f"KG not found at {self.kg_path}")
            return {"entities": [], "relationships": []}

        with open(self.kg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_indexes(self):
        """Build lookup indexes from KG data."""
        # Index entities
        for entity in self.kg.get("entities", []):
            eid = entity.get("id", "")
            if eid:
                self._entity_by_id[eid] = entity

                # Index by source article
                source_id = entity.get("metadata", {}).get("source_id", "")
                if source_id:
                    if source_id not in self._entity_by_source:
                        self._entity_by_source[source_id] = []
                    self._entity_by_source[source_id].append(eid)

        # Index relations
        for rel in self.kg.get("relationships", []):
            source = rel.get("source_id", rel.get("source", ""))
            target = rel.get("target_id", rel.get("target", ""))

            if source:
                if source not in self._relations_by_source:
                    self._relations_by_source[source] = []
                self._relations_by_source[source].append(rel)

            if target:
                if target not in self._relations_by_target:
                    self._relations_by_target[target] = []
                self._relations_by_target[target].append(rel)

        self.logger.info(
            f"Indexed {len(self._entity_by_id)} entities, "
            f"{len(self._entity_by_source)} source articles"
        )

    # ========== KG → SQLite Navigation ==========

    def get_articles_for_entity(
        self,
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get SQLite articles linked to a KG entity.

        Args:
            entity_id: KG entity ID

        Returns:
            List of article dicts with id, title, content
        """
        from .database_manager import LegalDocumentDB
        from .models import LegalArticleModel

        entity = self._entity_by_id.get(entity_id)
        if not entity:
            return []

        # Get source articles from entity metadata
        source_ids = set()
        source_id = entity.get("metadata", {}).get("source_id", "")
        if source_id:
            source_ids.add(source_id)

        # Also check mentions
        mentions = entity.get("metadata", {}).get("mentions", [])
        for mention in mentions:
            if isinstance(mention, dict):
                src = mention.get("source_id", "")
                if src:
                    source_ids.add(src)

        if not source_ids:
            return []

        # Fetch from database
        db = LegalDocumentDB(self.db_path)
        articles = []

        with db.SessionLocal() as session:
            for article_id in source_ids:
                article = session.query(LegalArticleModel).filter(
                    LegalArticleModel.id == article_id
                ).first()
                if article:
                    articles.append({
                        "id": article.id,
                        "number": article.article_number,
                        "title": article.title,
                        "content": article.raw_text,
                        "document_id": article.id.split(":")[0] if ":" in article.id else None,
                    })

        return articles

    def get_related_entities(
        self,
        entity_id: str,
        max_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get related entities from KG.

        Args:
            entity_id: Starting entity ID
            max_depth: Max hops in graph (1 = direct neighbors)

        Returns:
            List of related entity dicts
        """
        related = []
        visited = {entity_id}

        # Get outgoing relations
        for rel in self._relations_by_source.get(entity_id, []):
            target = rel.get("target_id", rel.get("target", ""))
            if target and target not in visited:
                target_entity = self._entity_by_id.get(target, {})
                related.append({
                    "id": target,
                    "name": target_entity.get("name", target),
                    "type": target_entity.get("type"),
                    "relation": rel.get("type", "related"),
                    "direction": "outgoing",
                })
                visited.add(target)

        # Get incoming relations
        for rel in self._relations_by_target.get(entity_id, []):
            source = rel.get("source_id", rel.get("source", ""))
            if source and source not in visited:
                source_entity = self._entity_by_id.get(source, {})
                related.append({
                    "id": source,
                    "name": source_entity.get("name", source),
                    "type": source_entity.get("type"),
                    "relation": rel.get("type", "related"),
                    "direction": "incoming",
                })
                visited.add(source)

        return related

    def get_context_for_entity(
        self,
        entity_id: str,
        include_related: bool = True,
        include_crossrefs: bool = True,
    ) -> EntityContext:
        """
        Get full context for an entity (for RAG).

        Args:
            entity_id: KG entity ID
            include_related: Include related entities
            include_crossrefs: Include cross-references

        Returns:
            EntityContext with all linked data
        """
        entity = self._entity_by_id.get(entity_id, {})

        context = EntityContext(
            entity_id=entity_id,
            entity_name=entity.get("name", entity_id),
            entity_type=entity.get("type"),
        )

        # Get linked articles
        articles = self.get_articles_for_entity(entity_id)
        context.article_ids = [a["id"] for a in articles]
        context.article_texts = [a.get("content", "")[:2000] for a in articles]

        # Get related entities
        if include_related:
            context.related_entities = self.get_related_entities(entity_id)

        # Get cross-references
        if include_crossrefs:
            context.crossrefs = self.get_crossrefs_for_articles(context.article_ids)

        return context

    # ========== SQLite → KG Navigation ==========

    def get_entities_for_article(
        self,
        article_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get KG entities mentioned in an article.

        Args:
            article_id: SQLite article ID

        Returns:
            List of entity dicts
        """
        entity_ids = self._entity_by_source.get(article_id, [])
        return [
            self._entity_by_id[eid]
            for eid in entity_ids
            if eid in self._entity_by_id
        ]

    def get_kg_subgraph_for_article(
        self,
        article_id: str,
    ) -> Dict[str, Any]:
        """
        Get KG subgraph related to an article.

        Args:
            article_id: SQLite article ID

        Returns:
            Dict with entities and relationships
        """
        entities = self.get_entities_for_article(article_id)
        entity_ids = {e.get("id") for e in entities}

        # Get relations between these entities
        relations = []
        for eid in entity_ids:
            for rel in self._relations_by_source.get(eid, []):
                target = rel.get("target_id", rel.get("target", ""))
                if target in entity_ids:
                    relations.append(rel)

        return {
            "entities": entities,
            "relationships": relations,
            "source_article": article_id,
        }

    # ========== Cross-reference Sync ==========

    def get_crossrefs_for_articles(
        self,
        article_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Get cross-references from database for articles."""
        if not article_ids:
            return []

        from .database_manager import LegalDocumentDB
        from .models import LegalCrossReferenceModel

        db = LegalDocumentDB(self.db_path)
        crossrefs = []

        with db.SessionLocal() as session:
            refs = session.query(LegalCrossReferenceModel).filter(
                LegalCrossReferenceModel.source_article_id.in_(article_ids)
            ).all()

            for ref in refs:
                crossrefs.append({
                    "id": ref.id,
                    "source": ref.source_article_id,
                    "target": ref.target_article_id,
                    "type": ref.reference_type,
                    "text": ref.reference_text,
                })

        return crossrefs

    def sync_crossrefs_to_kg(self) -> LinkResult:
        """
        Sync cross-references from SQLite to KG edges.

        Creates KG relationship edges for each cross-reference.
        """
        from .database_manager import LegalDocumentDB
        from .models import LegalCrossReferenceModel

        result = LinkResult()
        db = LegalDocumentDB(self.db_path)

        # Get existing KG relation keys for dedup
        existing_rels = set()
        for rel in self.kg.get("relationships", []):
            key = (
                rel.get("source_id", rel.get("source", "")),
                rel.get("target_id", rel.get("target", "")),
                rel.get("type", ""),
            )
            existing_rels.add(key)

        # Load crossrefs from database
        with db.SessionLocal() as session:
            crossrefs = session.query(LegalCrossReferenceModel).all()

            for xref in crossrefs:
                # Map crossref to KG edge
                rel_type = self._map_crossref_type(xref.reference_type)

                key = (xref.source_article_id, xref.target_article_id, rel_type)
                if key in existing_rels:
                    continue

                # Add new relation to KG
                new_rel = {
                    "source_id": xref.source_article_id,
                    "target_id": xref.target_article_id,
                    "type": rel_type,
                    "metadata": {
                        "crossref_id": xref.id,
                        "reference_text": xref.reference_text,
                        "confidence": xref.confidence or 0.8,
                    },
                }
                self.kg["relationships"].append(new_rel)
                existing_rels.add(key)
                result.crossrefs_added += 1

        # Save updated KG
        if result.crossrefs_added > 0:
            self._save_kg()
            self._build_indexes()  # Rebuild indexes

        result.relations_synced = result.crossrefs_added
        self.logger.info(f"Synced {result.crossrefs_added} crossrefs to KG")
        return result

    def _map_crossref_type(self, crossref_type: str) -> str:
        """Map crossref type to KG relation type."""
        mapping = {
            "references": "THAM_CHIẾU",
            "amends": "SỬA_ĐỔI",
            "replaces": "THAY_THẾ",
            "supplements": "BỔ_SUNG",
            "implements": "HƯỚNG_DẪN",
            "invalidates": "HẾT_HIỆU_LỰC",
        }
        return mapping.get(crossref_type, "LIÊN_QUAN")

    def _save_kg(self):
        """Save updated KG to disk."""
        self.kg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.kg_path, "w", encoding="utf-8") as f:
            json.dump(self.kg, f, ensure_ascii=False, indent=2)

    # ========== Utility Methods ==========

    def find_entity_by_name(
        self,
        name: str,
        fuzzy: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Find entity by name (exact or fuzzy match)."""
        name_lower = name.lower()

        for entity in self.kg.get("entities", []):
            entity_name = entity.get("name", "").lower()
            if fuzzy:
                if name_lower in entity_name or entity_name in name_lower:
                    return entity
            else:
                if entity_name == name_lower:
                    return entity

        return None

    def get_entity_provenance(
        self,
        entity_id: str,
    ) -> Dict[str, Any]:
        """
        Get full provenance for an entity.

        Returns document/article sources and extraction metadata.
        """
        entity = self._entity_by_id.get(entity_id, {})
        metadata = entity.get("metadata", {})

        articles = self.get_articles_for_entity(entity_id)

        return {
            "entity_id": entity_id,
            "entity_name": entity.get("name"),
            "source_id": metadata.get("source_id"),
            "extraction_method": metadata.get("extraction_method", "llm"),
            "confidence": metadata.get("confidence", 0.8),
            "articles": [
                {"id": a["id"], "title": a.get("title")}
                for a in articles
            ],
            "mentions_count": len(metadata.get("mentions", [])),
        }

    def validate_links(self) -> Dict[str, Any]:
        """
        Validate KG-SQLite links.

        Returns statistics and any broken links.
        """
        from .database_manager import LegalDocumentDB
        from .models import LegalArticleModel

        db = LegalDocumentDB(self.db_path)
        stats = {
            "total_entities": len(self._entity_by_id),
            "entities_with_source": 0,
            "broken_source_links": [],
            "total_relations": len(self.kg.get("relationships", [])),
        }

        with db.SessionLocal() as session:
            for eid, entity in self._entity_by_id.items():
                source_id = entity.get("metadata", {}).get("source_id", "")
                if source_id:
                    stats["entities_with_source"] += 1

                    # Check if article exists
                    exists = session.query(LegalArticleModel).filter(
                        LegalArticleModel.id == source_id
                    ).first()

                    if not exists:
                        stats["broken_source_links"].append({
                            "entity_id": eid,
                            "source_id": source_id,
                        })

        stats["valid_links"] = (
            stats["entities_with_source"] - len(stats["broken_source_links"])
        )
        return stats


def create_linker(
    kg_path: str,
    db_path: str = "data/legal_docs.db",
) -> KGSQLiteLinker:
    """Convenience function to create linker."""
    return KGSQLiteLinker(kg_path, db_path)
