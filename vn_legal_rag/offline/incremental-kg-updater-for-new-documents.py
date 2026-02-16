"""
Incremental Update Pipeline for Knowledge Graph.

Updates knowledge graph incrementally when new documents are added,
without rebuilding the entire graph.

Based on LightRAG's incremental update algorithm.

Features:
- Extract entities/relations from new documents
- Deduplicate against existing KG
- Merge new data into existing graph
- Update FAISS indexes incrementally
- Invalidate affected caches

Usage:
    >>> from vn_legal_rag.offline import IncrementalUpdater
    >>> updater = IncrementalUpdater(kg_path, db_path)
    >>> result = updater.update(new_documents)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils import get_logger


@dataclass
class UpdateResult:
    """Result of incremental update operation."""
    success: bool = True
    documents_processed: int = 0
    entities_added: int = 0
    entities_merged: int = 0  # Merged with existing
    relations_added: int = 0
    relations_remapped: int = 0

    # Affected items (for cache invalidation)
    affected_articles: Set[str] = field(default_factory=set)
    affected_entities: Set[str] = field(default_factory=set)

    # Timing
    extract_time_ms: float = 0
    dedup_time_ms: float = 0
    merge_time_ms: float = 0
    index_time_ms: float = 0
    total_time_ms: float = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "documents_processed": self.documents_processed,
            "entities_added": self.entities_added,
            "entities_merged": self.entities_merged,
            "relations_added": self.relations_added,
            "relations_remapped": self.relations_remapped,
            "affected_articles": list(self.affected_articles),
            "timing": {
                "extract_ms": self.extract_time_ms,
                "dedup_ms": self.dedup_time_ms,
                "merge_ms": self.merge_time_ms,
                "index_ms": self.index_time_ms,
                "total_ms": self.total_time_ms,
            },
            "errors": self.errors,
        }


class IncrementalUpdater:
    """
    Incremental KG updater for adding new legal documents.

    This class provides a pipeline for incrementally updating the knowledge
    graph when new documents are added, without requiring a full rebuild.
    """

    def __init__(
        self,
        kg_path: str,
        db_path: str = "data/legal_docs.db",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        dedup_threshold: float = 0.98,
        vector_store_path: Optional[str] = None,
        cache_db: Optional[str] = None,
    ):
        """
        Initialize incremental updater.

        Args:
            kg_path: Path to existing KG JSON file
            db_path: Path to SQLite database
            llm_provider: LLM provider for extraction
            llm_model: LLM model name
            dedup_threshold: Deduplication threshold
            vector_store_path: Path to FAISS vector store
            cache_db: Path to LLM cache database
        """
        self.kg_path = Path(kg_path)
        self.db_path = db_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.dedup_threshold = dedup_threshold
        self.vector_store_path = vector_store_path
        self.cache_db = cache_db

        self.logger = get_logger("incremental_updater")

        # Load existing KG
        self.kg = self._load_kg()

        # Build entity index
        self._entity_index = {
            e.get("id", ""): e
            for e in self.kg.get("entities", [])
        }

    def _load_kg(self) -> Dict[str, Any]:
        """Load existing KG from disk."""
        if not self.kg_path.exists():
            self.logger.warning(f"KG not found at {self.kg_path}, creating new")
            return {"entities": [], "relationships": [], "metadata": {}}

        with open(self.kg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_kg(self):
        """Save updated KG to disk."""
        self.kg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.kg_path, "w", encoding="utf-8") as f:
            json.dump(self.kg, f, ensure_ascii=False, indent=2)

    def update(
        self,
        article_ids: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        show_progress: bool = True,
    ) -> UpdateResult:
        """
        Update KG with new articles.

        Args:
            article_ids: Specific article IDs to process
            document_id: Process all articles from this document
            show_progress: Print progress

        Returns:
            UpdateResult with statistics
        """
        start_time = time.time()
        result = UpdateResult()

        try:
            # 1. Extract entities and relations from new articles
            if show_progress:
                print("[1/5] Extracting entities and relations...")

            extract_start = time.time()
            new_entities, new_relations = self._extract_from_articles(
                article_ids, document_id
            )
            result.extract_time_ms = (time.time() - extract_start) * 1000
            result.documents_processed = len(article_ids or [])

            if not new_entities:
                self.logger.info("No new entities extracted")
                result.total_time_ms = (time.time() - start_time) * 1000
                return result

            # 2. Deduplicate against existing entities
            if show_progress:
                print(f"[2/5] Deduplicating {len(new_entities)} entities...")

            dedup_start = time.time()
            merged_entities, remap = self._deduplicate(new_entities)
            result.dedup_time_ms = (time.time() - dedup_start) * 1000
            result.entities_merged = len(remap)

            # 3. Remap relations
            if remap:
                new_relations = self._remap_relations(new_relations, remap)
                result.relations_remapped = len([
                    r for r in new_relations
                    if r.get("_remapped", False)
                ])

            # 4. Merge into KG
            if show_progress:
                print(f"[3/5] Merging {len(merged_entities)} entities, {len(new_relations)} relations...")

            merge_start = time.time()
            self._merge_into_kg(merged_entities, new_relations, result)
            result.merge_time_ms = (time.time() - merge_start) * 1000

            # 5. Update indexes
            if show_progress:
                print("[4/5] Updating indexes...")

            index_start = time.time()
            self._update_indexes(merged_entities)
            result.index_time_ms = (time.time() - index_start) * 1000

            # 6. Save KG
            if show_progress:
                print("[5/5] Saving KG...")

            self._save_kg()

            # 7. Invalidate caches
            self._invalidate_caches(result.affected_articles)

            result.total_time_ms = (time.time() - start_time) * 1000
            result.success = True

            self.logger.info(
                f"Update complete: +{result.entities_added} entities, "
                f"+{result.relations_added} relations, "
                f"{result.total_time_ms:.0f}ms"
            )

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            self.logger.error(f"Update failed: {e}")

        return result

    def _extract_from_articles(
        self,
        article_ids: Optional[List[str]],
        document_id: Optional[str],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relations from articles using existing extractor."""
        from .database_manager import LegalDocumentDB
        from .unified_entity_relation_extractor import UnifiedLegalExtractor

        # Initialize database and extractor
        db = LegalDocumentDB(self.db_path)
        extractor = UnifiedLegalExtractor(
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
        )

        all_entities = []
        all_relations = []

        with db.SessionLocal() as session:
            from .models import LegalArticleModel, LegalDocumentModel

            # Get articles to process
            if article_ids:
                articles = session.query(LegalArticleModel).filter(
                    LegalArticleModel.id.in_(article_ids)
                ).all()
            elif document_id:
                doc = session.query(LegalDocumentModel).filter(
                    LegalDocumentModel.id == document_id
                ).first()
                if doc:
                    articles = []
                    for chapter in doc.chapters:
                        articles.extend(chapter.articles)
                        for section in chapter.sections:
                            articles.extend(section.articles)
                else:
                    articles = []
            else:
                articles = []

            # Extract from each article
            for article in articles:
                content = article.raw_text or ""
                if not content:
                    continue

                try:
                    result = extractor.extract(content, source_id=article.id)
                    all_entities.extend(result.entities)
                    all_relations.extend(result.relations)
                except Exception as e:
                    self.logger.warning(f"Extraction failed for {article.id}: {e}")

        return all_entities, all_relations

    def _deduplicate(
        self,
        new_entities: List[Dict],
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """Deduplicate new entities against existing KG."""
        from .deduplicator import LegalEntityDeduplicator

        dedup = LegalEntityDeduplicator(threshold=self.dedup_threshold)

        # Get existing entities as list
        existing = list(self._entity_index.values())

        # Deduplicate
        result = dedup.deduplicate(new_entities, existing)

        return result.merged_entities, result.remap

    def _remap_relations(
        self,
        relations: List[Dict],
        remap: Dict[str, str],
    ) -> List[Dict]:
        """Remap relation source/target IDs."""
        from .deduplicator import LegalEntityDeduplicator

        dedup = LegalEntityDeduplicator()
        updated = dedup.remap_relations(relations, remap)

        # Mark remapped relations
        for rel in updated:
            source = rel.get("source_id", rel.get("source", ""))
            target = rel.get("target_id", rel.get("target", ""))
            if source in remap or target in remap:
                rel["_remapped"] = True

        return updated

    def _merge_into_kg(
        self,
        new_entities: List[Dict],
        new_relations: List[Dict],
        result: UpdateResult,
    ):
        """Merge new entities and relations into existing KG."""
        # Add new entities
        for entity in new_entities:
            eid = entity.get("id", "")
            if eid and eid not in self._entity_index:
                self.kg["entities"].append(entity)
                self._entity_index[eid] = entity
                result.entities_added += 1

                # Track affected articles
                source = entity.get("metadata", {}).get("source_id", "")
                if source:
                    result.affected_articles.add(source)
                result.affected_entities.add(eid)

        # Add new relations (check for duplicates)
        existing_relations = set()
        for rel in self.kg.get("relationships", []):
            key = (
                rel.get("source_id", rel.get("source", "")),
                rel.get("target_id", rel.get("target", "")),
                rel.get("type", ""),
            )
            existing_relations.add(key)

        for rel in new_relations:
            key = (
                rel.get("source_id", rel.get("source", "")),
                rel.get("target_id", rel.get("target", "")),
                rel.get("type", ""),
            )
            if key not in existing_relations:
                # Clean up internal field
                rel.pop("_remapped", None)
                self.kg["relationships"].append(rel)
                existing_relations.add(key)
                result.relations_added += 1

        # Update metadata
        self.kg.setdefault("metadata", {})
        self.kg["metadata"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.kg["metadata"]["total_entities"] = len(self.kg["entities"])
        self.kg["metadata"]["total_relationships"] = len(self.kg["relationships"])

    def _update_indexes(self, new_entities: List[Dict]):
        """Update FAISS indexes with new entities."""
        if not self.vector_store_path:
            return

        # This would integrate with existing vector store
        self.logger.info(f"Would update FAISS index with {len(new_entities)} entities")

    def _invalidate_caches(self, affected_articles: Set[str]):
        """Invalidate LLM caches for affected articles."""
        if not self.cache_db or not affected_articles:
            return

        self.logger.info(f"Would invalidate cache for {len(affected_articles)} articles")

    def rollback(self, backup_path: str):
        """Rollback to a backup KG."""
        backup = Path(backup_path)
        if backup.exists():
            with open(backup, "r", encoding="utf-8") as f:
                self.kg = json.load(f)
            self._save_kg()
            self.logger.info(f"Rolled back to {backup_path}")
        else:
            self.logger.error(f"Backup not found: {backup_path}")

    def create_backup(self) -> str:
        """Create backup of current KG."""
        backup_path = self.kg_path.with_suffix(
            f".backup.{int(time.time())}.json"
        )
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(self.kg, f, ensure_ascii=False)
        self.logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
