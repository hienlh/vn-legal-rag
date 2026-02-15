"""
Database manager for Vietnamese legal documents.

Handles SQLite operations for legal document storage with hierarchical IDs.
"""

from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, joinedload, sessionmaker

from .models import (
    Base,
    LegalAbbreviationModel,
    LegalArticleModel,
    LegalChapterModel,
    LegalClauseModel,
    LegalDocumentModel,
    LegalPointModel,
    LegalSectionModel,
    make_article_id,
    make_chapter_id,
    make_clause_id,
    make_document_id,
    make_point_id,
    make_section_id,
)

DEFAULT_DB_PATH = "data/legal_docs.db"


class LegalDocumentDB:
    """
    SQLite database manager for legal document storage.

    Usage:
        db = LegalDocumentDB("data/legal_docs.db")
        stats = db.count_stats()
        article = db.get_article_by_id("59-2020-QH14:d5")
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize database connection."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_document(self, doc_id: str) -> Optional[LegalDocumentModel]:
        """Get document by ID with all relationships loaded."""
        with self.SessionLocal() as session:
            stmt = (
                select(LegalDocumentModel)
                .options(
                    joinedload(LegalDocumentModel.chapters)
                    .joinedload(LegalChapterModel.articles)
                    .joinedload(LegalArticleModel.clauses)
                    .joinedload(LegalClauseModel.points)
                )
                .where(LegalDocumentModel.id == doc_id)
            )
            result = session.scalar(stmt)
            if result:
                session.expunge(result)
            return result

    def get_document_by_so_hieu(self, so_hieu: str) -> Optional[LegalDocumentModel]:
        """Get document by số hiệu with chapters loaded."""
        with self.SessionLocal() as session:
            stmt = (
                select(LegalDocumentModel)
                .options(joinedload(LegalDocumentModel.chapters))
                .where(LegalDocumentModel.so_hieu == so_hieu)
            )
            result = session.scalar(stmt)
            if result:
                session.expunge(result)
            return result

    def get_article_by_id(self, article_id: str) -> Optional[LegalArticleModel]:
        """
        Get article by hierarchical ID.

        Args:
            article_id: e.g. "59-2020-QH14:d5"
        """
        with self.SessionLocal() as session:
            stmt = (
                select(LegalArticleModel)
                .options(
                    joinedload(LegalArticleModel.clauses)
                    .joinedload(LegalClauseModel.points)
                )
                .where(LegalArticleModel.id == article_id)
            )
            result = session.scalar(stmt)
            if result:
                session.expunge(result)
            return result

    def get_article(
        self, doc_id: str, article_number: int
    ) -> Optional[LegalArticleModel]:
        """Get article by document ID and article number."""
        article_id = make_article_id(doc_id, article_number)
        return self.get_article_by_id(article_id)

    def get_article_by_so_hieu(
        self, so_hieu: str, article_number: int
    ) -> Optional[LegalArticleModel]:
        """Get article by document số hiệu and article number."""
        doc_id = make_document_id(so_hieu)
        return self.get_article(doc_id, article_number)

    def get_clause_by_id(self, clause_id: str) -> Optional[LegalClauseModel]:
        """
        Get clause by hierarchical ID.

        Args:
            clause_id: e.g. "59-2020-QH14:d5:k1"
        """
        with self.SessionLocal() as session:
            stmt = (
                select(LegalClauseModel)
                .options(joinedload(LegalClauseModel.points))
                .where(LegalClauseModel.id == clause_id)
            )
            result = session.scalar(stmt)
            if result:
                session.expunge(result)
            return result

    def get_clause(
        self, article_id: str, clause_number: int
    ) -> Optional[LegalClauseModel]:
        """Get clause by article ID and clause number."""
        clause_id = make_clause_id(article_id, clause_number)
        return self.get_clause_by_id(clause_id)

    def list_documents(self) -> List[LegalDocumentModel]:
        """List all documents without relationships."""
        with self.SessionLocal() as session:
            stmt = select(LegalDocumentModel).order_by(LegalDocumentModel.created_at.desc())
            results = session.scalars(stmt).all()
            for r in results:
                session.expunge(r)
            return list(results)

    def count_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self.SessionLocal() as session:
            return {
                "documents": session.query(LegalDocumentModel).count(),
                "chapters": session.query(LegalChapterModel).count(),
                "sections": session.query(LegalSectionModel).count(),
                "articles": session.query(LegalArticleModel).count(),
                "clauses": session.query(LegalClauseModel).count(),
                "points": session.query(LegalPointModel).count(),
                "abbreviations": session.query(LegalAbbreviationModel).count(),
            }

    def link_to_kg(self, element_id: str, kg_node_id: str, element_type: str) -> bool:
        """
        Link a DB element to a KG node.

        Args:
            element_id: Hierarchical ID of the element
            kg_node_id: ID of the KG node
            element_type: One of 'document', 'chapter', 'section', 'article', 'clause', 'point'
        """
        model_map = {
            "document": LegalDocumentModel,
            "chapter": LegalChapterModel,
            "section": LegalSectionModel,
            "article": LegalArticleModel,
            "clause": LegalClauseModel,
            "point": LegalPointModel,
        }

        model_class = model_map.get(element_type)
        if not model_class:
            raise ValueError(f"Unknown element type: {element_type}")

        with self.SessionLocal() as session:
            element = session.get(model_class, element_id)
            if element:
                element.kg_node_id = kg_node_id
                session.commit()
                return True
            return False

    # Abbreviation methods
    def get_abbreviation(self, abbrev: str) -> Optional[LegalAbbreviationModel]:
        """Get abbreviation by ID."""
        with self.SessionLocal() as session:
            result = session.get(LegalAbbreviationModel, abbrev)
            if result:
                session.expunge(result)
            return result

    def list_abbreviations(
        self, category: Optional[str] = None, min_count: int = 0
    ) -> List[LegalAbbreviationModel]:
        """List abbreviations with optional filters."""
        with self.SessionLocal() as session:
            query = session.query(LegalAbbreviationModel)

            if category:
                query = query.filter(LegalAbbreviationModel.category == category)
            if min_count > 0:
                query = query.filter(LegalAbbreviationModel.corpus_count >= min_count)

            query = query.order_by(LegalAbbreviationModel.corpus_count.desc())
            results = query.all()

            for r in results:
                session.expunge(r)
            return results

    def get_abbreviation_full_form(self, abbrev: str) -> Optional[str]:
        """Get full form of an abbreviation."""
        result = self.get_abbreviation(abbrev)
        return result.full_form if result else None
