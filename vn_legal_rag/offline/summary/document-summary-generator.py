"""
Document summary generator - Hybrid approach for Loop 0 document selection.

Generates document-level summaries using 3-tier strategy:
1. Structure-based: Extract from Điều 1-2 (scope, subjects) - FREE
2. Chapter aggregation: Aggregate from chapter_summaries - FREE
3. LLM enhancement: Generate domain keywords - COSTS TOKENS (optional)

Output is used in Loop 0 of tree traversal to select relevant documents
before chapter selection.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ...utils import create_llm_provider, get_logger
from ..database_manager import LegalDocumentDB
from ..models import LegalArticleModel, LegalDocumentModel


@dataclass
class DocumentSummary:
    """Summary of a document for Loop 0 selection."""
    doc_id: str
    so_hieu: str
    loai_van_ban: str
    ten_van_ban: str

    # Core fields for Loop 0
    scope: str  # From Điều 1 (Phạm vi điều chỉnh)
    subjects: str  # From Điều 2 (Đối tượng áp dụng)
    key_terms: List[str]  # From abbreviations
    domain_keywords: List[str]  # Aggregated from chapters or LLM

    # Stats
    num_chapters: int = 0
    num_articles: int = 0

    # Compact representation for Loop 0 prompt
    def to_loop0_format(self) -> Dict[str, Any]:
        """Convert to compact format for Loop 0 prompt."""
        domain = ", ".join(self.domain_keywords[:15])  # Top 15 keywords
        return {
            "doc_id": self.doc_id,
            "name": self.ten_van_ban,
            "loai": self.loai_van_ban,
            "domain": domain,
            "scope_preview": self.scope[:200] if self.scope else "",
        }


CHECKPOINT_FILENAME = "document_summaries_checkpoint.json"


class DocumentSummaryCheckpoint:
    """Checkpoint for document summary generation with resume capability."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_path = output_dir / CHECKPOINT_FILENAME
        self.processed_docs: Set[str] = set()
        self.summaries: Dict[str, DocumentSummary] = {}
        self.stats = {"successful": 0, "failed": 0}

    def exists(self) -> bool:
        return self.checkpoint_path.exists()

    def load(self) -> bool:
        """Load checkpoint from disk."""
        if not self.exists():
            return False

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.processed_docs = set(data.get("processed_docs", []))
            self.stats = data.get("stats", self.stats)

            for s in data.get("summaries", []):
                summary = DocumentSummary(
                    doc_id=s["doc_id"],
                    so_hieu=s["so_hieu"],
                    loai_van_ban=s["loai_van_ban"],
                    ten_van_ban=s["ten_van_ban"],
                    scope=s.get("scope", ""),
                    subjects=s.get("subjects", ""),
                    key_terms=s.get("key_terms", []),
                    domain_keywords=s.get("domain_keywords", []),
                    num_chapters=s.get("num_chapters", 0),
                    num_articles=s.get("num_articles", 0),
                )
                self.summaries[summary.doc_id] = summary

            return True
        except Exception as e:
            print(f"  [Resume] Failed to load document checkpoint: {e}")
            return False

    def save(self) -> None:
        """Save checkpoint atomically."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "processed_docs": list(self.processed_docs),
            "summaries": [
                {
                    "doc_id": s.doc_id,
                    "so_hieu": s.so_hieu,
                    "loai_van_ban": s.loai_van_ban,
                    "ten_van_ban": s.ten_van_ban,
                    "scope": s.scope,
                    "subjects": s.subjects,
                    "key_terms": s.key_terms,
                    "domain_keywords": s.domain_keywords,
                    "num_chapters": s.num_chapters,
                    "num_articles": s.num_articles,
                }
                for s in self.summaries.values()
            ],
            "stats": self.stats,
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.rename(self.checkpoint_path)

    def add_summary(self, summary: DocumentSummary) -> None:
        """Add summary and save checkpoint."""
        self.processed_docs.add(summary.doc_id)
        self.summaries[summary.doc_id] = summary
        self.stats["successful"] += 1
        self.save()

    def add_failure(self, doc_id: str) -> None:
        """Record failed extraction."""
        self.processed_docs.add(doc_id)
        self.stats["failed"] += 1
        self.save()

    def is_processed(self, doc_id: str) -> bool:
        return doc_id in self.processed_docs

    def delete(self) -> None:
        """Delete checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


class DocumentSummaryGenerator:
    """
    Generate document summaries using hybrid approach for Loop 0.

    Three-tier strategy:
    1. Structure-based: Extract scope/subjects from Điều 1-2 (free)
    2. Chapter aggregation: Aggregate keywords from chapter_summaries (free)
    3. LLM enhancement: Generate domain keywords for key docs (costs tokens)
    """

    def __init__(
        self,
        db: LegalDocumentDB,
        output_dir: Path,
        chapter_summaries: Optional[Dict[str, Any]] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        use_llm: bool = False,  # Only enable for key documents
        resume: bool = True,
        use_cache: bool = True,
        cache_db_path: str = "data/llm_cache.db",
    ):
        """
        Initialize document summary generator.

        Args:
            db: LegalDocumentDB instance
            output_dir: Output directory for summaries
            chapter_summaries: Optional pre-loaded chapter summaries dict
            llm_provider: LLM provider name
            llm_model: LLM model name
            use_llm: Whether to use LLM for domain keyword extraction
            resume: Whether to resume from checkpoint
            use_cache: Whether to cache LLM responses
            cache_db_path: Path to LLM cache database
        """
        self.db = db
        self.output_dir = Path(output_dir)
        self.chapter_summaries = chapter_summaries or {}
        self.use_llm = use_llm
        self.resume = resume
        self.logger = get_logger("document_summary_generator")
        self.checkpoint = DocumentSummaryCheckpoint(self.output_dir)

        if use_llm:
            self.llm_provider = create_llm_provider(
                llm_provider,
                model=llm_model,
                use_cache=use_cache,
                cache_db_path=cache_db_path,
            )
        else:
            self.llm_provider = None

    def generate_all(self, doc_ids: Optional[List[str]] = None) -> Dict[str, DocumentSummary]:
        """
        Generate summaries for all documents.

        Args:
            doc_ids: Specific document IDs, or None for all documents

        Returns:
            Dict mapping doc_id to DocumentSummary
        """
        # Try to resume from checkpoint
        if self.resume and self.checkpoint.exists():
            if self.checkpoint.load():
                self.logger.info(
                    f"Resumed from checkpoint: {len(self.checkpoint.summaries)} summaries loaded"
                )

        with self.db.SessionLocal() as session:
            # Get documents
            if doc_ids:
                docs = session.query(LegalDocumentModel).filter(
                    LegalDocumentModel.id.in_(doc_ids)
                ).all()
            else:
                docs = session.query(LegalDocumentModel).all()

            total_docs = len(docs)

            for i, doc in enumerate(docs, 1):
                # Skip if already processed
                if self.checkpoint.is_processed(doc.id):
                    self.logger.debug(f"Skipping {doc.id} (already processed)")
                    continue

                try:
                    summary = self._generate_document_summary(doc, session)
                    self.checkpoint.add_summary(summary)
                    self.logger.info(
                        f"[{i}/{total_docs}] Generated: {doc.so_hieu} "
                        f"({summary.num_chapters} chapters, {len(summary.domain_keywords)} keywords)"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to generate summary for {doc.id}: {e}")
                    self.checkpoint.add_failure(doc.id)

        self.logger.info(
            f"Completed: {self.checkpoint.stats['successful']} success, "
            f"{self.checkpoint.stats['failed']} failed"
        )

        return self.checkpoint.summaries

    def _generate_document_summary(
        self, doc: LegalDocumentModel, session
    ) -> DocumentSummary:
        """
        Generate summary for a single document using hybrid approach.

        Tier 1: Structure-based extraction (FREE)
        Tier 2: Chapter aggregation (FREE)
        Tier 3: LLM enhancement (OPTIONAL)
        """
        # === TIER 1: Structure-based extraction ===
        scope = ""
        subjects = ""
        key_terms = []

        # Extract scope from Điều 1 (Phạm vi điều chỉnh)
        article_1 = self._get_article_by_number(doc, 1)
        if article_1 and article_1.raw_text:
            scope = self._extract_scope_text(article_1.raw_text)

        # Extract subjects from Điều 2 (Đối tượng áp dụng)
        article_2 = self._get_article_by_number(doc, 2)
        if article_2 and article_2.raw_text:
            subjects = self._extract_subjects_text(article_2.raw_text)

        # Extract key terms from abbreviations
        if hasattr(doc, 'abbreviations') and doc.abbreviations:
            key_terms = [abbr.full_form for abbr in doc.abbreviations if abbr.full_form]

        # === TIER 2: Chapter aggregation ===
        domain_keywords = self._aggregate_chapter_keywords(doc.id)

        # Add chapter names as domain keywords
        chapter_names = [ch.title for ch in doc.chapters if ch.title]
        domain_keywords.extend(chapter_names)

        # === TIER 3: LLM enhancement (optional) ===
        if self.use_llm and self.llm_provider:
            llm_keywords = self._generate_llm_keywords(doc, scope, subjects)
            domain_keywords.extend(llm_keywords)

        # Deduplicate and clean keywords
        domain_keywords = self._deduplicate_keywords(domain_keywords)

        # Count articles
        num_articles = sum(
            len(ch.articles) + sum(len(sec.articles) for sec in ch.sections)
            for ch in doc.chapters
        )

        return DocumentSummary(
            doc_id=doc.id,
            so_hieu=doc.so_hieu or doc.id,
            loai_van_ban=doc.loai_van_ban or "Văn bản",
            ten_van_ban=doc.title or doc.so_hieu or doc.id,
            scope=scope,
            subjects=subjects,
            key_terms=key_terms[:20],  # Limit to 20
            domain_keywords=domain_keywords[:50],  # Limit to 50
            num_chapters=len(doc.chapters),
            num_articles=num_articles,
        )

    def _get_article_by_number(
        self, doc: LegalDocumentModel, article_number: int
    ) -> Optional[LegalArticleModel]:
        """Get article by number from document."""
        for chapter in doc.chapters:
            for article in chapter.articles:
                if article.article_number == article_number:
                    return article
            for section in chapter.sections:
                for article in section.articles:
                    if article.article_number == article_number:
                        return article
        return None

    def _extract_scope_text(self, raw_text: str) -> str:
        """Extract clean scope text from Điều 1."""
        # Remove article number prefix
        text = raw_text.strip()

        # Common patterns to remove
        prefixes = ["Điều 1.", "Điều 1:", "1.", "Phạm vi điều chỉnh"]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Limit length
        if len(text) > 500:
            text = text[:500].rsplit(".", 1)[0] + "."

        return text

    def _extract_subjects_text(self, raw_text: str) -> str:
        """Extract clean subjects text from Điều 2."""
        text = raw_text.strip()

        prefixes = ["Điều 2.", "Điều 2:", "2.", "Đối tượng áp dụng"]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        if len(text) > 300:
            text = text[:300].rsplit(".", 1)[0] + "."

        return text

    def _aggregate_chapter_keywords(self, doc_id: str) -> List[str]:
        """Aggregate keywords from chapter summaries."""
        keywords = []

        for chapter_id, summary in self.chapter_summaries.items():
            if chapter_id.startswith(doc_id):
                kw_str = summary.get("keywords", "")
                if kw_str:
                    # Split by comma and clean
                    kws = [k.strip() for k in kw_str.split(",")]
                    keywords.extend(kws)

        return keywords

    def _generate_llm_keywords(
        self, doc: LegalDocumentModel, scope: str, subjects: str
    ) -> List[str]:
        """Generate domain keywords using LLM."""
        if not self.llm_provider:
            return []

        # Build context
        chapter_list = "\n".join(
            f"- {ch.title}" for ch in doc.chapters if ch.title
        )

        prompt = f"""Phân tích văn bản pháp luật và liệt kê 20-30 từ khóa tìm kiếm.

VĂN BẢN: {doc.title} ({doc.so_hieu})
LOẠI: {doc.loai_van_ban}

PHẠM VI ĐIỀU CHỈNH:
{scope[:400] if scope else 'Không có'}

ĐỐI TƯỢNG ÁP DỤNG:
{subjects[:300] if subjects else 'Không có'}

CÁC CHƯƠNG:
{chapter_list}

Liệt kê từ khóa ngắn gọn, cách nhau bằng dấu phẩy. Bao gồm:
1. Thuật ngữ chính (VD: công ty, doanh nghiệp)
2. Viết tắt phổ biến (VD: TNHH, CTCP)
3. Thủ tục/hành động (VD: đăng ký, thay đổi)
4. Đối tượng áp dụng (VD: nhà đầu tư, cổ đông)

Chỉ trả lời danh sách từ khóa, không giải thích."""

        try:
            response = self.llm_provider.generate(prompt)
            keywords = [k.strip() for k in response.strip().split(",")]
            return keywords[:30]
        except Exception as e:
            self.logger.warning(f"LLM keyword generation failed: {e}")
            return []

    def _deduplicate_keywords(self, keywords: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        result = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in seen and len(kw_lower) > 1:
                seen.add(kw_lower)
                result.append(kw.strip())
        return result

    def export_summaries(self, path: Optional[Path] = None) -> Path:
        """Export summaries to JSON file."""
        if path is None:
            path = self.output_dir / "document_summaries.json"

        data = {
            doc_id: {
                "doc_id": s.doc_id,
                "so_hieu": s.so_hieu,
                "loai_van_ban": s.loai_van_ban,
                "ten_van_ban": s.ten_van_ban,
                "scope": s.scope,
                "subjects": s.subjects,
                "key_terms": s.key_terms,
                "domain_keywords": s.domain_keywords,
                "num_chapters": s.num_chapters,
                "num_articles": s.num_articles,
            }
            for doc_id, s in self.checkpoint.summaries.items()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported {len(data)} document summaries to {path}")
        return path

    def export_loop0_format(self, path: Optional[Path] = None) -> Path:
        """Export compact format optimized for Loop 0 prompt."""
        if path is None:
            path = self.output_dir / "document_summaries_loop0.json"

        data = [
            s.to_loop0_format()
            for s in self.checkpoint.summaries.values()
        ]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported Loop 0 format ({len(data)} docs) to {path}")
        return path

    @staticmethod
    def load_summaries(path: Path) -> Dict[str, DocumentSummary]:
        """Load summaries from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            doc_id: DocumentSummary(
                doc_id=s["doc_id"],
                so_hieu=s["so_hieu"],
                loai_van_ban=s["loai_van_ban"],
                ten_van_ban=s["ten_van_ban"],
                scope=s.get("scope", ""),
                subjects=s.get("subjects", ""),
                key_terms=s.get("key_terms", []),
                domain_keywords=s.get("domain_keywords", []),
                num_chapters=s.get("num_chapters", 0),
                num_articles=s.get("num_articles", 0),
            )
            for doc_id, s in data.items()
        }
