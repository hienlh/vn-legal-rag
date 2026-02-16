"""
Article summary generator - LLM-based keyword extraction for tree navigation.

Runs during offline phase to generate article descriptions that help
LLM-guided tree traversal select the right article in loop 2.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

from ...utils import create_llm_provider, get_logger
from ..database_manager import LegalDocumentDB
from ..models import LegalArticleModel, LegalDocumentModel


@dataclass
class ArticleSummary:
    """Summary of an article for LLM navigation."""
    article_id: str
    article_number: str
    article_title: str
    keywords: str  # LLM-generated keywords from content
    description: str  # Combined title + keywords


CHECKPOINT_FILENAME = "article_summaries_checkpoint.json"


class ArticleSummaryCheckpoint:
    """Checkpoint for article summary generation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_path = output_dir / CHECKPOINT_FILENAME
        self.processed_articles: Set[str] = set()
        self.summaries: Dict[str, ArticleSummary] = {}
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

            self.processed_articles = set(data.get("processed_articles", []))
            self.stats = data.get("stats", self.stats)

            for s in data.get("summaries", []):
                summary = ArticleSummary(
                    article_id=s["article_id"],
                    article_number=s["article_number"],
                    article_title=s["article_title"],
                    keywords=s["keywords"],
                    description=s["description"],
                )
                self.summaries[summary.article_id] = summary

            return True
        except Exception as e:
            print(f"  [Resume] Failed to load article checkpoint: {e}")
            return False

    def save(self) -> None:
        """Save checkpoint atomically."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "processed_articles": list(self.processed_articles),
            "summaries": [
                {
                    "article_id": s.article_id,
                    "article_number": s.article_number,
                    "article_title": s.article_title,
                    "keywords": s.keywords,
                    "description": s.description,
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

    def add_summary(self, summary: ArticleSummary) -> None:
        """Add summary and save checkpoint."""
        self.processed_articles.add(summary.article_id)
        self.summaries[summary.article_id] = summary
        self.stats["successful"] += 1
        self.save()

    def add_failure(self, article_id: str) -> None:
        """Record failed extraction."""
        self.processed_articles.add(article_id)
        self.stats["failed"] += 1
        self.save()

    def is_processed(self, article_id: str) -> bool:
        return article_id in self.processed_articles


class ArticleSummaryGenerator:
    """
    Generate article summaries using LLM for better tree navigation.

    Extracts keywords from article content to help LLM select the right
    article during tree traversal loop 2.
    """

    def __init__(
        self,
        db: LegalDocumentDB,
        output_dir: Path,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-3-5-haiku-20241022",
        llm_base_url: Optional[str] = None,
        resume: bool = True,
        use_cache: bool = True,
        cache_db_path: str = "data/llm_cache.db",
    ):
        self.db = db
        self.output_dir = Path(output_dir)
        provider_kwargs = {
            "model": llm_model,
            "use_cache": use_cache,
            "cache_db_path": cache_db_path,
        }
        if llm_base_url:
            provider_kwargs["base_url"] = llm_base_url
        self.llm_provider = create_llm_provider(llm_provider, **provider_kwargs)
        self.logger = get_logger("article_summary_generator")
        self.resume = resume
        self.checkpoint = ArticleSummaryCheckpoint(self.output_dir)

    def generate_all(self, doc_id: Optional[str] = None) -> Dict[str, ArticleSummary]:
        """
        Generate summaries for all articles in document(s).

        Args:
            doc_id: Specific document ID, or None for all documents

        Returns:
            Dict mapping article_id to ArticleSummary
        """
        if self.resume and self.checkpoint.exists():
            if self.checkpoint.load():
                self.logger.info(f"Resumed from checkpoint: {len(self.checkpoint.summaries)} summaries loaded")

        with self.db.SessionLocal() as session:
            if doc_id:
                docs = session.query(LegalDocumentModel).filter(
                    LegalDocumentModel.id == doc_id
                ).all()
            else:
                docs = session.query(LegalDocumentModel).all()

            # Count total articles (avoid double-counting articles in sections)
            total_articles = 0
            for doc in docs:
                for chapter in doc.chapters:
                    # Only count articles NOT in a section
                    total_articles += len([a for a in chapter.articles if a.section_id is None])
                    for section in chapter.sections:
                        total_articles += len(section.articles)

            processed = 0
            self.logger.info(f"Found {len(docs)} documents, {total_articles} articles total")

            for doc in docs:
                self.logger.info(f"Document: {doc.so_hieu} ({doc.title[:50] if doc.title else 'N/A'}...)")

                for chapter in sorted(doc.chapters, key=lambda c: c.position):
                    # Direct articles (not in any section)
                    for article in sorted(
                        [a for a in chapter.articles if a.section_id is None],
                        key=lambda a: a.position
                    ):
                        processed += 1
                        self._process_article(article, processed, total_articles)

                    # Articles in sections
                    for section in chapter.sections:
                        for article in sorted(section.articles, key=lambda a: a.position):
                            processed += 1
                            self._process_article(article, processed, total_articles)

        self.logger.info(f"Completed: {self.checkpoint.stats['successful']} success, {self.checkpoint.stats['failed']} failed")

        return self.checkpoint.summaries

    def _process_article(self, article: LegalArticleModel, processed: int, total: int):
        """Process a single article."""
        if self.checkpoint.is_processed(article.id):
            return

        try:
            summary = self._generate_article_summary(article)
            self.checkpoint.add_summary(summary)
            self.logger.debug(f"[{processed}/{total}] {article.id} - Điều {article.article_number}: {article.title or 'N/A'}")
        except Exception as e:
            self.logger.error(f"[{processed}/{total}] FAILED {article.id}: {e}")
            self.checkpoint.add_failure(article.id)

    def _generate_article_summary(self, article: LegalArticleModel) -> ArticleSummary:
        """Generate summary for a single article using LLM."""
        # Get article content (truncate if too long)
        content = article.raw_text or ""
        if len(content) > 2000:
            content = content[:2000] + "..."

        # LLM prompt to extract keywords
        prompt = f"""Bạn là chuyên gia pháp lý Việt Nam. Dựa trên điều luật sau:

Điều {article.article_number}: {article.title or 'Không có tiêu đề'}

Nội dung:
{content}

Hãy tóm tắt điều luật này bằng 10-15 TỪ KHÓA TÌM KIẾM bao gồm:

1. CHỦ THỂ: ai là đối tượng áp dụng (công ty, cổ đông, thành viên, giám đốc...)
2. HÀNH ĐỘNG: động từ chính (thành lập, giải thể, họp, biểu quyết, chuyển nhượng...)
3. ĐIỀU KIỆN: yêu cầu, tỷ lệ, thời hạn (51%, 30 ngày, đa số...)
4. TỪ ĐỒNG NGHĨA: từ người dùng hay hỏi liên quan đến điều này

Trả lời NGẮN GỌN, chỉ liệt kê các từ khóa cách nhau bằng dấu phẩy."""

        try:
            response = self.llm_provider.generate(prompt)
            keywords = response.strip()

            # Truncate if too long
            if len(keywords) > 300:
                keywords = keywords[:300].rsplit(",", 1)[0]

        except Exception as e:
            self.logger.warning(f"LLM failed for {article.id}: {e}, using title")
            keywords = article.title or ""

        # Build description
        title = article.title or f"Điều {article.article_number}"
        if keywords:
            description = f"{title}. Từ khóa: {keywords}"
        else:
            description = title

        return ArticleSummary(
            article_id=article.id,
            article_number=str(article.article_number),
            article_title=title,
            keywords=keywords,
            description=description,
        )

    def export_summaries(self, path: Optional[Path] = None) -> Path:
        """Export summaries to JSON file."""
        if path is None:
            path = self.output_dir / "article_summaries.json"

        data = {
            article_id: {
                "article_id": s.article_id,
                "article_number": s.article_number,
                "article_title": s.article_title,
                "keywords": s.keywords,
                "description": s.description,
            }
            for article_id, s in self.checkpoint.summaries.items()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported {len(data)} article summaries to {path}")
        return path

    @staticmethod
    def load_summaries(path: Path) -> Dict[str, ArticleSummary]:
        """Load summaries from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            article_id: ArticleSummary(
                article_id=s["article_id"],
                article_number=s["article_number"],
                article_title=s["article_title"],
                keywords=s["keywords"],
                description=s["description"],
            )
            for article_id, s in data.items()
        }
