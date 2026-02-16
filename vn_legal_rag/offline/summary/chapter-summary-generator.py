"""
Chapter summary generator - LLM-based keyword extraction for tree navigation.

Runs during offline phase to generate chapter descriptions that help
LLM-guided tree traversal select the right chapter.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

from ...utils import create_llm_provider, get_logger
from ..database_manager import LegalDocumentDB
from ..models import LegalChapterModel, LegalDocumentModel


@dataclass
class ChapterSummary:
    """Summary of a chapter for LLM navigation."""
    chapter_id: str
    chapter_title: str
    article_range: str  # e.g., "Điều 1-16"
    keywords: str  # LLM-generated keywords
    description: str  # Combined article_range + keywords


CHECKPOINT_FILENAME = "chapter_summaries_checkpoint.json"


class ChapterSummaryCheckpoint:
    """Checkpoint for chapter summary generation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_path = output_dir / CHECKPOINT_FILENAME
        self.processed_chapters: Set[str] = set()
        self.summaries: Dict[str, ChapterSummary] = {}
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

            self.processed_chapters = set(data.get("processed_chapters", []))
            self.stats = data.get("stats", self.stats)

            # Reconstruct summaries
            for s in data.get("summaries", []):
                summary = ChapterSummary(
                    chapter_id=s["chapter_id"],
                    chapter_title=s["chapter_title"],
                    article_range=s["article_range"],
                    keywords=s["keywords"],
                    description=s["description"],
                )
                self.summaries[summary.chapter_id] = summary

            return True
        except Exception as e:
            print(f"  [Resume] Failed to load chapter checkpoint: {e}")
            return False

    def save(self) -> None:
        """Save checkpoint atomically."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "processed_chapters": list(self.processed_chapters),
            "summaries": [
                {
                    "chapter_id": s.chapter_id,
                    "chapter_title": s.chapter_title,
                    "article_range": s.article_range,
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

    def add_summary(self, summary: ChapterSummary) -> None:
        """Add summary and save checkpoint."""
        self.processed_chapters.add(summary.chapter_id)
        self.summaries[summary.chapter_id] = summary
        self.stats["successful"] += 1
        self.save()

    def add_failure(self, chapter_id: str) -> None:
        """Record failed extraction."""
        self.processed_chapters.add(chapter_id)
        self.stats["failed"] += 1
        self.save()

    def is_processed(self, chapter_id: str) -> bool:
        return chapter_id in self.processed_chapters

    def get_summary(self, chapter_id: str) -> Optional[ChapterSummary]:
        return self.summaries.get(chapter_id)

    def delete(self) -> None:
        """Delete checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


class ChapterSummaryGenerator:
    """
    Generate chapter summaries using LLM for better tree navigation.

    Extracts keywords from chapter content to help LLM select the right
    chapter during tree traversal.
    """

    def __init__(
        self,
        db: LegalDocumentDB,
        output_dir: Path,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        resume: bool = True,
        use_cache: bool = True,
        cache_db_path: str = "data/llm_cache.db",
    ):
        self.db = db
        self.output_dir = Path(output_dir)
        self.llm_provider = create_llm_provider(
            llm_provider,
            model=llm_model,
            use_cache=use_cache,
            cache_db_path=cache_db_path,
        )
        self.logger = get_logger("chapter_summary_generator")
        self.resume = resume
        self.checkpoint = ChapterSummaryCheckpoint(self.output_dir)

    def generate_all(self, doc_id: Optional[str] = None) -> Dict[str, ChapterSummary]:
        """
        Generate summaries for all chapters in document(s).

        Args:
            doc_id: Specific document ID, or None for all documents

        Returns:
            Dict mapping chapter_id to ChapterSummary
        """
        # Try to resume from checkpoint
        if self.resume and self.checkpoint.exists():
            if self.checkpoint.load():
                self.logger.info(
                    f"Resumed from checkpoint: {len(self.checkpoint.summaries)} summaries loaded"
                )

        with self.db.SessionLocal() as session:
            # Get documents
            if doc_id:
                docs = session.query(LegalDocumentModel).filter(
                    LegalDocumentModel.id == doc_id
                ).all()
            else:
                docs = session.query(LegalDocumentModel).all()

            total_chapters = sum(len(doc.chapters) for doc in docs)
            processed = 0

            for doc in docs:
                self.logger.info(f"Processing document: {doc.so_hieu} ({len(doc.chapters)} chapters)")

                for chapter in sorted(doc.chapters, key=lambda c: c.position):
                    processed += 1

                    # Skip if already processed
                    if self.checkpoint.is_processed(chapter.id):
                        self.logger.debug(f"Skipping {chapter.id} (already processed)")
                        continue

                    try:
                        doc_title = doc.title or doc.so_hieu
                        summary = self._generate_chapter_summary(chapter, doc_title)
                        self.checkpoint.add_summary(summary)
                        self.logger.info(
                            f"[{processed}/{total_chapters}] Generated: {chapter.title or chapter.id}"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to generate summary for {chapter.id}: {e}")
                        self.checkpoint.add_failure(chapter.id)

        self.logger.info(
            f"Completed: {self.checkpoint.stats['successful']} success, "
            f"{self.checkpoint.stats['failed']} failed"
        )

        return self.checkpoint.summaries

    def _generate_chapter_summary(
        self, chapter: LegalChapterModel, doc_title: str = ""
    ) -> ChapterSummary:
        """Generate summary for a single chapter using LLM."""
        # Calculate article range
        all_article_nums = []

        # Direct articles in chapter
        for article in chapter.articles:
            all_article_nums.append(article.article_number)

        # Articles in sections
        for section in chapter.sections:
            for article in section.articles:
                all_article_nums.append(article.article_number)

        all_article_nums.sort(key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else (x if isinstance(x, int) else 0))

        if all_article_nums:
            article_range = f"Điều {all_article_nums[0]}-{all_article_nums[-1]}"
        else:
            article_range = ""

        # Get ALL article titles for context
        article_titles = []
        for article in sorted(chapter.articles, key=lambda a: a.position):
            if article.title:
                article_titles.append(f"- Điều {article.article_number}: {article.title}")

        for section in chapter.sections:
            for article in sorted(section.articles, key=lambda a: a.position):
                if article.title:
                    article_titles.append(f"- Điều {article.article_number}: {article.title}")

        article_context = "\n".join(article_titles) if article_titles else "Không có tiêu đề điều luật"

        # Generic LLM prompt for any legal domain
        prompt = f"""Bạn là chuyên gia pháp lý Việt Nam. Phân tích chương sau để trích xuất từ khóa tìm kiếm.

VĂN BẢN: {doc_title}
CHƯƠNG: {chapter.title or 'Không tiêu đề'}
PHẠM VI: {article_range}

DANH SÁCH ĐIỀU LUẬT:
{article_context}

Hãy liệt kê 40-50 TỪ KHÓA TÌM KIẾM bao gồm:

1. TỪ KHÓA CHÍNH: Trích xuất từ tiêu đề các điều luật

2. CÂU HỎI/TÌNH HUỐNG: Các câu hỏi thực tế người dùng có thể hỏi liên quan đến chương này
   (VD: "thủ tục như thế nào", "cần hồ sơ gì", "điều kiện là gì")

3. VIẾT TẮT PHỔ BIẾN: Các từ viết tắt trong lĩnh vực này (nếu có)

4. HÀNH ĐỘNG/THỦ TỤC: Các động từ và hành động cụ thể trong chương

5. TỪ ĐỒNG NGHĨA/INFORMAL: Cách nói thông thường người dùng hay dùng thay cho thuật ngữ chính thức

6. KHÁI NIỆM LIÊN QUAN: Các khái niệm pháp lý liên quan đến nội dung chương

Trả lời NGẮN GỌN, chỉ liệt kê các từ khóa cách nhau bằng dấu phẩy (không giải thích, không đánh số)."""

        try:
            response = self.llm_provider.generate(prompt)
            keywords = response.strip()

            # Ensure not too long (800 chars for 40-50 keywords)
            if len(keywords) > 800:
                keywords = keywords[:800].rsplit(",", 1)[0]

        except Exception as e:
            self.logger.warning(f"LLM failed for {chapter.id}: {e}, using fallback")
            # Fallback: use chapter title as keywords
            keywords = chapter.title or ""

        # Build description
        if keywords:
            description = f"{article_range}. Nội dung: {keywords}"
        else:
            description = article_range

        return ChapterSummary(
            chapter_id=chapter.id,
            chapter_title=chapter.title or f"Chương {chapter.chapter_number}",
            article_range=article_range,
            keywords=keywords,
            description=description,
        )

    def export_summaries(self, path: Optional[Path] = None) -> Path:
        """Export summaries to JSON file."""
        if path is None:
            path = self.output_dir / "chapter_summaries.json"

        data = {
            chapter_id: {
                "chapter_id": s.chapter_id,
                "chapter_title": s.chapter_title,
                "article_range": s.article_range,
                "keywords": s.keywords,
                "description": s.description,
            }
            for chapter_id, s in self.checkpoint.summaries.items()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported {len(data)} chapter summaries to {path}")
        return path

    @staticmethod
    def load_summaries(path: Path) -> Dict[str, ChapterSummary]:
        """Load summaries from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            chapter_id: ChapterSummary(
                chapter_id=s["chapter_id"],
                chapter_title=s["chapter_title"],
                article_range=s["article_range"],
                keywords=s["keywords"],
                description=s["description"],
            )
            for chapter_id, s in data.items()
        }
