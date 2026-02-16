"""
Summary generators for Vietnamese legal RAG offline phase.

Exports:
- ChapterSummary, ChapterSummaryGenerator - Chapter keyword extraction for tree traversal
- ArticleSummary, ArticleSummaryGenerator - Article keyword extraction for loop 2
- DocumentSummary, DocumentSummaryGenerator - Document summaries for loop 0
"""

from importlib import import_module

# Chapter summary generator (kebab-case filename)
_chapter_mod = import_module(".chapter-summary-generator", "vn_legal_rag.offline.summary")
ChapterSummary = _chapter_mod.ChapterSummary
ChapterSummaryCheckpoint = _chapter_mod.ChapterSummaryCheckpoint
ChapterSummaryGenerator = _chapter_mod.ChapterSummaryGenerator

# Article summary generator (kebab-case filename)
_article_mod = import_module(".article-summary-generator", "vn_legal_rag.offline.summary")
ArticleSummary = _article_mod.ArticleSummary
ArticleSummaryCheckpoint = _article_mod.ArticleSummaryCheckpoint
ArticleSummaryGenerator = _article_mod.ArticleSummaryGenerator

# Document summary generator (kebab-case filename)
_document_mod = import_module(".document-summary-generator", "vn_legal_rag.offline.summary")
DocumentSummary = _document_mod.DocumentSummary
DocumentSummaryCheckpoint = _document_mod.DocumentSummaryCheckpoint
DocumentSummaryGenerator = _document_mod.DocumentSummaryGenerator

__all__ = [
    # Chapter summaries
    "ChapterSummary",
    "ChapterSummaryCheckpoint",
    "ChapterSummaryGenerator",
    # Article summaries
    "ArticleSummary",
    "ArticleSummaryCheckpoint",
    "ArticleSummaryGenerator",
    # Document summaries
    "DocumentSummary",
    "DocumentSummaryCheckpoint",
    "DocumentSummaryGenerator",
]
