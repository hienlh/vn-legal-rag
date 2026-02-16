"""
Legal document scraper package for Vietnamese legal sources.

Supports scraping from:
- thuvienphapluat.vn (TVPL) - Primary Vietnamese legal database

Exports:
- LegalDocument, LegalChapter, LegalSection, LegalArticle, etc. - Data models
- BaseLegalScraper - Abstract base class for scrapers
- TVPLScraper - TVPL website scraper
- HierarchyExtractor - Regex-based hierarchy extraction
"""

from importlib import import_module

# Base classes and data models (kebab-case filename)
_base_mod = import_module(".base-legal-scraper", "vn_legal_rag.offline.scraper")
LegalDocument = _base_mod.LegalDocument
LegalChapter = _base_mod.LegalChapter
LegalSection = _base_mod.LegalSection
LegalArticle = _base_mod.LegalArticle
LegalClause = _base_mod.LegalClause
LegalPoint = _base_mod.LegalPoint
LegalAppendix = _base_mod.LegalAppendix
LegalAppendixItem = _base_mod.LegalAppendixItem
BaseLegalScraper = _base_mod.BaseLegalScraper
DocumentStatus = _base_mod.DocumentStatus
DocumentType = _base_mod.DocumentType

# Hierarchy extractor (kebab-case filename)
_hierarchy_mod = import_module(".legal-hierarchy-extractor", "vn_legal_rag.offline.scraper")
HierarchyExtractor = _hierarchy_mod.HierarchyExtractor

# TVPL scraper (kebab-case filename)
_tvpl_mod = import_module(".tvpl-legal-scraper", "vn_legal_rag.offline.scraper")
TVPLScraper = _tvpl_mod.TVPLScraper

__all__ = [
    # Data models
    "LegalDocument",
    "LegalChapter",
    "LegalSection",
    "LegalArticle",
    "LegalClause",
    "LegalPoint",
    "LegalAppendix",
    "LegalAppendixItem",
    # Enums
    "DocumentStatus",
    "DocumentType",
    # Base class
    "BaseLegalScraper",
    # Implementations
    "TVPLScraper",
    "HierarchyExtractor",
]
