"""
Legal Entity Profiler

LLM-based profiling for Vietnamese legal entities.
Generates descriptions, search keywords, and related concepts for each entity.

Based on LightRAG's LLM Profiling concept adapted for Vietnamese legal domain.

Features:
- Batch processing with rate limiting
- Response caching to avoid redundant API calls
- Vietnamese legal domain-specific prompts
- Profile includes: description, keywords, related concepts, canonical form

Usage:
    >>> from vn_legal_rag.offline.entity_profiler import EntityProfiler
    >>> profiler = EntityProfiler(provider="gemini", model="gemini-2.0-flash")
    >>> profiles = profiler.batch_profile(entities, batch_size=10)
"""

import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.basic_llm_provider import create_llm_provider
from ..utils.simple_logger import get_logger


@dataclass
class EntityProfile:
    """Profile for a legal entity."""

    entity_id: str
    entity_name: str
    entity_type: str

    # LLM-generated fields
    description: str = ""  # Vietnamese, 2-3 sentences
    search_keywords: List[str] = field(default_factory=list)  # 5-10 keywords
    related_concepts: List[str] = field(default_factory=list)  # 2-5 concepts
    canonical_form: str = ""  # Normalized form for deduplication

    # Metadata
    source_id: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityProfile":
        return cls(**data)


# Prompt template for Vietnamese legal entity profiling
ENTITY_PROFILE_PROMPT_VI = """Bạn là chuyên gia pháp lý Việt Nam. Phân tích entity sau và tạo hồ sơ:

Entity: {name}
Loại: {type}
Ngữ cảnh: {context}
Tài liệu: {document}

Hãy sinh:
1. **Mô tả** (2-3 câu tiếng Việt): Giải thích ngắn gọn entity này trong bối cảnh pháp luật Việt Nam.
2. **Từ khóa tìm kiếm** (5-10 từ): Các từ/cụm từ liên quan để tìm kiếm entity này.
3. **Khái niệm liên quan** (2-5 mục): Các khái niệm pháp lý cấp cao hơn liên quan.
4. **Dạng chuẩn hóa**: Tên chuẩn của entity (viết thường, không dấu, đầy đủ).

Trả về JSON:
{{
  "description": "...",
  "search_keywords": ["từ 1", "từ 2", ...],
  "related_concepts": ["khái niệm 1", "khái niệm 2", ...],
  "canonical_form": "..."
}}

CHỈ trả về JSON, không có text khác."""


class ProfileCache:
    """SQLite-based cache for entity profiles."""

    def __init__(self, db_path: str = "data/profile_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profile_cache (
                    cache_key TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_id ON profile_cache(entity_id)"
            )

    def _make_key(self, entity_id: str, entity_name: str, model: str) -> str:
        """Create cache key from entity info."""
        content = f"{entity_id}:{entity_name}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(
        self, entity_id: str, entity_name: str, model: str
    ) -> Optional[EntityProfile]:
        """Get cached profile."""
        key = self._make_key(entity_id, entity_name, model)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT profile_json FROM profile_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row:
                return EntityProfile.from_dict(json.loads(row[0]))
        return None

    def set(self, profile: EntityProfile, model: str):
        """Cache a profile."""
        key = self._make_key(profile.entity_id, profile.entity_name, model)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO profile_cache
                (cache_key, entity_id, entity_name, profile_json, model)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    profile.entity_id,
                    profile.entity_name,
                    json.dumps(profile.to_dict(), ensure_ascii=False),
                    model,
                ),
            )

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM profile_cache").fetchone()[0]
            by_model = dict(
                conn.execute(
                    "SELECT model, COUNT(*) FROM profile_cache GROUP BY model"
                ).fetchall()
            )
        return {"total": total, "by_model": by_model}

    def clear(self):
        """Clear all cached profiles."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM profile_cache")


class EntityProfiler:
    """LLM-based entity profiler for Vietnamese legal entities."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        cache_db: str = "data/profile_cache.db",
        use_cache: bool = True,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize profiler.

        Args:
            provider: LLM provider (gemini, openai)
            model: Model name
            cache_db: Path to cache database
            use_cache: Whether to use caching
            timeout: LLM call timeout in seconds
            max_retries: Max retries on failure
            rate_limit_delay: Delay between API calls
        """
        self.provider_name = provider
        self.model = model
        self.use_cache = use_cache
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

        self.cache = ProfileCache(cache_db) if use_cache else None
        self.logger = get_logger("entity_profiler")

        # Create LLM provider
        try:
            self.llm = create_llm_provider(provider, model=model)
        except Exception as e:
            self.logger.warning(f"Failed to create LLM provider: {e}")
            self.llm = None

    def profile_entity(
        self,
        entity: Dict[str, Any],
        context: str = "",
        document: str = "",
    ) -> Optional[EntityProfile]:
        """
        Profile a single entity.

        Args:
            entity: Entity dict with 'id', 'name'/'text', 'type'/'label'
            context: Source text context
            document: Document title

        Returns:
            EntityProfile or None if failed
        """
        entity_id = entity.get("id", "")
        entity_name = entity.get("name", entity.get("text", ""))
        entity_type = entity.get("type", entity.get("label", ""))

        if not entity_name:
            return None

        # Check cache
        if self.cache:
            cached = self.cache.get(entity_id, entity_name, self.model)
            if cached:
                self.logger.debug(f"Cache hit for {entity_id}")
                return cached

        # Generate profile via LLM
        if not self.llm:
            self.logger.warning("No LLM available for profiling")
            return self._fallback_profile(entity_id, entity_name, entity_type)

        prompt = ENTITY_PROFILE_PROMPT_VI.format(
            name=entity_name,
            type=entity_type,
            context=context[:500] if context else "(không có ngữ cảnh)",
            document=document or "(không rõ)",
        )

        # Call LLM with retry
        response = self._call_llm_with_retry(prompt)
        if not response:
            return self._fallback_profile(entity_id, entity_name, entity_type)

        # Parse response
        profile = self._parse_response(response, entity_id, entity_name, entity_type)

        # Cache result
        if self.cache and profile:
            self.cache.set(profile, self.model)

        return profile

    def batch_profile(
        self,
        entities: List[Dict[str, Any]],
        batch_size: int = 10,
        contexts: Optional[Dict[str, str]] = None,
        document: str = "",
        show_progress: bool = True,
    ) -> List[EntityProfile]:
        """
        Profile multiple entities in batches.

        Args:
            entities: List of entity dicts
            batch_size: Number of entities per batch (for progress tracking)
            contexts: Optional dict mapping entity_id -> context text
            document: Document title
            show_progress: Print progress

        Returns:
            List of EntityProfile objects
        """
        profiles = []
        total = len(entities)
        contexts = contexts or {}

        self.logger.info(f"Profiling {total} entities")

        for idx, entity in enumerate(entities):
            entity_id = entity.get("id", "")
            context = contexts.get(entity_id, "")

            if show_progress and (idx + 1) % batch_size == 0:
                print(f"  [Profile] {idx + 1}/{total}")

            profile = self.profile_entity(entity, context, document)
            if profile:
                profiles.append(profile)

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        self.logger.info(f"Profiled {len(profiles)}/{total} entities")
        return profiles

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """Call LLM with timeout and retry."""
        for attempt in range(self.max_retries):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.llm.generate, prompt)
                    response = future.result(timeout=self.timeout)
                    return response
            except FuturesTimeoutError:
                self.logger.warning(f"LLM timeout (attempt {attempt + 1})")
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "429" in error_msg:
                    delay = self.rate_limit_delay * (2**attempt)
                    self.logger.warning(f"Rate limited, waiting {delay}s")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"LLM error: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.rate_limit_delay)

        return None

    def _parse_response(
        self,
        response: str,
        entity_id: str,
        entity_name: str,
        entity_type: str,
    ) -> Optional[EntityProfile]:
        """Parse LLM response to EntityProfile."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                self.logger.warning(f"No JSON in response for {entity_id}")
                return self._fallback_profile(entity_id, entity_name, entity_type)

            data = json.loads(json_match.group())

            return EntityProfile(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=entity_type,
                description=data.get("description", ""),
                search_keywords=data.get("search_keywords", []),
                related_concepts=data.get("related_concepts", []),
                canonical_form=data.get("canonical_form", entity_name.lower()),
                confidence=0.9,
            )
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse error for {entity_id}: {e}")
            return self._fallback_profile(entity_id, entity_name, entity_type)

    def _fallback_profile(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: str,
    ) -> EntityProfile:
        """Create fallback profile without LLM."""
        keywords = entity_name.lower().split()

        return EntityProfile(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            description=f"{entity_name} ({entity_type})",
            search_keywords=keywords[:5],
            related_concepts=[],
            canonical_form=entity_name.lower(),
            confidence=0.5,
        )

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.stats()
        return {"total": 0, "by_model": {}}
