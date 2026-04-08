"""
Unified Entity-Relation Extractor for Vietnamese Legal Documents

Single LLM call extraction for both entities and relations (LightRAG-style).
Optimized for Vietnamese legal domain with evidence tracking.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..utils import create_llm_provider
from ..utils.simple_logger import get_logger

# Entity types for prompt
ENTITY_TYPES_PROMPT = """
- TỔ_CHỨC: Công ty, doanh nghiệp, cơ quan
- VAI_TRÒ: Chức vụ, vai trò
- THUẬT_NGỮ: Thuật ngữ pháp lý
- THAM_CHIẾU: Điều khoản luật
- TIỀN_TỆ: Số tiền
- TỶ_LỆ: Phần trăm
- THỜI_HẠN: Thời gian
- ĐIỀU_KIỆN: Điều kiện áp dụng
- HÀNH_VI: Hành động pháp lý
- CHẾ_TÀI: Hình phạt, chế tài
"""

# Relation types for prompt
RELATION_TYPES_PROMPT = """
- YÊU_CẦU: A bắt buộc phải có/thực hiện B
- CÓ_QUYỀN: A có quyền thực hiện B
- CÔNG_NHẬN: A công nhận B
- BẢO_ĐẢM: A bảo đảm/bảo hộ B
- CHO_PHÉP: A cho phép B
- BAO_GỒM: A bao gồm B
- ĐỊNH_NGHĨA: A được định nghĩa là B
- THAM_CHIẾU: A tham chiếu đến B
- ÁP_DỤNG: A áp dụng cho B
- ĐIỀU_KIỆN: A là điều kiện của B
- LIÊN_QUAN: A liên quan đến B
"""

EXTRACTION_PROMPT = """Bạn là chuyên gia trích xuất thông tin pháp lý Việt Nam.

Trích xuất TẤT CẢ thực thể và quan hệ từ văn bản pháp luật.

## Loại thực thể
{entity_types}

## Loại quan hệ
{relation_types}

## Quy tắc
1. Trích xuất ĐẦY ĐỦ các thực thể quan trọng
2. Mỗi quan hệ phải có EVIDENCE (trích dẫn nguyên văn)
3. Dùng tên thực thể CHÍNH XÁC như trong văn bản
4. Nếu có viết tắt, ghi cả viết tắt lẫn tên đầy đủ

## Văn bản
{text}

## Kết quả JSON:
```json
{{
  "entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"source": "...", "target": "...", "predicate": "...", "evidence": "..."}}]
}}
```
"""


@dataclass
class ExtractionResult:
    """Result of unified extraction."""
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    raw_response: str = ""
    source_id: str = ""
    document_id: str = ""


class UnifiedLegalExtractor:
    """Unified entity-relation extractor for Vietnamese legal documents."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_retries: int = 2,
        use_cache: bool = True,
        cache_db_path: str = "data/llm_cache.db",
        base_url: Optional[str] = None,
    ):
        self.provider_name = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.logger = get_logger("unified_extractor")

        cache_db = cache_db_path if use_cache else None
        llm_kwargs = {}
        if base_url:
            llm_kwargs["base_url"] = base_url
        self._provider = create_llm_provider(
            provider,
            model=model,
            cache_db=cache_db,
            **llm_kwargs,
        )

    # Texts longer than this trigger chunked extraction
    CHUNK_THRESHOLD = 2000
    CHUNK_SIZE = 1500

    def extract(
        self,
        text: str,
        source_id: str = "",
        document_id: str = "",
    ) -> ExtractionResult:
        """Extract entities and relations in single LLM call.

        For long texts (>CHUNK_THRESHOLD chars), splits into chunks
        and merges results to avoid LLM output truncation.
        """
        if not text or not text.strip():
            return ExtractionResult(
                entities=[],
                relations=[],
                source_id=source_id,
                document_id=document_id,
            )

        # Try single-call first
        result = self._extract_single(text, source_id, document_id)
        if result.entities or result.relations:
            return result

        # Fallback: chunked extraction for long texts
        if len(text) > self.CHUNK_THRESHOLD:
            self.logger.info(f"Falling back to chunked extraction for {source_id} ({len(text)} chars)")
            return self._extract_chunked(text, source_id, document_id)

        return result

    def _extract_single(
        self,
        text: str,
        source_id: str = "",
        document_id: str = "",
    ) -> ExtractionResult:
        """Single-call extraction with retries."""
        prompt = EXTRACTION_PROMPT.format(
            entity_types=ENTITY_TYPES_PROMPT,
            relation_types=RELATION_TYPES_PROMPT,
            text=text,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self._provider.generate(prompt, temperature=self.temperature, max_tokens=8192)
                entities, relations = self._parse_response(response)

                # Add metadata
                for entity in entities:
                    entity["source_id"] = source_id
                    entity["document_id"] = document_id

                for relation in relations:
                    relation["source_id"] = source_id
                    relation["document_id"] = document_id

                self.logger.info(
                    f"Extracted {len(entities)} entities, {len(relations)} relations from {source_id}"
                )

                return ExtractionResult(
                    entities=entities,
                    relations=relations,
                    raw_response=response,
                    source_id=source_id,
                    document_id=document_id,
                )

            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    self.logger.error(f"Single-call failed for {source_id}: {e}")
                    return ExtractionResult(
                        entities=[],
                        relations=[],
                        raw_response=str(e),
                        source_id=source_id,
                        document_id=document_id,
                    )

        return ExtractionResult(
            entities=[],
            relations=[],
            source_id=source_id,
            document_id=document_id,
        )

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks at paragraph boundaries."""
        paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.)', text)
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > self.CHUNK_SIZE and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + "\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _extract_chunked(
        self,
        text: str,
        source_id: str,
        document_id: str,
    ) -> ExtractionResult:
        """Extract from long text by splitting into chunks and merging."""
        chunks = self._split_into_chunks(text)
        self.logger.info(f"Split {source_id} into {len(chunks)} chunks")

        all_entities = []
        all_relations = []
        seen_entity_names = set()

        for i, chunk in enumerate(chunks):
            result = self._extract_single(chunk, source_id, document_id)
            # Deduplicate entities across chunks
            for e in result.entities:
                name_lower = e.get("name", "").lower()
                if name_lower not in seen_entity_names:
                    seen_entity_names.add(name_lower)
                    all_entities.append(e)
            all_relations.extend(result.relations)

        self.logger.info(
            f"Chunked extraction for {source_id}: "
            f"{len(all_entities)} entities, {len(all_relations)} relations "
            f"from {len(chunks)} chunks"
        )

        return ExtractionResult(
            entities=all_entities,
            relations=all_relations,
            source_id=source_id,
            document_id=document_id,
        )

    def _repair_truncated_json(self, json_str: str) -> str:
        """Repair JSON truncated by LLM output limit.

        Handles: unterminated strings, missing closing brackets,
        trailing commas, incomplete key-value pairs.
        """
        s = json_str.rstrip()

        # Remove trailing incomplete key-value pair (e.g. `"key": ` or `"key":`)
        s = re.sub(r',\s*"[^"]*"\s*:\s*$', '', s)
        # Remove trailing comma + incomplete string value (e.g. `"evidence": "some text`)
        s = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', s)

        # Count open/close braces and brackets
        in_string = False
        escape = False
        opens = []
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                opens.append(ch)
            elif ch == '}' and opens and opens[-1] == '{':
                opens.pop()
            elif ch == ']' and opens and opens[-1] == '[':
                opens.pop()

        # If we're inside a string, close it
        if in_string:
            s += '"'

        # Remove trailing commas before closing
        s = re.sub(r',\s*$', '', s)

        # Close unclosed brackets/braces
        for opener in reversed(opens):
            s += ']' if opener == '[' else '}'

        return s

    def _parse_response(self, response: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse LLM response to extract entities and relations."""
        # Try ```json ... ``` first, then raw JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # For truncated responses, ``` closing may be missing
            json_match = re.search(r'```json\s*(.*)', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

        # Try parsing as-is first
        for attempt_repair in [False, True]:
            try:
                candidate = json_str
                if attempt_repair:
                    candidate = self._repair_truncated_json(candidate)
                # Clean trailing commas
                candidate = re.sub(r',\s*}', '}', candidate)
                candidate = re.sub(r',\s*]', ']', candidate)
                data = json.loads(candidate)

                entities = self._validate_entities(data.get("entities", []))
                relations = self._validate_relations(data.get("relations", []), entities)

                if attempt_repair and (entities or relations):
                    self.logger.info(f"Repaired truncated JSON: {len(entities)} entities, {len(relations)} relations")

                return entities, relations
            except (json.JSONDecodeError, ValueError):
                if attempt_repair:
                    raise
                continue

        raise ValueError("Could not parse JSON from response")

    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate and clean entities."""
        valid = []
        seen_names = set()

        for e in entities:
            name = e.get("name", "").strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            valid.append({
                "name": name,
                "type": e.get("type", "THUẬT_NGỮ"),
                "description": e.get("description", ""),
                "confidence": 0.9,
            })

        return valid

    def _validate_relations(self, relations: List[Dict], entities: List[Dict]) -> List[Dict]:
        """Validate relations."""
        entity_names = {e["name"].lower() for e in entities}
        valid = []

        for r in relations:
            source = r.get("source", "").strip()
            target = r.get("target", "").strip()

            if not source or not target:
                continue

            source_exists = source.lower() in entity_names
            target_exists = target.lower() in entity_names

            valid.append({
                "source": source,
                "target": target,
                "predicate": r.get("predicate", "LIÊN_QUAN").strip().upper(),
                "evidence": r.get("evidence", ""),
                "confidence": 0.85 if (source_exists and target_exists) else 0.6,
                "grounded": source_exists and target_exists,
            })

        return valid
