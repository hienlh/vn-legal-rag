"""
Unified Entity-Relation Extractor for Vietnamese Legal Documents

Single LLM call extraction for both entities and relations (LightRAG-style).
Optimized for Vietnamese legal domain with evidence tracking.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..utils.basic_llm_provider import create_llm_provider
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
    ):
        self.provider_name = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.logger = get_logger("unified_extractor")

        self._provider = create_llm_provider(
            provider,
            model=model,
            use_cache=use_cache,
            cache_db_path=cache_db_path,
        )

    def extract(
        self,
        text: str,
        source_id: str = "",
        document_id: str = "",
    ) -> ExtractionResult:
        """Extract entities and relations in single LLM call."""
        if not text or not text.strip():
            return ExtractionResult(
                entities=[],
                relations=[],
                source_id=source_id,
                document_id=document_id,
            )

        prompt = EXTRACTION_PROMPT.format(
            entity_types=ENTITY_TYPES_PROMPT,
            relation_types=RELATION_TYPES_PROMPT,
            text=text,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self._provider.generate(prompt, temperature=self.temperature)
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
                    self.logger.error(f"All attempts failed: {e}")
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

    def _parse_response(self, response: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse LLM response to extract entities and relations."""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            data = json.loads(json_str)

        entities = self._validate_entities(data.get("entities", []))
        relations = self._validate_relations(data.get("relations", []), entities)

        return entities, relations

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
