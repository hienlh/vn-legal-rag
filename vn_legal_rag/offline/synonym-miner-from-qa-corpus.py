"""
Synonym Miner for Vietnamese Legal Query Expansion.

Mine informal→formal term mappings from Q&A corpus to improve query understanding.
Uses LLM-based extraction to identify colloquial terms and their formal equivalents.

Features:
- Extract from training CSV/Q&A pairs
- LLM-powered synonym detection
- Output for DomainConfig.synonyms
- Enables query expansion for informal queries

Example mappings:
- "lập công ty" → ["thành lập doanh nghiệp"]
- "mở công ty" → ["thành lập doanh nghiệp", "đăng ký kinh doanh"]
- "vốn điều lệ" → ["vốn điều lệ công ty"]

Usage:
    >>> from vn_legal_rag.offline import SynonymMiner
    >>> miner = SynonymMiner()
    >>> synonyms = miner.mine_from_csv("data/qa_pairs.csv")
    >>> miner.update_domain_config("59-2020-QH14", synonyms)
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from ..utils import create_llm_provider, get_logger


@dataclass
class SynonymPair:
    """A synonym pair mapping informal to formal term."""

    informal: str  # Colloquial/informal term
    formal: str  # Official/formal term
    context: Optional[str] = None  # Example context
    confidence: float = 0.8
    source: Optional[str] = None  # Source Q&A or document


@dataclass
class SynonymMiningResult:
    """Result of synonym mining operation."""

    pairs: List[SynonymPair] = field(default_factory=list)
    source_count: int = 0  # Number of Q&A pairs processed
    errors: List[str] = field(default_factory=list)

    def to_synonyms_dict(self) -> Dict[str, List[str]]:
        """Convert to synonyms dict for DomainConfig."""
        result: Dict[str, List[str]] = {}
        for pair in self.pairs:
            if pair.informal not in result:
                result[pair.informal] = []
            if pair.formal not in result[pair.informal]:
                result[pair.informal].append(pair.formal)
        return result

    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps(
            {
                "pairs": [
                    {
                        "informal": p.informal,
                        "formal": p.formal,
                        "confidence": p.confidence,
                    }
                    for p in self.pairs
                ],
                "source_count": self.source_count,
            },
            ensure_ascii=False,
            indent=2,
        )


class SynonymMiner:
    """
    Mine synonym pairs from Q&A corpus for query expansion.

    Uses LLM to identify informal/colloquial terms in questions
    and map them to formal legal terminology in answers.
    """

    EXTRACTION_PROMPT = """Phân tích cặp câu hỏi-trả lời pháp luật Việt Nam sau để tìm các cặp từ đồng nghĩa:
- Từ informal/colloquial trong câu hỏi
- Từ formal/chính thức tương ứng trong câu trả lời

Câu hỏi: {question}

Câu trả lời: {answer}

Trả về JSON array với format:
[{{"informal": "từ informal", "formal": "từ formal", "confidence": 0.8}}]

Chỉ trả về các cặp có confidence >= 0.7. Nếu không tìm thấy, trả về [].
Ví dụ: "lập công ty" → "thành lập doanh nghiệp"

JSON:"""

    BATCH_EXTRACTION_PROMPT = """Phân tích các cặp Q&A pháp luật Việt Nam sau để tìm cặp từ đồng nghĩa informal→formal.

{qa_pairs}

Trả về JSON array consolidate tất cả cặp từ tìm được:
[{{"informal": "từ informal", "formal": "từ formal", "confidence": 0.8, "example": "context ngắn"}}]

Chỉ giữ các cặp có confidence >= 0.7 và xuất hiện trong >= 2 Q&A pairs.
Loại bỏ duplicates, merge các cặp giống nhau.

JSON:"""

    # Known informal→formal mappings (seed data)
    SEED_SYNONYMS: Dict[str, List[str]] = {
        "lập công ty": ["thành lập doanh nghiệp", "đăng ký kinh doanh"],
        "mở công ty": ["thành lập doanh nghiệp"],
        "đóng cửa công ty": ["giải thể doanh nghiệp"],
        "dẹp công ty": ["giải thể doanh nghiệp"],
        "vốn góp": ["phần vốn góp", "vốn điều lệ"],
        "cổ phần": ["cổ phần công ty"],
        "hợp đồng": ["hợp đồng dân sự", "khế ước"],
        "sếp": ["người quản lý", "giám đốc"],
        "chủ công ty": ["chủ sở hữu", "thành viên công ty"],
        "nhân viên": ["người lao động"],
        "sa thải": ["chấm dứt hợp đồng lao động"],
        "đuổi việc": ["chấm dứt hợp đồng lao động"],
    }

    def __init__(
        self,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-haiku-4-5-20251001",
        config_dir: str = "config/domains",
        batch_size: int = 10,
    ):
        """
        Initialize synonym miner.

        Args:
            llm_provider: LLM provider name
            llm_model: LLM model name
            config_dir: Domain config directory
            batch_size: Number of Q&A pairs per batch
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.config_dir = Path(config_dir)
        self.batch_size = batch_size
        self.logger = get_logger("synonym_miner")

        self._llm = None

    @property
    def llm(self):
        """Lazy-load LLM provider."""
        if self._llm is None:
            self._llm = create_llm_provider(self.llm_provider, self.llm_model)
        return self._llm

    def mine_from_csv(
        self,
        csv_path: str,
        question_col: str = "question",
        answer_col: str = "answer",
        limit: Optional[int] = None,
    ) -> SynonymMiningResult:
        """
        Mine synonyms from Q&A CSV file.

        Args:
            csv_path: Path to CSV file
            question_col: Column name for questions
            answer_col: Column name for answers
            limit: Max number of pairs to process

        Returns:
            SynonymMiningResult with extracted pairs
        """
        result = SynonymMiningResult()
        qa_pairs: List[Tuple[str, str]] = []

        # Load Q&A pairs from CSV
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                question = row.get(question_col, "").strip()
                answer = row.get(answer_col, "").strip()
                if question and answer:
                    qa_pairs.append((question, answer))

        result.source_count = len(qa_pairs)
        self.logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {csv_path}")

        # Process in batches
        all_pairs: Dict[str, SynonymPair] = {}

        for i in range(0, len(qa_pairs), self.batch_size):
            batch = qa_pairs[i : i + self.batch_size]
            try:
                batch_pairs = self._extract_from_batch(batch)
                for pair in batch_pairs:
                    key = f"{pair.informal}|{pair.formal}"
                    if key not in all_pairs or pair.confidence > all_pairs[key].confidence:
                        all_pairs[key] = pair
            except Exception as e:
                result.errors.append(f"Batch {i}: {str(e)}")
                self.logger.warning(f"Batch extraction failed: {e}")

        result.pairs = list(all_pairs.values())
        self.logger.info(f"Extracted {len(result.pairs)} synonym pairs")
        return result

    def mine_from_qa_list(
        self,
        qa_pairs: List[Tuple[str, str]],
    ) -> SynonymMiningResult:
        """
        Mine synonyms from list of Q&A pairs.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            SynonymMiningResult
        """
        result = SynonymMiningResult(source_count=len(qa_pairs))
        all_pairs: Dict[str, SynonymPair] = {}

        for i in range(0, len(qa_pairs), self.batch_size):
            batch = qa_pairs[i : i + self.batch_size]
            try:
                batch_pairs = self._extract_from_batch(batch)
                for pair in batch_pairs:
                    key = f"{pair.informal}|{pair.formal}"
                    if key not in all_pairs or pair.confidence > all_pairs[key].confidence:
                        all_pairs[key] = pair
            except Exception as e:
                result.errors.append(str(e))

        result.pairs = list(all_pairs.values())
        return result

    def _extract_from_batch(
        self,
        batch: List[Tuple[str, str]],
    ) -> List[SynonymPair]:
        """Extract synonyms from a batch of Q&A pairs using LLM."""
        # Format Q&A pairs for prompt
        qa_text = "\n\n".join(
            f"Q{i+1}: {q}\nA{i+1}: {a[:500]}"  # Truncate long answers
            for i, (q, a) in enumerate(batch)
        )

        prompt = self.BATCH_EXTRACTION_PROMPT.format(qa_pairs=qa_text)

        # Call LLM
        response = self.llm.generate(prompt, max_tokens=1000)

        # Parse JSON response
        pairs = self._parse_llm_response(response)
        return pairs

    def _parse_llm_response(self, response: str) -> List[SynonymPair]:
        """Parse LLM response to extract synonym pairs."""
        pairs: List[SynonymPair] = []

        # Extract JSON from response
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            return pairs

        try:
            data = json.loads(json_match.group())
            for item in data:
                if isinstance(item, dict):
                    informal = item.get("informal", "").strip()
                    formal = item.get("formal", "").strip()
                    confidence = float(item.get("confidence", 0.8))

                    if informal and formal and confidence >= 0.7:
                        pairs.append(
                            SynonymPair(
                                informal=informal,
                                formal=formal,
                                confidence=confidence,
                                context=item.get("example"),
                            )
                        )
        except json.JSONDecodeError:
            pass

        return pairs

    def get_seed_synonyms(self) -> SynonymMiningResult:
        """Get pre-defined seed synonyms."""
        result = SynonymMiningResult()
        for informal, formals in self.SEED_SYNONYMS.items():
            for formal in formals:
                result.pairs.append(
                    SynonymPair(
                        informal=informal,
                        formal=formal,
                        confidence=1.0,
                        source="seed",
                    )
                )
        return result

    def merge_with_seeds(
        self,
        mined: SynonymMiningResult,
    ) -> SynonymMiningResult:
        """Merge mined synonyms with seed synonyms."""
        seeds = self.get_seed_synonyms()

        # Combine, preferring higher confidence
        all_pairs: Dict[str, SynonymPair] = {}

        for pair in seeds.pairs:
            key = f"{pair.informal}|{pair.formal}"
            all_pairs[key] = pair

        for pair in mined.pairs:
            key = f"{pair.informal}|{pair.formal}"
            if key not in all_pairs or pair.confidence > all_pairs[key].confidence:
                all_pairs[key] = pair

        result = SynonymMiningResult(
            pairs=list(all_pairs.values()),
            source_count=mined.source_count,
            errors=mined.errors,
        )
        return result

    def update_domain_config(
        self,
        document_id: str,
        result: SynonymMiningResult,
        merge: bool = True,
    ) -> Path:
        """
        Update domain config YAML with mined synonyms.

        Args:
            document_id: Document ID
            result: Mining result
            merge: Merge with existing or replace

        Returns:
            Path to updated config file
        """
        config_path = self.config_dir / f"{document_id}.yaml"

        config: Dict[str, Any] = {}
        if merge and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Update synonyms
        synonyms = config.get("synonyms", {})
        for pair in result.pairs:
            if pair.informal not in synonyms:
                synonyms[pair.informal] = []
            if pair.formal not in synonyms[pair.informal]:
                synonyms[pair.informal].append(pair.formal)

        config["synonyms"] = synonyms

        # Save
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        self.logger.info(
            f"Updated {config_path} with {len(result.pairs)} synonyms"
        )
        return config_path

    def export_to_json(
        self,
        result: SynonymMiningResult,
        output_path: str,
    ) -> Path:
        """Export synonyms to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        return path


def mine_synonyms_for_domain(
    document_id: str,
    csv_path: str,
    update_config: bool = True,
    include_seeds: bool = True,
    config_dir: str = "config/domains",
) -> SynonymMiningResult:
    """
    Convenience function to mine synonyms and update domain config.

    Args:
        document_id: Document ID for config update
        csv_path: Path to Q&A CSV
        update_config: Whether to update domain config
        include_seeds: Include seed synonyms
        config_dir: Config directory

    Returns:
        SynonymMiningResult
    """
    miner = SynonymMiner(config_dir=config_dir)

    result = miner.mine_from_csv(csv_path)

    if include_seeds:
        result = miner.merge_with_seeds(result)

    if update_config:
        miner.update_domain_config(document_id, result)

    return result
