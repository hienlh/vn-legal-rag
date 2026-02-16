"""
Vietnamese Legal Abbreviation Extractor.

Generalizable approach using underthesea POS tagging + consonant ratio analysis.
No hardcoded word lists - automatically detects abbreviations vs Vietnamese words.

Detection Strategy:
1. POS tagging with underthesea to identify Vietnamese words (N, V, A, E, C, R, P tags)
2. Consonant ratio analysis - Vietnamese abbreviations are typically all consonants
3. Pattern matching for hyphenated abbreviations (NĐ-CP)
4. Length and frequency filtering

Usage:
    extractor = AbbreviationExtractor()
    abbreviations = extractor.extract_from_text(text)
    # Returns: [AbbreviationMatch(abbrev="HĐQT", count=5, confidence=100, ...)]
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Vietnamese vowels (including diacritics)
VIETNAMESE_VOWELS = set(
    "AĂÂEÊIOÔƠUƯYaăâeêioôơuưy"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)

# POS tags that indicate a common Vietnamese word (not abbreviation)
# N=noun, V=verb, A=adjective, E=preposition, C=conjunction, R=adverb, P=pronoun
VIETNAMESE_WORD_POS_TAGS = {"N", "V", "A", "E", "C", "R", "P", "L", "T", "I"}

# Lazy-load underthesea to avoid import overhead when not needed
_pos_tag = None


def _get_pos_tagger():
    """Lazy load underthesea pos_tag function."""
    global _pos_tag
    if _pos_tag is None:
        try:
            from underthesea import pos_tag
            _pos_tag = pos_tag
        except ImportError:
            _pos_tag = False  # Mark as unavailable
    return _pos_tag if _pos_tag else None


@dataclass
class AbbreviationMatch:
    """Result of abbreviation detection."""

    abbreviation: str
    count: int = 0
    confidence: int = 100  # 0-100
    detection_reason: str = ""
    pos_tag: Optional[str] = None  # POS tag from underthesea
    sample_context: Optional[str] = None
    full_form: Optional[str] = None  # Auto-detected full form
    full_form_confidence: int = 0  # 0-100 confidence of full form detection


@dataclass
class AbbreviationExtractor:
    """
    Extract Vietnamese abbreviations from text using generalizable detection.

    Uses multi-signal approach:
    1. POS tagging - Vietnamese words get N/V/A tags, abbreviations don't
    2. Consonant ratio - abbreviations are typically all consonants
    3. Hyphen pattern - compound abbreviations like NĐ-CP
    4. Length filter - abbreviations are typically 2-6 characters
    5. Full form detection - looks for "X (ABBREV)" or "X (gọi tắt là ABBREV)"
    """

    # Detection thresholds
    min_length: int = 2
    max_length: int = 10
    min_count: int = 1  # minimum occurrences to include

    # Regex pattern for potential abbreviations
    # Matches UPPERCASE words (may contain hyphen or numbers)
    _abbrev_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"\b([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]"
            r"[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ0-9\-]{1,9})\b"
        )
    )

    # Patterns for full form detection
    _full_form_patterns: List[Tuple[re.Pattern, int]] = field(default_factory=lambda: [
        # "Hội đồng quản trị (sau đây gọi tắt là HĐQT)"
        (re.compile(
            r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴa-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ\s]{3,50})"
            r"\s*\(\s*(?:sau đây |từ đây )?(?:gọi tắt là|viết tắt là|gọi là)\s+"
            r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ0-9\-]{1,9})\s*\)",
            re.IGNORECASE
        ), 95),
        # "Hội đồng quản trị (HĐQT)" - direct parenthesis
        (re.compile(
            r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴa-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ\s]{3,50})"
            r"\s*\(\s*"
            r"([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ0-9\-]{1,9})\s*\)"
        ), 80),
    ])

    # Cache for POS tag results to avoid repeated calls
    _pos_cache: Dict[str, str] = field(default_factory=dict)

    # Cache for full form results
    _full_form_cache: Dict[str, Tuple[str, int]] = field(default_factory=dict)

    def _extract_full_forms_from_text(self, text: str) -> None:
        """Scan text for full form definitions and cache them."""
        for pattern, confidence in self._full_form_patterns:
            for match in pattern.finditer(text):
                full_form = match.group(1).strip()
                abbrev = match.group(2).strip().upper()

                # Validate: full form should be longer than abbreviation
                if len(full_form) <= len(abbrev):
                    continue

                # Validate: full form should contain Vietnamese words
                if full_form.isupper() and len(full_form.split()) <= 1:
                    continue

                # Only update if this is a higher confidence match
                if abbrev not in self._full_form_cache:
                    self._full_form_cache[abbrev] = (full_form, confidence)
                elif confidence > self._full_form_cache[abbrev][1]:
                    self._full_form_cache[abbrev] = (full_form, confidence)

    def _get_full_form(self, abbrev: str) -> Tuple[Optional[str], int]:
        """Get cached full form for an abbreviation."""
        abbrev_upper = abbrev.upper()
        if abbrev_upper in self._full_form_cache:
            return self._full_form_cache[abbrev_upper]
        return None, 0

    def _get_pos_tag_in_context(self, word: str, context: str) -> Optional[str]:
        """Get POS tag for a word in its sentence context using underthesea."""
        # Check cache first
        cache_key = f"{word}:{context[:50]}"
        if cache_key in self._pos_cache:
            return self._pos_cache[cache_key]

        pos_tagger = _get_pos_tagger()
        if not pos_tagger:
            return None

        try:
            # POS tag the context
            tagged = pos_tagger(context)
            word_upper = word.upper()

            # Find the word in tagged results
            for token, tag in tagged:
                token_upper = token.upper()
                if token_upper == word_upper:
                    self._pos_cache[cache_key] = tag
                    return tag
                if token_upper.startswith(word_upper + " "):
                    self._pos_cache[cache_key] = tag
                    return tag
                if word_upper in token_upper.split():
                    self._pos_cache[cache_key] = tag
                    return tag
        except Exception:
            pass

        return None

    def _analyze_word(
        self, word: str, context: str
    ) -> Tuple[bool, int, str, Optional[str]]:
        """
        Analyze if a word is an abbreviation using multiple signals.

        Returns:
            (is_abbreviation, confidence, reason, pos_tag)
        """
        clean = word.replace("-", "")

        # Calculate consonant ratio
        consonants = sum(1 for c in clean if c.upper() not in VIETNAMESE_VOWELS)
        has_vowel = any(c.upper() in VIETNAMESE_VOWELS for c in clean)
        cons_ratio = consonants / len(clean) if clean else 0

        # Get POS tag
        pos_tag = self._get_pos_tag_in_context(word, context)

        # Rule 1: All consonants = definitely abbreviation
        if not has_vowel and len(clean) >= 2:
            return True, 100, "all_consonants", pos_tag

        # Rule 2: Hyphenated + short = likely abbreviation
        if "-" in word and len(clean) <= 6:
            return True, 95, "hyphenated", pos_tag

        # Rule 3: If POS tag indicates common Vietnamese word, NOT abbreviation
        if pos_tag in VIETNAMESE_WORD_POS_TAGS:
            return False, 10, f"pos_{pos_tag}", pos_tag

        # Rule 4: Has vowels AND tagged as proper noun
        if has_vowel and pos_tag in {"Np", "Ny", "Nb", "Nu"}:
            if len(clean) > 4 or cons_ratio < 0.70:
                return False, 15, f"pos_{pos_tag}_has_vowel", pos_tag

        # Rule 5: High consonant ratio + short + NOT tagged as common word
        if cons_ratio >= 0.75 and len(clean) <= 5:
            if pos_tag is None or pos_tag not in VIETNAMESE_WORD_POS_TAGS:
                return True, 85, "high_cons_short_untagged", pos_tag

        # Rule 6: POS = M (classifier) with high consonant ratio
        if pos_tag == "M":
            if cons_ratio >= 0.7:
                return True, 75, f"pos_{pos_tag}_high_cons", pos_tag

        # Rule 7: Very high consonant ratio even for longer words
        if cons_ratio >= 0.85 and len(clean) <= 6:
            if pos_tag is None or pos_tag not in VIETNAMESE_WORD_POS_TAGS:
                return True, 70, "very_high_cons", pos_tag

        return False, 0, "", pos_tag

    def extract_from_text(self, text: str) -> List[AbbreviationMatch]:
        """Extract abbreviations from text."""
        if not text:
            return []

        # Step 1: Scan for full form definitions first
        self._extract_full_forms_from_text(text)

        # Step 2: Find all potential abbreviations
        candidates: Dict[str, Dict] = {}

        for match in self._abbrev_pattern.finditer(text):
            word = match.group(1)

            # Skip if too short/long
            clean_word = word.replace("-", "")
            if not (self.min_length <= len(clean_word) <= self.max_length):
                continue

            # Get context (100 chars around for better POS tagging)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].replace("\n", " ")

            # Check if already seen
            if word in candidates:
                candidates[word]["count"] += 1
            else:
                is_abbrev, confidence, reason, pos_tag = self._analyze_word(
                    word, context
                )
                if is_abbrev:
                    candidates[word] = {
                        "count": 1,
                        "confidence": confidence,
                        "reason": reason,
                        "pos_tag": pos_tag,
                        "context": context,
                    }

        # Step 3: Filter by min count and convert to results
        results = []
        for abbrev, data in candidates.items():
            if data["count"] >= self.min_count:
                full_form, ff_confidence = self._get_full_form(abbrev)
                results.append(
                    AbbreviationMatch(
                        abbreviation=abbrev,
                        count=data["count"],
                        confidence=data["confidence"],
                        detection_reason=data["reason"],
                        pos_tag=data["pos_tag"],
                        sample_context=data["context"],
                        full_form=full_form,
                        full_form_confidence=ff_confidence,
                    )
                )

        results.sort(key=lambda x: x.count, reverse=True)
        return results

    def extract_from_texts(self, texts: List[str]) -> List[AbbreviationMatch]:
        """Extract abbreviations from multiple texts and aggregate counts."""
        aggregated: Dict[str, AbbreviationMatch] = {}

        for text in texts:
            matches = self.extract_from_text(text)
            for match in matches:
                if match.abbreviation in aggregated:
                    aggregated[match.abbreviation].count += match.count
                    if (match.full_form and
                        match.full_form_confidence >
                        aggregated[match.abbreviation].full_form_confidence):
                        aggregated[match.abbreviation].full_form = match.full_form
                        aggregated[match.abbreviation].full_form_confidence = (
                            match.full_form_confidence
                        )
                else:
                    aggregated[match.abbreviation] = match

        results = list(aggregated.values())
        results.sort(key=lambda x: x.count, reverse=True)
        return results

    def clear_cache(self) -> None:
        """Clear all caches (POS tags and full forms)."""
        self._pos_cache.clear()
        self._full_form_cache.clear()


# Helper functions for database integration

def get_full_form(abbrev: str) -> Optional[str]:  # noqa: ARG001
    """Get full form of an abbreviation (from database)."""
    return None


def expand_search_terms(query: str) -> List[str]:
    """Expand search query with abbreviation variations."""
    return [query]


# Backward compatibility - empty dict since we no longer use hardcoded abbreviations
KNOWN_LEGAL_ABBREVIATIONS: Dict[str, Dict[str, str]] = {}
