"""
Legal Entity Deduplicator

Entity deduplication for Vietnamese legal documents, running during extraction
(not post-hoc) to prevent duplicate entities from entering the KG.

Features:
- Jaro-Winkler similarity with configurable threshold (default: 0.98)
- Vietnamese-specific canonical form normalization
- Entity type-aware matching
- Remap dict for relation source/target ID updates

Ported from semantica/legal/deduplicator.py
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.simple_logger import get_logger

# Try to import jellyfish, fallback to simple implementation
try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False


# Vietnamese diacritics mapping for normalization
VIETNAMESE_DIACRITICS_MAP = {
    # a variants
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    # e variants
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    # i variants
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    # o variants
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    # u variants
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    # y variants
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    # d variant
    'đ': 'd',
}

# Common Vietnamese legal abbreviations
LEGAL_ABBREVIATIONS = {
    'ctcp': 'công ty cổ phần',
    'tnhh': 'trách nhiệm hữu hạn',
    'htx': 'hợp tác xã',
    'dntn': 'doanh nghiệp tư nhân',
    'tgđ': 'tổng giám đốc',
    'gđ': 'giám đốc',
    'hđqt': 'hội đồng quản trị',
    'bks': 'ban kiểm soát',
    'đhđcđ': 'đại hội đồng cổ đông',
    'vđl': 'vốn điều lệ',
    'nđ': 'nghị định',
    'tt': 'thông tư',
    'qđ': 'quyết định',
}


def _jaro_winkler_fallback(s1: str, s2: str) -> float:
    """Simple Jaro-Winkler fallback when jellyfish is not available."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity."""
    if HAS_JELLYFISH:
        return jellyfish.jaro_winkler_similarity(s1, s2)
    return _jaro_winkler_fallback(s1, s2)


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""
    merged_entities: List[Dict[str, Any]]
    remap: Dict[str, str]  # old_id -> canonical_id
    duplicates_found: int
    original_count: int
    final_count: int
    duplicate_pairs: List[Tuple[str, str, float]]


@dataclass
class EntityMatch:
    """A matched entity pair with similarity score."""
    entity: Dict[str, Any]
    canonical: Dict[str, Any]
    similarity: float
    match_reasons: List[str] = field(default_factory=list)


class LegalEntityDeduplicator:
    """
    Entity deduplicator for Vietnamese legal documents.

    Uses Jaro-Winkler similarity with Vietnamese-specific normalization.
    """

    def __init__(
        self,
        threshold: float = 0.98,
        type_weight: float = 0.1,
        use_abbreviation_expansion: bool = True,
        case_sensitive: bool = False,
        require_exact_for_short: bool = True,
        short_threshold: int = 15,
    ):
        self.threshold = threshold
        self.type_weight = type_weight
        self.use_abbreviation_expansion = use_abbreviation_expansion
        self.case_sensitive = case_sensitive
        self.require_exact_for_short = require_exact_for_short
        self.short_threshold = short_threshold
        self.logger = get_logger("legal_deduplicator")
        self._canonical_cache: Dict[str, str] = {}

    def get_canonical_form(self, text: str) -> str:
        """Get canonical form of text for comparison."""
        if text in self._canonical_cache:
            return self._canonical_cache[text]

        result = text

        if not self.case_sensitive:
            result = result.lower()

        if self.use_abbreviation_expansion:
            words = result.split()
            expanded_words = []
            for word in words:
                expanded = LEGAL_ABBREVIATIONS.get(word.lower(), word)
                expanded_words.append(expanded)
            result = ' '.join(expanded_words)

        result = self._remove_vietnamese_diacritics(result)
        result = re.sub(r'[^\w\s]', '', result)
        result = ' '.join(result.split())

        self._canonical_cache[text] = result
        return result

    def _remove_vietnamese_diacritics(self, text: str) -> str:
        """Remove Vietnamese diacritics from text."""
        result = []
        for char in text:
            lower_char = char.lower()
            if lower_char in VIETNAMESE_DIACRITICS_MAP:
                if char.isupper():
                    result.append(VIETNAMESE_DIACRITICS_MAP[lower_char].upper())
                else:
                    result.append(VIETNAMESE_DIACRITICS_MAP[lower_char])
            else:
                result.append(char)

        text = ''.join(result)
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        return text

    def calculate_similarity(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Calculate similarity between two entities."""
        name1 = entity1.get('name', entity1.get('text', ''))
        name2 = entity2.get('name', entity2.get('text', ''))

        if not name1 or not name2:
            return 0.0, []

        canonical1 = self.get_canonical_form(name1)
        canonical2 = self.get_canonical_form(name2)

        if not canonical1 or not canonical2:
            return 0.0, []

        jw_sim = jaro_winkler_similarity(canonical1, canonical2)
        reasons = []

        if canonical1 == canonical2:
            final_sim = 1.0
            reasons.append("exact_canonical_match")
        else:
            # Check for legal reference patterns
            legal_ref_pattern = re.compile(r'dieu\s+\d+|khoan\s+\d+|diem\s+\w+')
            has_ref1 = bool(legal_ref_pattern.search(canonical1))
            has_ref2 = bool(legal_ref_pattern.search(canonical2))

            # Check for numeric differences
            nums1 = set(re.findall(r'\d+', canonical1))
            nums2 = set(re.findall(r'\d+', canonical2))
            has_different_numbers = nums1 and nums2 and nums1 != nums2

            if has_ref1 or has_ref2:
                final_sim = 0.0
                reasons.append("legal_reference_no_exact_match")
            elif has_different_numbers:
                final_sim = 0.0
                reasons.append(f"different_numbers={nums1}!={nums2}")
            elif self.require_exact_for_short and min(len(canonical1), len(canonical2)) < self.short_threshold:
                final_sim = 0.0
                reasons.append("short_entity_no_exact_match")
            else:
                final_sim = jw_sim
                reasons.append(f"jaro_winkler={jw_sim:.3f}")

        type1 = entity1.get('type', entity1.get('label', ''))
        type2 = entity2.get('type', entity2.get('label', ''))

        if type1 and type2:
            if type1 == type2:
                reasons.append(f"same_type={type1}")
            else:
                final_sim *= 0.8
                reasons.append(f"type_mismatch={type1}!={type2}")

        return final_sim, reasons

    def find_best_match(
        self,
        entity: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[EntityMatch]:
        """Find best matching entity from candidates."""
        best_match = None
        best_score = 0.0
        best_reasons = []

        for candidate in candidates:
            entity_id = entity.get('id', '')
            candidate_id = candidate.get('id', '')
            if entity_id and candidate_id and entity_id == candidate_id:
                continue

            score, reasons = self.calculate_similarity(entity, candidate)

            if score >= self.threshold and score > best_score:
                best_score = score
                best_match = candidate
                best_reasons = reasons

        if best_match:
            return EntityMatch(
                entity=entity,
                canonical=best_match,
                similarity=best_score,
                match_reasons=best_reasons,
            )

        return None

    def deduplicate(
        self,
        new_entities: List[Dict[str, Any]],
        existing_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> DeduplicationResult:
        """Deduplicate entities, optionally against existing entities."""
        if not new_entities:
            return DeduplicationResult(
                merged_entities=[],
                remap={},
                duplicates_found=0,
                original_count=0,
                final_count=0,
                duplicate_pairs=[],
            )

        original_count = len(new_entities)
        remap: Dict[str, str] = {}
        duplicate_pairs: List[Tuple[str, str, float]] = []

        all_entities = list(existing_entities) if existing_entities else []
        merged_entities: List[Dict[str, Any]] = []

        self.logger.info(
            f"Deduplicating {len(new_entities)} entities "
            f"(existing: {len(all_entities)}, threshold: {self.threshold})"
        )

        for entity in new_entities:
            entity_id = entity.get('id', '')
            match = self.find_best_match(entity, all_entities)

            if match:
                canonical_id = match.canonical.get('id', '')
                if entity_id and canonical_id:
                    remap[entity_id] = canonical_id
                    duplicate_pairs.append((entity_id, canonical_id, match.similarity))
                    self._merge_metadata(match.canonical, entity)
            else:
                merged_entities.append(entity)
                all_entities.append(entity)

        duplicates_found = len(remap)
        final_count = len(merged_entities)

        self.logger.info(
            f"Deduplication complete: {original_count} -> {final_count} "
            f"({duplicates_found} duplicates)"
        )

        return DeduplicationResult(
            merged_entities=merged_entities,
            remap=remap,
            duplicates_found=duplicates_found,
            original_count=original_count,
            final_count=final_count,
            duplicate_pairs=duplicate_pairs,
        )

    def _merge_metadata(
        self,
        canonical: Dict[str, Any],
        duplicate: Dict[str, Any],
    ) -> None:
        """Merge metadata from duplicate into canonical entity."""
        canonical_meta = canonical.get('metadata', {})
        duplicate_meta = duplicate.get('metadata', {})

        if 'source_ids' not in canonical_meta:
            canonical_meta['source_ids'] = []

        dup_source = duplicate.get('source_id', duplicate_meta.get('source_id', ''))
        if dup_source and dup_source not in canonical_meta['source_ids']:
            canonical_meta['source_ids'].append(dup_source)

        dup_conf = duplicate.get('confidence', 0.0)
        can_conf = canonical.get('confidence', 0.0)
        if dup_conf > can_conf:
            canonical['confidence'] = dup_conf

        if 'aliases' not in canonical_meta:
            canonical_meta['aliases'] = []

        dup_name = duplicate.get('name', duplicate.get('text', ''))
        can_name = canonical.get('name', canonical.get('text', ''))
        if dup_name and dup_name != can_name and dup_name not in canonical_meta['aliases']:
            canonical_meta['aliases'].append(dup_name)

        canonical['metadata'] = canonical_meta

    def remap_relations(
        self,
        relations: List[Dict[str, Any]],
        remap: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Update relation source/target IDs using remap dict."""
        if not remap:
            return relations

        updated = []
        remapped_count = 0

        for rel in relations:
            new_rel = rel.copy()

            source_id = rel.get('source_id', rel.get('source', ''))
            if source_id in remap:
                new_rel['source_id'] = remap[source_id]
                new_rel['source'] = remap[source_id]
                remapped_count += 1

            target_id = rel.get('target_id', rel.get('target', ''))
            if target_id in remap:
                new_rel['target_id'] = remap[target_id]
                new_rel['target'] = remap[target_id]
                remapped_count += 1

            updated.append(new_rel)

        if remapped_count > 0:
            self.logger.debug(f"Remapped {remapped_count} relation endpoints")

        return updated

    def clear_cache(self) -> None:
        """Clear canonical form cache."""
        self._canonical_cache.clear()

    def merge_by_canonical_slug(
        self,
        entities: List[Dict[str, Any]],
        use_global_id: bool = True,
    ) -> DeduplicationResult:
        """
        Merge entities by canonical slug (LightRAG-style).

        Entity IDs follow format "{document_id}:{slug}" where slug
        is already a canonical form.
        """
        if not entities:
            return DeduplicationResult(
                merged_entities=[],
                remap={},
                duplicates_found=0,
                original_count=0,
                final_count=0,
                duplicate_pairs=[],
            )

        original_count = len(entities)
        remap: Dict[str, str] = {}
        duplicate_pairs: List[Tuple[str, str, float]] = []

        slug_groups: Dict[str, List[Dict[str, Any]]] = {}

        for entity in entities:
            entity_id = entity.get('id', '')
            if not entity_id:
                continue

            parts = entity_id.rsplit(':', 1)
            slug = parts[1] if len(parts) == 2 else entity_id

            if slug not in slug_groups:
                slug_groups[slug] = []
            slug_groups[slug].append(entity)

        merged_entities: List[Dict[str, Any]] = []

        for slug, group in slug_groups.items():
            if len(group) == 1:
                entity = group[0]
                if use_global_id:
                    old_id = entity.get('id', '')
                    entity = entity.copy()
                    entity['id'] = slug
                    if old_id != slug:
                        remap[old_id] = slug
                merged_entities.append(entity)
            else:
                canonical = self._merge_entity_group(group, slug, use_global_id)
                merged_entities.append(canonical)

                canonical_id = canonical.get('id', slug)
                for entity in group:
                    old_id = entity.get('id', '')
                    if old_id and old_id != canonical_id:
                        remap[old_id] = canonical_id
                        duplicate_pairs.append((old_id, canonical_id, 1.0))

        duplicates_found = original_count - len(merged_entities)

        self.logger.info(
            f"Slug-based merge: {original_count} -> {len(merged_entities)} "
            f"({duplicates_found} merged)"
        )

        return DeduplicationResult(
            merged_entities=merged_entities,
            remap=remap,
            duplicates_found=duplicates_found,
            original_count=original_count,
            final_count=len(merged_entities),
            duplicate_pairs=duplicate_pairs,
        )

    def _merge_entity_group(
        self,
        group: List[Dict[str, Any]],
        slug: str,
        use_global_id: bool = True,
    ) -> Dict[str, Any]:
        """Merge a group of entities with the same slug."""
        canonical = group[0].copy()
        canonical['metadata'] = canonical.get('metadata', {}).copy()

        if use_global_id:
            canonical['id'] = slug

        source_ids: Set[str] = set()
        aliases: Set[str] = set()
        document_ids: Set[str] = set()
        max_confidence = canonical.get('confidence', 0.0)

        first_meta = group[0].get('metadata', {})
        if first_meta.get('source_id'):
            source_ids.add(first_meta['source_id'])
        if first_meta.get('document_id'):
            document_ids.add(first_meta['document_id'])

        for entity in group[1:]:
            entity_meta = entity.get('metadata', {})

            if entity_meta.get('source_id'):
                source_ids.add(entity_meta['source_id'])

            if entity_meta.get('document_id'):
                document_ids.add(entity_meta['document_id'])

            entity_name = entity.get('name', entity.get('text', ''))
            canonical_name = canonical.get('name', canonical.get('text', ''))
            if entity_name and entity_name != canonical_name:
                aliases.add(entity_name)

            entity_conf = entity.get('confidence', 0.0)
            if entity_conf > max_confidence:
                max_confidence = entity_conf
                canonical['name'] = entity_name

        canonical['confidence'] = max_confidence
        canonical['metadata']['source_ids'] = sorted(source_ids)
        canonical['metadata']['document_ids'] = sorted(document_ids)
        canonical['metadata']['aliases'] = sorted(aliases)
        canonical['metadata']['merge_count'] = len(group)

        return canonical


def deduplicate_entities(
    entities: List[Dict[str, Any]],
    threshold: float = 0.98,
    existing: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Quick deduplication function."""
    dedup = LegalEntityDeduplicator(threshold=threshold)
    result = dedup.deduplicate(entities, existing)
    return result.merged_entities, result.remap


def merge_entities_by_slug(
    entities: List[Dict[str, Any]],
    use_global_id: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Merge entities by canonical slug (LightRAG-style)."""
    dedup = LegalEntityDeduplicator()
    result = dedup.merge_by_canonical_slug(entities, use_global_id=use_global_id)
    return result.merged_entities, result.remap
