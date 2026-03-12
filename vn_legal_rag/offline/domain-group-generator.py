"""
Domain Group Generator — Clusters documents into domain groups using keyword overlap.

Reads document_summaries.json and produces domain_groups.json.
Each group contains a label, representative keywords, and member doc IDs.

Algorithm:
1. Identify anchor documents (Luật/Law = primary laws) as domain centers
2. For each non-anchor doc, compute keyword overlap with each anchor
3. Assign to the anchor with highest overlap (above threshold)
4. Docs without a matching anchor form an "uncategorized" group
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def _normalize_keyword(kw: str) -> str:
    """Lowercase and strip a keyword for matching."""
    return kw.strip().lower()


def _compute_keyword_overlap(keywords_a: List[str], keywords_b: List[str]) -> float:
    """Compute Jaccard-like overlap between two keyword lists (syllable-level)."""
    if not keywords_a or not keywords_b:
        return 0.0
    # Split compound keywords into syllables for better matching
    syllables_a = set()
    for kw in keywords_a:
        for syllable in _normalize_keyword(kw).split():
            syllables_a.add(syllable)
    syllables_b = set()
    for kw in keywords_b:
        for syllable in _normalize_keyword(kw).split():
            syllables_b.add(syllable)
    # Remove generic legal terms that appear in every document
    generic = {"phạm", "vi", "điều", "chỉnh", "đối", "tượng", "áp", "dụng",
               "giải", "thích", "từ", "ngữ", "quy", "định", "nghị", "luật"}
    syllables_a -= generic
    syllables_b -= generic
    if not syllables_a or not syllables_b:
        return 0.0
    intersection = syllables_a & syllables_b
    union = syllables_a | syllables_b
    return len(intersection) / len(union)


def generate_domain_groups(
    document_summaries_path: str,
    output_path: str,
    overlap_threshold: float = 0.08,
) -> Dict[str, Any]:
    """
    Generate domain groups from document summaries.

    Args:
        document_summaries_path: Path to document_summaries.json
        output_path: Path to write domain_groups.json
        overlap_threshold: Min keyword overlap to assign doc to an anchor's group

    Returns:
        Domain groups dict
    """
    with open(document_summaries_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    # Step 1: Identify anchor documents (primary laws)
    anchors = {}  # doc_id → keywords
    non_anchors = {}  # doc_id → keywords
    for doc_id, s in summaries.items():
        name = s.get("ten_van_ban", "").lower()
        keywords = s.get("domain_keywords", [])
        is_law = "luật" in name and "nghị định" not in name
        if is_law:
            anchors[doc_id] = keywords
        else:
            non_anchors[doc_id] = keywords

    # Step 2: Build groups around anchors
    groups = {}
    for anchor_id, anchor_kw in anchors.items():
        s = summaries[anchor_id]
        # Generate a short label from the law name
        label = s.get("ten_van_ban", anchor_id)
        # Extract core domain keywords (excluding generic ones)
        domain_kw = [kw for kw in anchor_kw[:10]
                     if _normalize_keyword(kw) not in {
                         "phạm vi điều chỉnh", "đối tượng áp dụng",
                         "giải thích từ ngữ", "quy định chung"
                     }]
        groups[anchor_id] = {
            "label": label,
            "anchor_doc_id": anchor_id,
            "domain_keywords": domain_kw[:8],
            "documents": [anchor_id],
        }

    # Step 3: Assign non-anchors to best matching anchor
    uncategorized = []
    for doc_id, doc_kw in non_anchors.items():
        best_anchor = None
        best_score = 0.0
        for anchor_id, anchor_kw in anchors.items():
            score = _compute_keyword_overlap(doc_kw, anchor_kw)
            if score > best_score:
                best_score = score
                best_anchor = anchor_id
        if best_anchor and best_score >= overlap_threshold:
            groups[best_anchor]["documents"].append(doc_id)
        else:
            uncategorized.append(doc_id)

    # Step 4: Handle uncategorized — try to match with existing group members
    still_uncategorized = []
    for doc_id in uncategorized:
        doc_kw = non_anchors[doc_id]
        best_group = None
        best_score = 0.0
        for group_id, group in groups.items():
            # Combine keywords from all group members
            group_kw = []
            for member_id in group["documents"]:
                group_kw.extend(summaries[member_id].get("domain_keywords", []))
            score = _compute_keyword_overlap(doc_kw, group_kw)
            if score > best_score:
                best_score = score
                best_group = group_id
        if best_group and best_score >= overlap_threshold:
            groups[best_group]["documents"].append(doc_id)
        else:
            still_uncategorized.append(doc_id)

    # Step 5: Create "other" group for remaining uncategorized docs
    if still_uncategorized:
        groups["other"] = {
            "label": "Khác",
            "anchor_doc_id": None,
            "domain_keywords": [],
            "documents": still_uncategorized,
        }

    # Build final output with summary for each group
    result = {"domain_groups": {}}
    for group_id, group in groups.items():
        # Collect all domain keywords from group members for the group-level overview
        all_kw = set()
        for member_id in group["documents"]:
            for kw in summaries.get(member_id, {}).get("domain_keywords", [])[:10]:
                normalized = _normalize_keyword(kw)
                if normalized not in {
                    "phạm vi điều chỉnh", "đối tượng áp dụng",
                    "giải thích từ ngữ", "quy định chung"
                }:
                    all_kw.add(kw)
        result["domain_groups"][group_id] = {
            "label": group["label"],
            "anchor_doc_id": group["anchor_doc_id"],
            "domain_keywords": list(all_kw)[:15],
            "documents": group["documents"],
            "doc_count": len(group["documents"]),
        }

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
