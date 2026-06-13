"""Generate KG relationships from entity co-occurrence in articles.

Focus on creating edges between entities that co-occur in the same article,
prioritizing articles that are frequently missed in benchmark.
This enables PPR score propagation (currently 0 relationships → PPR is dead).
"""
import json
from collections import defaultdict
from itertools import combinations

kg = json.load(open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8"))
ents = kg.get("entities", [])
print(f"Entities: {len(ents)}, Existing relationships: {len(kg.get('relationships', []))}")

# Build article→entities mapping
article_to_entities = defaultdict(list)
entity_id_to_obj = {}
for e in ents:
    eid = e.get("id", "")
    entity_id_to_obj[eid] = e
    meta = e.get("metadata", {})
    source_ids = meta.get("source_ids", [])
    if isinstance(source_ids, str):
        source_ids = [source_ids]
    source_id = meta.get("source_id", "")
    if source_id and source_id not in source_ids:
        source_ids.append(source_id)
    for sid in source_ids:
        article_to_entities[sid].append(eid)

print(f"Articles with entities: {len(article_to_entities)}")

# Generate co-occurrence relationships
# For each article, connect entities that co-occur
# Limit: max 10 entities per article to avoid explosion
relationships = []
seen_pairs = set()

HUB_ENTITIES = {
    "doanh-nghiep", "nghi-dinh-nay", "luat-nay", "ca-nhan", "chinh-phu",
    "nghi-dinh", "nha-nuoc"
}

for article_id, entity_ids in article_to_entities.items():
    # Skip hub-only articles and filter out hub entities
    non_hub = [eid for eid in entity_ids if eid not in HUB_ENTITIES]
    if len(non_hub) < 2:
        continue

    # Limit to first 8 entities per article to control graph size
    sample = non_hub[:8]

    for e1, e2 in combinations(sample, 2):
        pair = tuple(sorted([e1, e2]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        relationships.append({
            "source": e1,
            "target": e2,
            "type": "LIÊN_QUAN_ĐẾN",
            "confidence": 0.8,
        })

print(f"Generated {len(relationships)} co-occurrence relationships")

# Add cross-document reference relationships
# Connect entities from NĐ articles to corresponding Luật articles
CROSS_DOC_REFS = {
    # NĐ 01/2021 references Luật DN 59/2020
    "01-2021-ND": "59-2020-QH14",
    # NĐ 168/2025 references Luật DN 59/2020
    "168-2025-ND": "59-2020-QH14",
    # NĐ 168/2024 references Luật ATGT 36/2024
    "168-2024-ND-CP": "36-2024-QH15",
    # NĐ 100/2019 references Luật ATGT 36/2024
    "100-2019-ND-CP": "36-2024-QH15",
}

cross_ref_count = 0
for nd_doc, luat_doc in CROSS_DOC_REFS.items():
    # Find articles from both docs
    nd_articles = {aid: eids for aid, eids in article_to_entities.items() if aid.startswith(nd_doc)}
    luat_articles = {aid: eids for aid, eids in article_to_entities.items() if aid.startswith(luat_doc)}

    # Connect entities that appear in BOTH documents
    nd_entity_set = set()
    for eids in nd_articles.values():
        nd_entity_set.update(eids)

    luat_entity_set = set()
    for eids in luat_articles.values():
        luat_entity_set.update(eids)

    shared = (nd_entity_set & luat_entity_set) - HUB_ENTITIES
    # For each shared entity, add THAM_CHIẾU relationships to Luật-only entities
    for shared_eid in shared:
        # Find Luật articles containing this entity
        for luat_aid, luat_eids in luat_articles.items():
            if shared_eid in luat_eids:
                # Connect shared entity to other entities in same Luật article
                luat_non_hub = [eid for eid in luat_eids if eid not in HUB_ENTITIES and eid != shared_eid]
                for target_eid in luat_non_hub[:3]:
                    pair = tuple(sorted([shared_eid, target_eid]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        relationships.append({
                            "source": shared_eid,
                            "target": target_eid,
                            "type": "THAM_CHIẾU",
                            "confidence": 0.7,
                        })
                        cross_ref_count += 1

print(f"Added {cross_ref_count} cross-document reference relationships")
print(f"Total relationships: {len(relationships)}")

# Save
kg["relationships"] = relationships
with open("data/kg_enhanced/legal_kg.json", "w", encoding="utf-8") as f:
    json.dump(kg, f, ensure_ascii=False, indent=2)
print("Saved to legal_kg.json")
