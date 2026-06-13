"""Analyze KG entities to design relationships for missed articles."""
import json
from collections import defaultdict

kg = json.load(open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8"))
ents = kg.get("entities", [])
rels = kg.get("relationships", [])
print(f"Entities: {len(ents)}, Relationships: {len(rels)}")

# Build entity→articles and article→entities mappings
entity_to_articles = defaultdict(set)
article_to_entities = defaultdict(set)
for e in ents:
    meta = e.get("metadata", {})
    eid = e.get("id", "")
    source_ids = meta.get("source_ids", [])
    if isinstance(source_ids, str):
        source_ids = [source_ids]
    source_id = meta.get("source_id", "")
    if source_id and source_id not in source_ids:
        source_ids.append(source_id)
    for sid in source_ids:
        entity_to_articles[eid].add(sid)
        article_to_entities[sid].add(eid)

# Focus on most-missed articles
MISSED_ARTICLES = [
    "168-2024-ND-CP:d6", "168-2024-ND-CP:d47", "168-2024-ND-CP:d14",
    "168-2024-ND-CP:d18", "168-2024-ND-CP:d32",
    "36-2024-QH15:d36", "36-2024-QH15:d9", "36-2024-QH15:d11",
    "59-2020-QH14:d21", "59-2020-QH14:d30", "59-2020-QH14:d26",
    "59-2020-QH14:d31", "59-2020-QH14:d47", "59-2020-QH14:d45",
]

print("\n=== SHARED ENTITIES BETWEEN MISSED ARTICLES ===")
for i, art1 in enumerate(MISSED_ARTICLES):
    ents1 = article_to_entities.get(art1, set())
    if not ents1:
        print(f"\n{art1}: NO ENTITIES")
        continue
    for art2 in MISSED_ARTICLES[i+1:]:
        ents2 = article_to_entities.get(art2, set())
        shared = ents1 & ents2
        if len(shared) >= 3:
            print(f"\n{art1} <-> {art2}: {len(shared)} shared entities")
            print(f"  Shared: {list(shared)[:10]}")

# Check which entities appear in many articles (hub entities)
print("\n=== HUB ENTITIES (appearing in 10+ articles) ===")
hubs = [(eid, len(arts)) for eid, arts in entity_to_articles.items() if len(arts) >= 10]
hubs.sort(key=lambda x: -x[1])
for eid, cnt in hubs[:20]:
    ent_obj = next((e for e in ents if e.get("id") == eid), {})
    name = ent_obj.get("name", eid)
    etype = ent_obj.get("type", "?")
    print(f"  {eid} ({etype}): {cnt} articles — {name}")

# Potential relationships: entities shared between missed article and its commonly-retrieved-instead article
print("\n=== ENTITY OVERLAP: missed 59-2020-QH14 vs wrongly-returned 01-2021-ND ===")
# Check if 59-2020-QH14:d21 and 01-2021-ND:d21 share entities
for dieu in [21, 26, 30, 31, 45, 47]:
    art_59 = f"59-2020-QH14:d{dieu}"
    art_01 = f"01-2021-ND:d{dieu}"
    ents_59 = article_to_entities.get(art_59, set())
    ents_01 = article_to_entities.get(art_01, set())
    shared = ents_59 & ents_01
    only_59 = ents_59 - ents_01
    only_01 = ents_01 - ents_59
    print(f"\n  Điều {dieu}: 59-QH14 has {len(ents_59)} ents, 01-ND has {len(ents_01)} ents")
    print(f"    Shared: {len(shared)}, Only in 59: {len(only_59)}, Only in 01: {len(only_01)}")
    if only_59:
        print(f"    Unique to 59: {list(only_59)[:5]}")
