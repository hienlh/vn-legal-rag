"""Analyze KG entity and relationship structure."""
import json
from collections import Counter

kg = json.load(open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8"))
ents = kg.get("entities", [])
rels = kg.get("relationships", [])
print(f"Total entities: {len(ents)}")
print(f"Total relationships: {len(rels)}")
print()

# Entity types
type_counter = Counter(e.get("type", "") for e in ents)
print("Entity types:")
for t, c in type_counter.most_common():
    print(f"  {t}: {c}")
print()

# Sample entities
print("Sample entities (first 10):")
for e in ents[:10]:
    print(f"  id={e.get('id','')} type={e.get('type','')} name={str(e.get('name',''))[:80]}")
print()

# Relationship types
rel_type_counter = Counter(r.get("type", "") for r in rels)
print("Relationship types:")
for t, c in rel_type_counter.most_common():
    print(f"  {t}: {c}")
print()

# Sample relationships
print("Sample relationships:")
for r in rels[:10]:
    src = r.get("source", "")
    tgt = r.get("target", "")
    rtype = r.get("type", "")
    print(f"  {src} --[{rtype}]--> {tgt}")
print()

# Check how entities reference documents
print("Entity ID patterns (first 20):")
for e in ents[:20]:
    eid = e.get("id", "")
    print(f"  {eid}")
