"""Check document and chapter summaries for problematic docs."""
import json

ds = json.load(open("data/kg_enhanced/document_summaries.json", "r", encoding="utf-8"))
cs = json.load(open("data/kg_enhanced/chapter_summaries.json", "r", encoding="utf-8"))

target_docs = ["59-2020-QH14", "01-2021-ND", "168-2024-ND-CP", "168-2025-ND", "36-2024-QH15"]

print("=" * 80)
print("DOCUMENT SUMMARIES")
print("=" * 80)
for doc_id in target_docs:
    for k, v in ds.items():
        if doc_id in k:
            print(f"\n--- {k} ---")
            print(f"Title: {v.get('doc_title', '')}")
            print(f"Keywords: {v.get('keywords', '')[:300]}")
            break

print("\n" + "=" * 80)
print("CHAPTER SUMMARIES FOR 59-2020-QH14 (Luat DN - most misses)")
print("=" * 80)
for k, v in sorted(cs.items()):
    if "59-2020-QH14" in k:
        print(f"\n--- {k} ---")
        print(f"Title: {v.get('chapter_title', '')}")
        print(f"Range: {v.get('article_range', '')}")
        print(f"Keywords: {v.get('keywords', '')[:300]}")

print("\n" + "=" * 80)
print("CHAPTER SUMMARIES FOR 168-2024-ND-CP (most misses)")
print("=" * 80)
for k, v in sorted(cs.items()):
    if "168-2024-ND-CP" in k:
        print(f"\n--- {k} ---")
        print(f"Title: {v.get('chapter_title', '')}")
        print(f"Range: {v.get('article_range', '')}")
        print(f"Keywords: {v.get('keywords', '')[:300]}")

# Also check KG entities for these docs
print("\n" + "=" * 80)
print("KG ENTITY COUNT PER DOCUMENT")
print("=" * 80)
kg = json.load(open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8"))
from collections import Counter
doc_entity_count = Counter()
for ent in kg.get("entities", []):
    eid = ent.get("id", "")
    for doc_id in target_docs:
        if doc_id in eid:
            doc_entity_count[doc_id] += 1
            break

for doc_id in target_docs:
    print(f"  {doc_id}: {doc_entity_count.get(doc_id, 0)} entities")

doc_rel_count = Counter()
for rel in kg.get("relationships", []):
    src = rel.get("source", "")
    tgt = rel.get("target", "")
    for doc_id in target_docs:
        if doc_id in src or doc_id in tgt:
            doc_rel_count[doc_id] += 1
            break

print()
for doc_id in target_docs:
    print(f"  {doc_id}: {doc_rel_count.get(doc_id, 0)} relationships")
