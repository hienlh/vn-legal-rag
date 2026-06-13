"""Analyze which specific articles are missed most often and check if they're in KG."""
import json
from collections import Counter

d = json.load(open("results/benchmark_legal_400_full.json", "r", encoding="utf-8"))
results = d.get("results", [])
misses = [x for x in results if not x.get("hit")]

# Count most-missed articles
article_miss_counter = Counter()
for x in misses:
    for aid in x.get("expected", []):
        article_miss_counter[aid] += 1

print("=== MOST MISSED ARTICLES ===")
for aid, cnt in article_miss_counter.most_common(30):
    print(f"  {aid}: missed {cnt} times")

# Check which of these articles exist in KG entity source_ids
kg = json.load(open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8"))
ents = kg.get("entities", [])

# Build article→entity mapping
article_to_entities = {}
for e in ents:
    meta = e.get("metadata", {})
    source_ids = meta.get("source_ids", [])
    if isinstance(source_ids, str):
        source_ids = [source_ids]
    source_id = meta.get("source_id", "")
    if source_id and source_id not in source_ids:
        source_ids.append(source_id)
    for sid in source_ids:
        if sid not in article_to_entities:
            article_to_entities[sid] = []
        article_to_entities[sid].append(e.get("id", ""))

print(f"\n=== KG COVERAGE: {len(article_to_entities)} articles have entities ===")
print("\nMost-missed articles KG coverage:")
for aid, cnt in article_miss_counter.most_common(30):
    ent_count = len(article_to_entities.get(aid, []))
    status = f"{ent_count} entities" if ent_count > 0 else "NO ENTITIES IN KG"
    print(f"  {aid} (missed {cnt}x): {status}")

# Check chapter summaries for most-missed articles
cs = json.load(open("data/kg_enhanced/chapter_summaries.json", "r", encoding="utf-8"))
print("\n=== CHAPTER COVERAGE FOR 59-2020-QH14 ===")
for k in sorted(cs.keys()):
    if "59-2020-QH14" in k:
        v = cs[k]
        print(f"  {k}: {v.get('chapter_title','')}")
        print(f"    Range: {v.get('article_range','')}")

# Which missed articles map to which chapters
print("\n=== MISSED ARTICLES → CHAPTERS (59-2020-QH14) ===")
missed_59 = [(aid, cnt) for aid, cnt in article_miss_counter.most_common() if "59-2020-QH14" in aid]
for aid, cnt in missed_59:
    # Extract article number
    dieu_num = aid.split(":d")[-1] if ":d" in aid else "?"
    print(f"  {aid} (Điều {dieu_num}, missed {cnt}x)")
