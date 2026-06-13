"""Check if benchmark article IDs exist in the rebuilt database."""
import json
import csv
import sqlite3

# 1. Get all expected article IDs from benchmark
with open("data/benchmark/hard-400-qa-benchmark.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_expected = set()
    for row in reader:
        aids = row.get("Article_IDs", "") or row.get("article_ids", "")
        # Split by semicolon AND comma
        for aid in aids.replace(";", ",").split(","):
            aid = aid.strip()
            if aid:
                all_expected.add(aid)

print(f"Unique expected article IDs in benchmark: {len(all_expected)}")

# 2. Get all article IDs from rebuilt database
conn = sqlite3.connect("data/legal_docs.db")
cursor = conn.execute("SELECT id FROM articles")
db_articles = set(row[0] for row in cursor.fetchall())
conn.close()
print(f"Articles in rebuilt DB: {len(db_articles)}")

# 3. Get all article IDs from KG entities
kg = json.load(open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8"))
kg_articles = set()
for e in kg.get("entities", []):
    meta = e.get("metadata", {})
    for sid in meta.get("source_ids", []):
        kg_articles.add(sid)
    sid = meta.get("source_id", "")
    if sid:
        kg_articles.add(sid)
print(f"Articles in KG: {len(kg_articles)}")

# 4. Check matches
found_db = all_expected & db_articles
found_kg = all_expected & kg_articles
missing_db = all_expected - db_articles
missing_kg = all_expected - kg_articles

print(f"\nBenchmark IDs found in DB: {len(found_db)}/{len(all_expected)}")
print(f"Benchmark IDs found in KG: {len(found_kg)}/{len(all_expected)}")

if missing_db:
    print(f"\nMISSING from DB ({len(missing_db)}):")
    for m in sorted(missing_db)[:20]:
        print(f"  {m}")

# 5. Check format differences
print("\n=== SAMPLE DB article IDs ===")
for aid in sorted(db_articles)[:10]:
    print(f"  {aid}")

print("\n=== SAMPLE expected article IDs ===")
for aid in sorted(all_expected)[:10]:
    print(f"  {aid}")

# 6. Check how eval script parses article IDs
print("\n=== Format check ===")
# Check if there's a mapping function in eval script
import re
sample_expected = list(all_expected)[:5]
sample_db = list(db_articles)[:5]
for s in sample_expected:
    # Check format: doc_id:dN
    match = re.match(r"(.+):d(\d+)", s)
    if match:
        print(f"  Expected: {s} → doc={match.group(1)}, dieu={match.group(2)}")
for s in sample_db:
    match = re.match(r"(.+):d(\d+)", s)
    if match:
        print(f"  DB:       {s} → doc={match.group(1)}, dieu={match.group(2)}")
