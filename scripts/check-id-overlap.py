"""Check article ID overlap between benchmark, DB, and KG."""
import json, csv, sqlite3

# Benchmark expected IDs
with open("data/benchmark/hard-400-qa-benchmark.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_expected = set()
    for row in reader:
        aids = row.get("Article_IDs", "") or row.get("article_ids", "")
        for aid in aids.replace(";", ",").split(","):
            aid = aid.strip()
            if aid:
                all_expected.add(aid)

# DB article IDs
conn = sqlite3.connect("data/legal_docs.db")
db_articles = set(r[0] for r in conn.execute("SELECT id FROM legal_articles").fetchall())
conn.close()

# KG article IDs (from entity source_ids)
kg = json.load(open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8"))
kg_articles = set()
for e in kg.get("entities", []):
    meta = e.get("metadata", {})
    for sid in meta.get("source_ids", []):
        kg_articles.add(sid)
    sid = meta.get("source_id", "")
    if sid:
        kg_articles.add(sid)

print(f"Benchmark expected: {len(all_expected)} unique article IDs")
print(f"DB articles: {len(db_articles)}")
print(f"KG articles: {len(kg_articles)}")

found_db = all_expected & db_articles
found_kg = all_expected & kg_articles
missing_db = all_expected - db_articles
missing_kg = all_expected - kg_articles

print(f"\nIn DB: {len(found_db)}/{len(all_expected)} ({100*len(found_db)/len(all_expected):.1f}%)")
print(f"In KG: {len(found_kg)}/{len(all_expected)} ({100*len(found_kg)/len(all_expected):.1f}%)")

if missing_db:
    print(f"\nMISSING from DB ({len(missing_db)}):")
    for m in sorted(missing_db):
        print(f"  {m}")
