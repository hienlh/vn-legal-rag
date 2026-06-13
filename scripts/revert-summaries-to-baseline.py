"""Revert summaries to baseline: remove ALL routing tags, restore original format."""
import json

# === Fix chapter_summaries.json — remove all [TAG] prefixes ===
cs = json.load(open("data/kg_enhanced/chapter_summaries.json", "r", encoding="utf-8"))
for ch_id, ch in cs.items():
    desc = ch.get("description", "")
    # Remove any [TAG] prefix
    if desc.startswith("["):
        bracket_end = desc.find("] ")
        if bracket_end != -1:
            desc = desc[bracket_end + 2:]
            ch["description"] = desc
    kw = ch.get("keywords", "")
    ar = ch.get("article_range", "")
    # Ensure description matches original format: "article_range. Nội dung: keywords"
    if kw and ar and not desc.startswith(ar):
        ch["description"] = f"{ar}. Nội dung: {kw}"

with open("data/kg_enhanced/chapter_summaries.json", "w", encoding="utf-8") as f:
    json.dump(cs, f, ensure_ascii=False, indent=2)
print(f"Reverted {len(cs)} chapter summaries")

# === Fix document_summaries.json — remove all [TAG] prefixes and restore loai_van_ban ===
ds = json.load(open("data/kg_enhanced/document_summaries.json", "r", encoding="utf-8"))
for doc_id, doc in ds.items():
    # Remove scope prefix tags
    scope = doc.get("scope", "")
    if scope.startswith("["):
        bracket_end = scope.find("] ")
        if bracket_end != -1:
            scope = scope[bracket_end + 2:]
            doc["scope"] = scope
    # Restore loai_van_ban to original
    doc["loai_van_ban"] = "Văn bản"

with open("data/kg_enhanced/document_summaries.json", "w", encoding="utf-8") as f:
    json.dump(ds, f, ensure_ascii=False, indent=2)
print(f"Reverted {len(ds)} document summaries")
print("Done — summaries restored to baseline")
