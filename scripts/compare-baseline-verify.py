"""Compare first 20 STTs in baseline vs verify."""
import json

base = json.load(open("results/benchmark_legal_400_full.json", "r", encoding="utf-8"))
verify = json.load(open("results/benchmark_verify_baseline.json", "r", encoding="utf-8"))
base_by_stt = {str(r["stt"]): r for r in base["results"]}

changed_count = 0
for r in verify["results"]:
    stt = str(r["stt"])
    b = base_by_stt.get(stt, {})
    bh = "HIT" if b.get("hit") else "MISS"
    vh = "HIT" if r.get("hit") else "MISS"
    bt = int(b.get("tree_hit", 0))
    vt = int(r.get("tree_hit", 0))
    bk = int(b.get("kg_hit", 0))
    vk = int(r.get("kg_hit", 0))
    changed = " <<<" if bh != vh else ""
    if changed:
        changed_count += 1
    print(f"STT {stt:4s}: base={bh:4s} verify={vh:4s}  T:{bt}->{vt}  K:{bk}->{vk}{changed}")

print(f"\nChanged: {changed_count}/{len(verify['results'])}")

# Check if chapter summaries were properly reverted
cs = json.load(open("data/kg_enhanced/chapter_summaries.json", "r", encoding="utf-8"))
# Check c2 description length
c2 = cs.get("59-2020-QH14:c2", {})
print(f"\nc2 description length: {len(c2.get('description', ''))}")
print(f"c2 keywords length: {len(c2.get('keywords', ''))}")
print(f"c2 desc first 100: {c2.get('description', '')[:100]}")
