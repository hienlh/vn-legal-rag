"""Analyze benchmark MISS results to identify fixable patterns."""
import json
from collections import Counter, defaultdict

d = json.load(open("results/benchmark_legal_400_full.json", "r", encoding="utf-8"))
results = d.get("results", [])
misses = [x for x in results if not x.get("hit")]

print(f"Total MISS: {len(misses)}/400\n")

# Categorize
tree_miss_kg_hit = [x for x in misses if not x.get("tree_hit") and x.get("kg_hit")]
tree_miss_kg_miss = [x for x in misses if not x.get("tree_hit") and not x.get("kg_hit")]
print(f"Tree MISS + KG HIT (ranking issue): {len(tree_miss_kg_hit)}")
print(f"Tree MISS + KG MISS (both failed): {len(tree_miss_kg_miss)}")
print()

# Document analysis
doc_counter = Counter()
for x in misses:
    for aid in x.get("expected", []):
        # Extract doc from article ID like "168-2024-ND_dieu_1"
        parts = aid.rsplit("_dieu_", 1)
        if len(parts) == 2:
            doc_counter[parts[0]] += 1
        else:
            doc_counter[aid] += 1

print("Documents with most MISS articles:")
for doc, cnt in doc_counter.most_common(20):
    print(f"  {doc}: {cnt}")
print()

# Detailed miss info
print("=" * 80)
print("DETAILED MISS ANALYSIS")
print("=" * 80)
for x in misses:
    stt = x.get("stt")
    q = x.get("question", "")[:100]
    expected = x.get("expected", [])
    retrieved = x.get("retrieved", [])[:10]
    tree_arts = x.get("tree_articles", [])
    kg_arts = x.get("kg_articles", [])
    tree_hit = x.get("tree_hit", False)
    kg_hit = x.get("kg_hit", False)

    # Check if expected is in retrieved but beyond top-10
    all_retrieved = x.get("retrieved", [])
    found_positions = {}
    for exp in expected:
        if exp in all_retrieved:
            found_positions[exp] = all_retrieved.index(exp) + 1

    print(f"\nSTT {stt}: T{'✓' if tree_hit else '✗'} K{'✓' if kg_hit else '✗'}")
    print(f"  Q: {q}")
    print(f"  Expected: {expected}")
    if found_positions:
        print(f"  Found at positions: {found_positions}")
    else:
        print(f"  NOT in retrieved at all")
    if tree_arts:
        print(f"  Tree returned: {tree_arts[:5]}")
    if kg_arts:
        print(f"  KG returned: {kg_arts[:5]}")

# Hit@K breakdown for misses only
print("\n" + "=" * 80)
print("HIT@K FOR MISS QUESTIONS (where expected was found but ranked too low)")
print("=" * 80)
position_counts = Counter()
not_found_count = 0
for x in misses:
    expected = x.get("expected", [])
    all_retrieved = x.get("retrieved", [])
    best_pos = None
    for exp in expected:
        if exp in all_retrieved:
            pos = all_retrieved.index(exp) + 1
            if best_pos is None or pos < best_pos:
                best_pos = pos
    if best_pos:
        if best_pos <= 20:
            position_counts["11-20"] += 1
        elif best_pos <= 30:
            position_counts["21-30"] += 1
        elif best_pos <= 50:
            position_counts["31-50"] += 1
        else:
            position_counts["51+"] += 1
    else:
        not_found_count += 1

print(f"  Expected found at rank 11-20: {position_counts.get('11-20', 0)}")
print(f"  Expected found at rank 21-30: {position_counts.get('21-30', 0)}")
print(f"  Expected found at rank 31-50: {position_counts.get('31-50', 0)}")
print(f"  Expected found at rank 51+: {position_counts.get('51+', 0)}")
print(f"  Expected NOT in retrieved at all: {not_found_count}")
