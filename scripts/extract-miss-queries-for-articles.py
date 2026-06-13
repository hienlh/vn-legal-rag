"""Extract miss queries grouped by expected article (round 2)."""
import json
import csv

d = json.load(open("results/benchmark_run3.json", "r", encoding="utf-8"))
results = d.get("results", [])
misses = [x for x in results if not x.get("hit")]

questions = {}
with open("data/benchmark/hard-400-qa-benchmark.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        questions[str(row.get("STT", ""))] = row.get("Content", "")

arts = json.load(open("data/kg_enhanced/article_summaries.json", "r", encoding="utf-8"))

# Round-2 targets: articles NOT enriched in round 1
ROUND1 = {"59-2020-QH14:d30","59-2020-QH14:d26","59-2020-QH14:d21","59-2020-QH14:d12",
          "59-2020-QH14:d31","59-2020-QH14:d4","59-2020-QH14:d52","59-2020-QH14:d207",
          "59-2020-QH14:d45","59-2020-QH14:d209","59-2020-QH14:d46","59-2020-QH14:d79",
          "59-2020-QH14:d34","59-2020-QH14:d35","59-2020-QH14:d127","59-2020-QH14:d32",
          "59-2020-QH14:d68","168-2024-ND-CP:d18","168-2024-ND-CP:d7","168-2024-ND-CP:d6"}

from collections import Counter
ac = Counter()
for x in misses:
    for aid in x.get("expected", []):
        ac[aid] += 1

# Show NEW articles (not in round 1) first
print("##### NEW MISSED ARTICLES (not enriched in round 1) #####")
for aid, cnt in ac.most_common():
    if aid in ROUND1:
        continue
    related = [x for x in misses if aid in x.get("expected", [])]
    summ = arts.get(aid, {})
    print("=" * 90)
    print(f"ARTICLE {aid} ({cnt}x) — {summ.get('article_title','?')}")
    for x in related:
        stt = str(x.get("stt"))
        q = questions.get(stt, "")[:180].replace("\n", " ")
        print(f"  [STT {stt}] {q}")
