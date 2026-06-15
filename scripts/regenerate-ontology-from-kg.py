"""Regenerate the Vietnamese legal ontology from the knowledge graph via LLM.

Loads legal_kg.json, normalizes the relations key, drives the LLM-based
LegalOntologyGenerator (prompt caps hierarchy at 4 levels with Vietnamese
labels), and exports to data/kg_enhanced/ontology.json.

The ontology is the conceptual layer (entity classes) described in the thesis;
it is not used at query time, so this does not affect benchmark numbers.
"""
import json
from importlib import import_module

import yaml

from vn_legal_rag.utils import create_llm_provider

LegalOntologyGenerator = import_module(
    "vn_legal_rag.offline.legal-ontology-generator"
).LegalOntologyGenerator

CONFIG = "config/default.yaml"
KG_PATH = "data/kg_enhanced/legal_kg.json"
OUT_PATH = "data/kg_enhanced/ontology.json"

cfg = yaml.safe_load(open(CONFIG, "r", encoding="utf-8"))
llm = cfg.get("llm", {})

kg = json.load(open(KG_PATH, "r", encoding="utf-8"))
# Generator reads kg["relationships"]; our KG stores them under "relations".
if "relationships" not in kg:
    kg["relationships"] = kg.get("relations", [])
print(f"KG: {len(kg.get('entities', []))} entities, {len(kg['relationships'])} relations")

provider = create_llm_provider(
    provider=llm.get("provider", "anthropic"),
    model=llm.get("model", "claude-sonnet-4-20250514"),
    base_url=llm.get("base_url"),
    cache_db=None,  # disable cache: force fresh generation each run
)

gen = LegalOntologyGenerator(
    base_uri="https://hienle.tech/legal/ontology#",
    llm_provider=provider,
    llm_model=llm.get("model", "claude-sonnet-4-20250514"),
    use_llm=True,
    min_occurrences=5,
)

ontology = gen.generate_from_kg(kg, name="VietnameseLegalOntology")
ontology.to_json_file(OUT_PATH)

# Report structure
d = json.load(open(OUT_PATH, "r", encoding="utf-8"))
classes = d.get("classes", {})
classes = list(classes.values()) if isinstance(classes, dict) else classes
byname = {c.get("name"): c for c in classes}


def depth(c):
    n, p = 1, c.get("parent")
    while p and p in byname:
        n += 1
        p = byname[p].get("parent")
    return n


maxd = max((depth(c) for c in classes), default=0)
print(f"\nGenerated ontology: {len(classes)} classes, max depth {maxd}, "
      f"{len(d.get('properties', []))} properties")
print(f"Saved to {OUT_PATH}")
