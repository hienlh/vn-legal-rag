"""Check if legal_kg.json has relationships."""
import json

with open("data/kg_enhanced/legal_kg.json", "r", encoding="utf-8") as f:
    content = f.read()

print(f"File size: {len(content)} chars")
idx = content.find('"relationships"')
print(f'"relationships" key at position: {idx}')
if idx > 0:
    snippet = content[idx:idx+300]
    print(f"Context: {snippet}")
else:
    print("No 'relationships' key found!")
    # Check top-level keys
    d = json.loads(content)
    print(f"Top-level keys: {list(d.keys())}")
    for k, v in d.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} items")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        else:
            print(f"  {k}: {type(v).__name__}")

# Check last 500 chars
print(f"\nLast 300 chars: {content[-300:]}")
