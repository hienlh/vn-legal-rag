"""Check DB tables and article ID format."""
import sqlite3

conn = sqlite3.connect("data/legal_docs.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t[0] for t in tables])

for table in tables:
    name = table[0]
    count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
    cols = conn.execute(f"PRAGMA table_info([{name}])").fetchall()
    col_names = [c[1] for c in cols]
    print(f"\n{name}: {count} rows, cols={col_names}")
    if count > 0:
        sample = conn.execute(f"SELECT * FROM [{name}] LIMIT 3").fetchall()
        for row in sample:
            print(f"  {row[:3]}...")

conn.close()
