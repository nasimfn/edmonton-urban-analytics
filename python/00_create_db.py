import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"

db = sqlite3.connect(DB_PATH)
db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
db.commit()
db.close()

print("Database created successfully.")
