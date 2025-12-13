import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"
VIEWS_SQL = ROOT / "sql" / "views.sql"

with sqlite3.connect(DB_PATH) as conn:
    conn.executescript(VIEWS_SQL.read_text(encoding="utf-8"))
    conn.commit()

print("Views created successfully.")
