import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"

def run(sql: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)

print("\nA) Row count")
print(run("SELECT COUNT(*) AS n_rows FROM permits;"))

print("\nB) Date range")
print(run("SELECT MIN(issued_date) AS min_date, MAX(issued_date) AS max_date FROM permits;"))

print("\nC) Missingness in key fields")
print(run("""
SELECT
  SUM(CASE WHEN issued_date IS NULL THEN 1 ELSE 0 END) AS missing_date,
  SUM(CASE WHEN project_value IS NULL THEN 1 ELSE 0 END) AS missing_value,
  SUM(CASE WHEN neighborhood IS NULL OR neighborhood = '' THEN 1 ELSE 0 END) AS missing_neighborhood,
  COUNT(*) AS total
FROM permits;
"""))

print("\nD) Top 10 neighbourhoods by permit count")
print(run("""
SELECT neighborhood, COUNT(*) AS n
FROM permits
GROUP BY neighborhood
ORDER BY n DESC
LIMIT 10;
"""))
