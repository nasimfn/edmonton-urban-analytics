import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"

def run(sql: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)

print("\nA) Row count")
print(run("SELECT COUNT(*) AS n_rows FROM properties;"))

print("\nB) Assessed value range (sanity)")
print(run("""
SELECT
  MIN(assessed_value) AS min_value,
  MAX(assessed_value) AS max_value,
  AVG(assessed_value) AS avg_value
FROM properties;
"""))

print("\nC) Missingness in key fields")
print(run("""
SELECT
  SUM(CASE WHEN assessed_value IS NULL THEN 1 ELSE 0 END) AS missing_assessed_value,
  SUM(CASE WHEN neighborhood IS NULL OR neighborhood = '' THEN 1 ELSE 0 END) AS missing_neighborhood,
  SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) AS missing_latitude,
  COUNT(*) AS total
FROM properties;
"""))

print("\nD) Top 10 neighbourhoods by average assessed value (min 200 properties)")
print(run("""
SELECT neighborhood,
       COUNT(*) AS n_props,
       AVG(assessed_value) AS avg_assessed_value
FROM properties
WHERE assessed_value IS NOT NULL
GROUP BY neighborhood
HAVING COUNT(*) >= 200
ORDER BY avg_assessed_value DESC
LIMIT 10;
"""))
