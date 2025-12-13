import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"
OUT = ROOT / "powerbi" / "neighborhood_monthly.csv"

OUT.parent.mkdir(exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(
        "SELECT * FROM v_neighborhood_monthly;",
        conn
    )

df.to_csv(OUT, index=False)
print(f"Exported {len(df):,} rows to {OUT}")
