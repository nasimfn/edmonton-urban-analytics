import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data_raw"
DB_PATH = ROOT / "edmonton.db"

PERMITS_FILE = next(RAW_DATA.glob("*Permit*.csv"))
df = pd.read_csv(PERMITS_FILE, encoding="utf-8", engine="python")
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

df = df.rename(columns={
    "permit_date": "issued_date",
    "construction_value": "project_value",
    "neighbourhood": "neighborhood"
})
df["issued_date"] = pd.to_datetime(df["issued_date"], errors="coerce")

df["project_value"] = (
    df["project_value"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["project_value"] = pd.to_numeric(df["project_value"], errors="coerce")

df["neighborhood"] = (
    df["neighborhood"]
    .astype(str)
    .str.strip()
    .str.upper()
)

permits_clean = df[[
    "issued_date",
    "year",
    "month_number",
    "job_category",
    "work_type",
    "project_value",
    "address",
    "neighborhood",
    "latitude",
    "longitude"
]].copy()

conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM permits;")
conn.commit()

permits_clean.to_sql(
    "permits",
    conn,
    if_exists="append",
    index=False
)

conn.close()
print("Permits table loaded successfully.")
