import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data_raw"
DB_PATH = ROOT / "edmonton.db"

Property_FILE = next(RAW_DATA.glob("*Property*.csv"))
df = pd.read_csv(Property_FILE, encoding="utf-8", engine="python")
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

df = df.rename(columns={
    "neighbourhood": "neighborhood",
    "neighbourhood_id": "neighborhood_id"
})



df["neighborhood"] = (
    df["neighborhood"]
    .astype(str)
    .str.strip()
    .str.upper()
)


property_clean = df[[
    "account_number",
    "house_number",
    "street_name",
    "neighborhood_id",
    "neighborhood",
    "assessed_value",
    "tax_class",
    "garage",
    "ward",
    "latitude",
    "longitude",
    "point_location"
]].copy()

property_clean["assessed_value"] = pd.to_numeric(property_clean["assessed_value"], errors="coerce")

conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM property;")
conn.commit()

property_clean.to_sql(
    "property",
    conn,
    if_exists="append",
    index=False
)

conn.close()
print("Property table loaded successfully.")
print(f"Loaded {len(property_clean):,} property records.")
