import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"
OUT_CSV = ROOT / "powerbi" / "neighborhood_clusters.csv"
OUT_CSV.parent.mkdir(exist_ok=True)


# 1) Load data from the SQL view

QUERY = """
SELECT
  neighborhood,
  year,
  month_number,
  permits_count,
  total_construction_value,
  avg_construction_value,
  n_properties,
  avg_assessed_value
FROM v_neighborhood_monthly
"""

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(QUERY, conn)


# 2) Aggregate to ONE ROW per neighborhood

# Notes:
# - I treat missing project values as 0 for totals
# - I'll also create "missing rate" as a feature (useful signal)
df["total_construction_value"] = df["total_construction_value"].fillna(0)


neigh = (
    df.groupby("neighborhood", as_index=False)
      .agg(
          months_observed=("permits_count", "size"),
          total_permits=("permits_count", "sum"),
          avg_monthly_permits=("permits_count", "mean"),
          total_construction_value=("total_construction_value", "sum"),
          avg_monthly_construction_value=("total_construction_value", "mean"),
          # avg of avg can be noisy; still useful after log transform
          avg_permit_size=("avg_construction_value", "mean"),
          n_properties=("n_properties", "max"),
          avg_assessed_value=("avg_assessed_value", "max"),
      )
)

# Feature: how often permit-size was missing (proxy for data quality / permit types)
missing_rate = (
    df.assign(missing_avg_val=df["avg_construction_value"].isna().astype(int))
      .groupby("neighborhood", as_index=False)["missing_avg_val"].mean()
      .rename(columns={"missing_avg_val": "avg_value_missing_rate"})
)
neigh = neigh.merge(missing_rate, on="neighborhood", how="left")

# Keep only neighborhoods that exist in properties (recommended)
neigh = neigh.dropna(subset=["n_properties", "avg_assessed_value"])

# 3) Feature engineering (reduce skew with log1p)

# log1p(x) = log(1+x), safe for zeros, common for skewed money/count data
for col in ["total_permits", "total_construction_value", "avg_monthly_construction_value", "avg_assessed_value"]:
    neigh[f"log_{col}"] = np.log1p(neigh[col].clip(lower=0))


# 4) Select features for clustering

FEATURES = [
    "avg_monthly_permits",
    "log_total_permits",
    "log_total_construction_value",
    "avg_value_missing_rate",
    "log_avg_assessed_value",
    "n_properties",
]

X = neigh[FEATURES].copy()

# If something is still missing, fill conservatively
X["avg_permit_size"] = neigh["avg_permit_size"]  # optional, but often noisy
X["avg_permit_size"] = X["avg_permit_size"].fillna(X["avg_permit_size"].median())
FEATURES_FINAL = FEATURES + ["avg_permit_size"]

X = X[FEATURES_FINAL]

# ---- DEBUG: find NaNs before scaling ----
na_counts = X.isna().sum().sort_values(ascending=False)
print("\nNaN counts in features:")
print(na_counts[na_counts > 0])

if X.isna().any().any():
    bad_rows = X[X.isna().any(axis=1)].copy()
    print(f"\nRows with NaNs: {len(bad_rows)}")
    print("Example rows (first 10):")
    print(pd.concat([neigh.loc[bad_rows.index, ["neighborhood"]], bad_rows], axis=1).head(10))

# ---- FIX: impute remaining missing values ----
# For counts/sizes: fill with 0
zero_fill = ["n_properties"]
for c in zero_fill:
    if c in X.columns:
        X[c] = X[c].fillna(0)

# For log features / rates: fill with median (robust)
median_fill = ["avg_value_missing_rate", "log_avg_assessed_value",
               "log_total_permits", "log_total_construction_value",
               "log_avg_monthly_construction_value", "avg_monthly_permits"]
for c in median_fill:
    if c in X.columns:
        X[c] = X[c].fillna(X[c].median())

# Safety check
if X.isna().any().any():
    raise ValueError("Still have NaNs after imputation. Check feature engineering.")


# 5) Scale features (KMeans needs this)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 6) Choose K 


inertias = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias[k] = km.inertia_

# Pick a default K (you can change this)
K = 5

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
neigh["cluster"] = kmeans.fit_predict(X_scaled)

# Label clusters by a human-friendly ordering (e.g., by avg assessed value)
cluster_order = (
    neigh.groupby("cluster")["avg_assessed_value"].mean()
    .sort_values()
    .reset_index()
)
mapping = {row["cluster"]: i for i, row in cluster_order.iterrows()}
neigh["cluster_ranked"] = neigh["cluster"].map(mapping)


# 7) Save results to SQLite + CSV for Power BI

with sqlite3.connect(DB_PATH) as conn:
    conn.execute("DROP TABLE IF EXISTS neighborhood_clusters;")
    neigh.to_sql("neighborhood_clusters", conn, index=False)

neigh.to_csv(OUT_CSV, index=False)

print("Clustering complete")
print(f"- Neighborhoods clustered: {len(neigh):,}")
print(f"- Saved table: neighborhood_clusters (in {DB_PATH.name})")
print(f"- Saved CSV: {OUT_CSV}")
print("\nInertia by K (lower is better; look for an elbow):")
for k, v in inertias.items():
    print(f"  K={k}: {v:,.2f}")
print(f"\nUsed K={K}. You can change K in the script and rerun.")
