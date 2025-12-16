import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# --------------------
# Paths
# --------------------
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"

# --------------------
# Load neighborhood-level data
# --------------------
QUERY = """
SELECT
  neighborhood,
  avg_assessed_value,
  n_properties,
  permits_count,
  total_construction_value,
  avg_construction_value
FROM v_neighborhood_monthly
"""

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(QUERY, conn)

# --------------------
# Aggregate to ONE row per neighborhood
# --------------------
neigh = (
    df.groupby("neighborhood", as_index=False)
      .agg(
          avg_assessed_value=("avg_assessed_value", "max"),
          n_properties=("n_properties", "max"),
          total_permits=("permits_count", "sum"),
          avg_monthly_permits=("permits_count", "mean"),
          total_construction_value=("total_construction_value", "sum"),
          avg_construction_value=("avg_construction_value", "mean"),
      )
)

# --------------------
# Feature engineering (handle skew)
# --------------------
for col in ["total_permits", "total_construction_value"]:
    neigh[f"log_{col}"] = np.log1p(neigh[col])

neigh["avg_construction_value"] = neigh["avg_construction_value"].fillna(0)

# --------------------
# Features & target
# --------------------
FEATURES = [
    "n_properties",
    "avg_monthly_permits",
    "log_total_permits",
    "log_total_construction_value",
    "avg_construction_value",
]

X = neigh[FEATURES]
y = neigh["avg_assessed_value"]

# --------------------
# Train / test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --------------------
# Model: Random Forest
# --------------------
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# --------------------
# Evaluation
# --------------------
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nProperty Value Prediction Results")
print("---------------------------------")
print(f"MAE  : ${mae:,.0f}")
print(f"RMSE : ${rmse:,.0f}")
print(f"R²   : {r2:.3f}")

# --------------------
# Feature importance (interpretability)
# --------------------
importances = (
    pd.Series(rf.feature_importances_, index=FEATURES)
      .sort_values(ascending=False)
)

print("\nFeature Importance:")
print(importances)
