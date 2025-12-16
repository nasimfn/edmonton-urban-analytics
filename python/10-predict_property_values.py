import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"
OUT_DIR = ROOT / "ml_outputs"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_N = 120_000      # keep it memory-safe on most laptops
TEST_SIZE = 0.20
RANDOM_STATE = 42

# --- 1) Load modeling data
# We join properties to neighborhood-level permit features.
# NOTE: This does NOT do address matching; it's neighborhood-level features.
QUERY = """
WITH permit_features AS (
  SELECT
    neighborhood,
    SUM(permits_count)                        AS permits_total,
    AVG(permits_count)                        AS permits_avg_monthly,
    SUM(COALESCE(total_construction_value,0))  AS permits_total_value,
    AVG(avg_construction_value)               AS permits_avg_value,
    AVG(CASE WHEN avg_construction_value IS NULL THEN 1.0 ELSE 0.0 END) AS permits_value_missing_rate
  FROM v_neighborhood_monthly
  GROUP BY neighborhood
)
SELECT
  p.account_number,
  p.neighborhood,
  p.neighborhood_id,
  p.ward,
  p.tax_class,
  p.garage,
  p.latitude,
  p.longitude,
  p.assessed_value,

  f.permits_total,
  f.permits_avg_monthly,
  f.permits_total_value,
  f.permits_avg_value,
  f.permits_value_missing_rate

FROM properties p
LEFT JOIN permit_features f
  ON p.neighborhood = f.neighborhood
"""

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(QUERY, conn)

print(f"Loaded properties rows: {len(df):,}")

# --- 2) Basic filtering to remove obvious extremes (optional but helps stability)
df = df[df["assessed_value"].notna()]
df = df[(df["assessed_value"] > 0) & (df["assessed_value"] < 50_000_000)]
print(f"After basic filtering (0 < value < 50M): {len(df):,}")

# --- 3) Show missingness (so you understand where NaNs come from)
nulls = df.isna().sum().sort_values(ascending=False)
print("\nNulls per feature column (top 12):")
print(nulls.head(12))

# --- 4) Sample to avoid memory errors (you hit this before)
if len(df) > SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=RANDOM_STATE)
print(f"\nSampled to {len(df):,} rows for modeling (memory-safe).")

# --- 5) Create target (log transform helps skewed housing values)
y = np.log1p(df["assessed_value"].values)

# Feature set (keep it simple + strong)
feature_cols_num = [
    "latitude", "longitude",
    "permits_total", "permits_avg_monthly",
    "permits_total_value", "permits_avg_value",
    "permits_value_missing_rate",
]
feature_cols_cat = [
    "tax_class", "garage", "ward",
    # neighborhood_id is sometimes missing; treat as categorical or drop.
    # We'll keep it as categorical because it can capture area structure.
    "neighborhood_id",
]

X = df[feature_cols_num + feature_cols_cat].copy()

# --- 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# --- 7) Preprocessing (THIS fixes your NaN crash)
# - Numeric: median imputation
# - Categorical: most_frequent imputation + OneHot
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
)

# Model that is strong + fast and works well on tabular data
# (Also tolerates non-linearities better than linear regression)
model = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE,
    max_depth=6,
    learning_rate=0.06,
    max_iter=300,
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# --- 8) Fit + evaluate on log target
pipe.fit(X_train, y_train)
pred_log = pipe.predict(X_test)

mae_log = mean_absolute_error(y_test, pred_log)
rmse_log = mean_squared_error(y_test, pred_log, squared=False)
r2_log = r2_score(y_test, pred_log)

# Baseline: median predictor (log space)
baseline_pred = np.full_like(y_test, np.median(y_train))
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
baseline_r2 = r2_score(y_test, baseline_pred)

print("\nAccuracy (target is log(assessed_value)):")
print(pd.DataFrame([
    {"model": "baseline_median_log", "MAE": baseline_mae, "RMSE": baseline_rmse, "R2": baseline_r2, "n_test": len(y_test)},
    {"model": "hgb_log_target", "MAE": mae_log, "RMSE": rmse_log, "R2": r2_log, "n_test": len(y_test)},
]))

# --- 9) Optional: “approx dollars” view for intuition (not a perfect metric)
# Back-transform both y and pred, then compute MAE in dollars.
y_test_dollars = np.expm1(y_test)
pred_dollars = np.expm1(pred_log)

mae_dollars = mean_absolute_error(y_test_dollars, pred_dollars)
rmse_dollars = mean_squared_error(y_test_dollars, pred_dollars, squared=False)
r2_dollars = r2_score(y_test_dollars, pred_dollars)

print("\nApprox error in dollars (back-transformed):")
print(pd.DataFrame([
    {"model": "hgb_dollars_approx", "MAE": mae_dollars, "RMSE": rmse_dollars, "R2": r2_dollars, "n_test": len(y_test_dollars)},
]))

# --- 10) Save a sample output for Power BI / portfolio
out = X_test.copy()
out["assessed_value_true"] = np.expm1(y_test)
out["assessed_value_pred"] = np.expm1(pred_log)

sample_path = OUT_DIR / "property_value_predictions_sample.csv"
out.sample(5000, random_state=RANDOM_STATE).to_csv(sample_path, index=False)
print(f"\nSaved prediction sample: {sample_path}")
