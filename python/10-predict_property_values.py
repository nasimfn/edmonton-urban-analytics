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

# Models
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


# ----------------------------
# Config (edit these safely)
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"

# To avoid MemoryError on 32-bit Python, keep sample sizes moderate.
# You can increase later if you move to 64-bit Python.
MAX_ROWS_FOR_MODEL = 120_000   # sample from 431k
TEST_SIZE = 0.20
RANDOM_STATE = 42

MODEL_TYPE = "hgb"  # "hgb" (recommended) or "rf"

OUT_DIR = ROOT / "ml_outputs"
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, label="model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    return pd.DataFrame([{
        "model": label,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_%": mape,
        "n_test": len(y_true),
    }])


# ----------------------------
# 1) Load property table
# ----------------------------
PROPS_QUERY = """
SELECT
  account_number,
  neighborhood,
  neighborhood_id,
  ward,
  assessed_value,
  tax_class,
  garage,
  latitude,
  longitude
FROM properties
WHERE assessed_value IS NOT NULL;
"""

with sqlite3.connect(DB_PATH) as conn:
    props = pd.read_sql_query(PROPS_QUERY, conn)

print(f"Loaded properties rows: {len(props):,}")

# Optional: remove obvious junk/outliers for modeling stability
# (keep it light; you can mention this in README)
props = props[(props["assessed_value"] > 0) & (props["assessed_value"] < 50_000_000)].copy()

print(f"After basic filtering (0 < value < 50M): {len(props):,}")


# ----------------------------
# 2) Build neighborhood permits features and merge
# ----------------------------
# We aggregate permits by neighborhood across all months
PERM_FEATS_QUERY = """
SELECT
  neighborhood,
  SUM(permits_count) AS permits_total,
  AVG(permits_count) AS permits_avg_monthly,
  SUM(COALESCE(total_construction_value, 0)) AS permits_total_value,
  AVG(total_construction_value) AS permits_avg_value,
  AVG(CASE WHEN avg_construction_value IS NULL THEN 1.0 ELSE 0.0 END) AS permits_value_missing_rate
FROM v_permits_monthly
GROUP BY neighborhood;
"""

with sqlite3.connect(DB_PATH) as conn:
    perm = pd.read_sql_query(PERM_FEATS_QUERY, conn)

# Standardize join key just in case
props["neighborhood"] = props["neighborhood"].astype(str).str.strip().str.upper()
perm["neighborhood"] = perm["neighborhood"].astype(str).str.strip().str.upper()

df = props.merge(perm, on="neighborhood", how="left")

print(f"Merged modeling rows: {len(df):,}")

# Log features (helpful for skew)
for col in ["permits_total", "permits_total_value", "assessed_value"]:
    df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

# Target
y = df["log_assessed_value"].copy()

# Features
feature_cols = [
    # numeric
    "latitude", "longitude",
    "permits_total", "permits_avg_monthly",
    "permits_total_value", "permits_avg_value",
    "permits_value_missing_rate",
    "neighborhood_id",

    # categorical
    "tax_class", "garage", "ward"
]

X = df[feature_cols].copy()

# Report null counts (learning/debug)
nulls = X.isna().sum().sort_values(ascending=False)
print("\nNulls per feature column (top 12):")
print(nulls.head(12))


# ----------------------------
# 3) Sample rows to avoid memory issues
# ----------------------------
if len(X) > MAX_ROWS_FOR_MODEL:
    df_sample = df.sample(n=MAX_ROWS_FOR_MODEL, random_state=RANDOM_STATE)
    X = df_sample[feature_cols].copy()
    y = df_sample["log_assessed_value"].copy()
    print(f"\nSampled to {len(X):,} rows for modeling (memory-safe).")


# ----------------------------
# 4) Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Baseline: predict median of training (in log space)
baseline_pred = np.full(shape=len(y_test), fill_value=np.median(y_train))
results = evaluate(y_test, baseline_pred, label="baseline_median_log")


# ----------------------------
# 5) Preprocessing (impute + one-hot)
# ----------------------------
numeric_features = [
    "latitude", "longitude",
    "permits_total", "permits_avg_monthly",
    "permits_total_value", "permits_avg_value",
    "permits_value_missing_rate",
    "neighborhood_id",
]
categorical_features = ["tax_class", "garage", "ward"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)


# ----------------------------
# 6) Model
# ----------------------------
if MODEL_TYPE == "hgb":
    model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_depth=8,
        learning_rate=0.08,
        max_iter=300
    )
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model),
    ])

elif MODEL_TYPE == "rf":
    # Memory-safe RF settings for 32-bit Python:
    # - fewer trees
    # - limited depth
    # - single-thread (n_jobs=1) to reduce joblib RAM overhead
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model),
    ])
else:
    raise ValueError("MODEL_TYPE must be 'hgb' or 'rf'.")


# ----------------------------
# 7) Fit + evaluate
# ----------------------------
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

results = pd.concat([results, evaluate(y_test, pred, label=f"{MODEL_TYPE}_log_target")], ignore_index=True)

print("\nAccuracy (target is log(assessed_value)):")
print(results.sort_values("model"))

# Convert to dollars for human-readable MAE/RMSE (approx) by exponentiating
# Note: because we trained in log space, back-transform introduces bias;
# this is still a useful portfolio summary.
y_test_dollars = np.expm1(y_test)
pred_dollars = np.expm1(pred)

dollar_metrics = evaluate(y_test_dollars, pred_dollars, label=f"{MODEL_TYPE}_dollars_approx")
print("\nApprox accuracy in dollars (back-transformed):")
print(dollar_metrics)

# Save predictions sample
out_pred = pd.DataFrame({
    "y_true_assessed_value": y_test_dollars,
    "y_pred_assessed_value": pred_dollars
}).head(2000)
out_path = OUT_DIR / "property_value_predictions_sample.csv"
out_pred.to_csv(out_path, index=False)
print(f"\nSaved prediction sample: {out_path}")
