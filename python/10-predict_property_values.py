import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"
OUT_DIR = ROOT / "powerbi"
OUT_DIR.mkdir(exist_ok=True)

OUT_PRED = OUT_DIR / "property_value_predictions.csv"
OUT_IMPORTANCE = OUT_DIR / "rf_feature_importance.csv"


# ----------------------------
# 1) Build a modeling dataset in SQL
#    - property-level target: assessed_value
#    - neighborhood-level features from permits
# ----------------------------
QUERY = """
WITH permits_by_neighborhood AS (
  SELECT
    neighborhood,
    COUNT(*) AS permits_total,
    SUM(COALESCE(project_value, 0)) AS permits_total_value,
    AVG(project_value) AS permits_avg_value,
    SUM(CASE WHEN project_value IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS permits_value_missing_rate
  FROM permits
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

  -- neighborhood-level permit features (can be NULL if no permits found)
  pb.permits_total,
  pb.permits_total_value,
  pb.permits_avg_value,
  pb.permits_value_missing_rate

FROM properties p
LEFT JOIN permits_by_neighborhood pb
  ON p.neighborhood = pb.neighborhood
WHERE p.assessed_value IS NOT NULL;
"""

with sqlite3.connect(DB_PATH) as conn:
  df = pd.read_sql_query(QUERY, conn)

print(f"Loaded modeling rows: {len(df):,}")
print("Nulls per column (top 12):")
print(df.isna().sum().sort_values(ascending=False).head(12))


# ----------------------------
# 2) Light cleaning / feature transforms
# ----------------------------

# Remove extreme weird targets if you want (optional):
# df = df[df["assessed_value"].between(1, 20_000_000)].copy()

# Reduce skew in target: predict log(assessed_value)
df["y_log"] = np.log1p(df["assessed_value"].clip(lower=0))

# Create a few simple engineered numeric features
df["permits_total"] = df["permits_total"]  # keep as is (may be NaN -> imputer)
df["permits_total_value"] = df["permits_total_value"]
df["permits_avg_value"] = df["permits_avg_value"]

# Log features to reduce skew (safe even if NaN; imputer handles later)
for col in ["permits_total", "permits_total_value", "permits_avg_value"]:
    df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

# Keep features we will model with
feature_cols = [
    # categorical
    "neighborhood", "ward", "tax_class", "garage",
    # numeric
    "latitude", "longitude",
    "log_permits_total", "log_permits_total_value", "log_permits_avg_value",
    "permits_value_missing_rate",
]

X = df[feature_cols].copy()
y = df["y_log"].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify column types
cat_cols = ["neighborhood", "ward", "tax_class", "garage"]
num_cols = [c for c in feature_cols if c not in cat_cols]


# ----------------------------
# 3) Preprocessing: impute + encode
# ----------------------------
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ]
)

# ----------------------------
# 4) Models
# ----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5
    )
}

results = []
pred_tables = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred_log = pipe.predict(X_test)

    # Convert back to dollars for interpretation
    y_true = np.expm1(y_test)
    y_pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    results.append({
        "model": name,
        "MAE_$": mae,
        "RMSE_$": rmse,
        "R2": r2,
        "n_test": len(y_test)
    })

    # Save a small prediction table for Power BI
    tmp = X_test.copy()
    tmp["actual_assessed_value"] = y_true.values
    tmp["predicted_assessed_value"] = y_pred
    tmp["model"] = name
    pred_tables.append(tmp)

    # Feature importance for RF
    if name == "RandomForest":
        # Get feature names after preprocessing
        ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([np.array(num_cols), cat_feature_names])

        importances = pipe.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        imp_df.to_csv(OUT_IMPORTANCE, index=False)
        print(f"Saved feature importance: {OUT_IMPORTANCE}")

# Results table
res_df = pd.DataFrame(results).sort_values("MAE_$")
print("\nModel comparison (lower MAE/RMSE is better):")
print(res_df.to_string(index=False))

# Save prediction rows
pred_df = pd.concat(pred_tables, ignore_index=True)
pred_df.to_csv(OUT_PRED, index=False)
print(f"\nSaved predictions for Power BI: {OUT_PRED}")

print("\nDone.")
