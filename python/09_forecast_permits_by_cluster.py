import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "edmonton.db"
OUT = ROOT / "powerbi" / "cluster_forecasts.csv"
OUT.parent.mkdir(exist_ok=True)

# -------------------------------------------------
# 1) Load data: monthly permits + clusters
# -------------------------------------------------

QUERY = """
SELECT
  v.year,
  v.month_number,
  v.permits_count,
  c.cluster_ranked AS cluster
FROM v_neighborhood_monthly v
JOIN neighborhood_clusters c
  ON v.neighborhood = c.neighborhood
"""

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(QUERY, conn)

# Create a proper date
df["date"] = pd.to_datetime(
    dict(year=df.year, month=df.month_number, day=1)
)

# -------------------------------------------------
# 2) Aggregate to CLUSTER x MONTH
# -------------------------------------------------

cluster_month = (
    df.groupby(["cluster", "date"], as_index=False)
      .agg(
          permits=("permits_count", "sum")
      )
      .sort_values(["cluster", "date"])
)

# -------------------------------------------------
# 3) Create lag features (time series ML)
# -------------------------------------------------

cluster_month["permits_lag1"] = (
    cluster_month.groupby("cluster")["permits"].shift(1)
)
cluster_month["permits_lag2"] = (
    cluster_month.groupby("cluster")["permits"].shift(2)
)

cluster_month["month"] = cluster_month["date"].dt.month
cluster_month["year"] = cluster_month["date"].dt.year

# Drop rows with missing lags
model_df = cluster_month.dropna().copy()

# -------------------------------------------------
# 4) Train one model PER CLUSTER
# -------------------------------------------------

results = []

FEATURES = ["permits_lag1", "permits_lag2", "month"]
TARGET = "permits"

for cluster_id in model_df["cluster"].unique():
    data = model_df[model_df["cluster"] == cluster_id]

    X = data[FEATURES]
    y = data[TARGET]

    # Train / test split (time-based)
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ---- Choose model ----
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    # model = LinearRegression()  # <- switch if you want

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results.append({
        "cluster": cluster_id,
        "n_obs": len(data),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    })

# -------------------------------------------------
# 5) Forecast NEXT month for each cluster
# -------------------------------------------------

forecasts = []

for cluster_id in model_df["cluster"].unique():
    data = model_df[model_df["cluster"] == cluster_id]

    last_row = data.iloc[-1]

    X_next = pd.DataFrame([{
        "permits_lag1": last_row["permits"],
        "permits_lag2": data.iloc[-2]["permits"],
        "month": (last_row["month"] % 12) + 1
    }])

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(data[FEATURES], data[TARGET])

    forecast = model.predict(X_next)[0]

    forecasts.append({
        "cluster": cluster_id,
        "forecast_next_month_permits": forecast
    })

# -------------------------------------------------
# 6) Save outputs
# -------------------------------------------------

results_df = pd.DataFrame(results)
forecast_df = pd.DataFrame(forecasts)

final = forecast_df.merge(results_df, on="cluster")

final.to_csv(OUT, index=False)

print("Forecasting complete")
print(final.sort_values("forecast_next_month_permits", ascending=False))
