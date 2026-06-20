# ─────────────────────────────────────────────────────────────
# zip_model.py
# LightGBM panel model for ZIP-level sales forecasting
# Handles ~30,000 ZIPs in a single model via shared parameters
# Routes sparse ZIPs to Croston's intermittent demand method
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error

# ── 1. Load panel data ───────────────────────────────────────
# Expected schema: zip_code, quarter_end_date, trx, + feature cols
df = pd.read_parquet("data/zip_quarterly_panel.parquet")
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values(["zip_code", "ds"]).reset_index(drop=True)

# ── 2. Route sparse ZIPs ─────────────────────────────────────
def classify_zip(series: pd.Series) -> str:
    """
    Dense   : >= 10 non-zero quarters  → LightGBM panel
    Sparse  : 4–9 non-zero quarters   → Croston's method
    Dead    : < 4 non-zero quarters   → propagate from county forecast
    """
    nonzero = (series > 0).sum()
    if nonzero >= 10:
        return "dense"
    elif nonzero >= 4:
        return "sparse"
    else:
        return "dead"

zip_class = (
    df.groupby("zip_code")["trx"]
    .apply(classify_zip)
    .reset_index()
    .rename(columns={"trx": "zip_class"})
)
df = df.merge(zip_class, on="zip_code")

print(df["zip_class"].value_counts())   # sanity check routing split

df_dense  = df[df["zip_class"] == "dense"].copy()
df_sparse = df[df["zip_class"] == "sparse"].copy()

# ── 3. Feature engineering (dense ZIPs) ─────────────────────
def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["zip_code", "ds"])

    # Lag features — previous quarters' sales
    for lag in [1, 2, 4]:
        panel[f"trx_lag{lag}"] = (
            panel.groupby("zip_code")["trx"].shift(lag)
        )

    # Rolling statistics
    panel["trx_roll4_mean"] = (
        panel.groupby("zip_code")["trx"]
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )
    panel["trx_roll4_std"] = (
        panel.groupby("zip_code")["trx"]
        .transform(lambda x: x.shift(1).rolling(4).std())
    )

    # Quarter-of-year (seasonality signal)
    panel["quarter_num"] = panel["ds"].dt.quarter

    # Encode ZIP as integer (LightGBM handles categorical natively)
    le = LabelEncoder()
    panel["zip_encoded"] = le.fit_transform(panel["zip_code"])

    return panel, le

df_dense, zip_encoder = make_features(df_dense)

# ── 4. Define feature set ────────────────────────────────────
FEATURES = [
    "zip_encoded",
    "quarter_num",
    "trx_lag1", "trx_lag2", "trx_lag4",
    "trx_roll4_mean", "trx_roll4_std",
    # add external regressors here if available at ZIP level:
    # "rep_visit_count", "hcp_count_in_zip", "median_hh_income",
]
TARGET = "trx"

# Drop rows with NaN lags (first few periods per ZIP)
df_model = df_dense.dropna(subset=FEATURES)

HOLDOUT_QTR = "2023-10-01"   # last 4 quarters = holdout
df_train = df_model[df_model["ds"] < HOLDOUT_QTR]
df_test  = df_model[df_model["ds"] >= HOLDOUT_QTR]

X_train, y_train = df_train[FEATURES], df_train[TARGET]
X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]

# ── 5. Train LightGBM panel model ───────────────────────────
params = {
    "objective":        "regression_l1",   # MAE loss; robust to outliers
    "metric":           "mape",
    "learning_rate":    0.05,
    "num_leaves":       64,
    "min_child_samples": 20,               # prevents over-fitting small ZIPs
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "n_estimators":     500,
    "early_stopping_rounds": 30,
    "verbose":          -1,
}

model_zip = lgb.LGBMRegressor(**params)
model_zip.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)

mape = mean_absolute_percentage_error(y_test, model_zip.predict(X_test))
print(f"ZIP panel holdout MAPE: {mape:.1%}")

# ── 6. Sparse ZIP handler — Croston's method ─────────────────
def crostons_forecast(series: np.ndarray, horizon: int = 4) -> np.ndarray:
    """
    Classic Croston's method for intermittent demand.
    Separately smooths the non-zero demand and inter-demand intervals.
    Returns a flat forecast array of length `horizon`.
    """
    alpha = 0.1       # smoothing parameter
    demand, interval = [], []
    d_hat, i_hat = None, None
    last_nonzero = None

    for t, val in enumerate(series):
        if val > 0:
            if d_hat is None:
                d_hat = val
                i_hat = 1 if last_nonzero is None else t - last_nonzero
            else:
                d_hat = alpha * val + (1 - alpha) * d_hat
                gap   = t - last_nonzero
                i_hat = alpha * gap + (1 - alpha) * i_hat
            last_nonzero = t

    if d_hat is None or i_hat is None or i_hat == 0:
        return np.zeros(horizon)

    rate = d_hat / i_hat   # expected demand per period
    return np.full(horizon, rate)


sparse_forecasts = {}
for zip_code, grp in df_sparse.groupby("zip_code"):
    series = grp.sort_values("ds")["trx"].values
    sparse_forecasts[zip_code] = crostons_forecast(series, horizon=4)

# ── 7. Assemble ZIP-level forecast output ────────────────────
HORIZON_DATES = pd.date_range("2024-01-01", periods=4, freq="QS")

# Dense ZIPs — predict on constructed future feature rows
# (In production: build a future feature dataframe the same way as training)
# Here we show the structure; replace with actual future feature rows
zip_forecast_rows = []
for zip_code in df_dense["zip_code"].unique():
    preds = model_zip.predict(X_test[X_test.index.isin(
        df_test[df_test["zip_code"] == zip_code].index
    )])
    for dt, pred in zip(HORIZON_DATES[:len(preds)], preds):
        zip_forecast_rows.append({
            "geo_id":   zip_code,
            "ds":       dt,
            "forecast": max(pred, 0),   # floor at 0 — no negative Rx
            "level":    "zip",
            "method":   "lgbm_panel",
        })

# Sparse ZIPs
for zip_code, preds in sparse_forecasts.items():
    for dt, pred in zip(HORIZON_DATES, preds):
        zip_forecast_rows.append({
            "geo_id":   zip_code,
            "ds":       dt,
            "forecast": float(pred),
            "level":    "zip",
            "method":   "crostons",
        })

zip_out = pd.DataFrame(zip_forecast_rows)
zip_out.to_parquet("data/forecasts/zip_base.parquet", index=False)
print(f"ZIP forecasts written: {len(zip_out)} rows")
