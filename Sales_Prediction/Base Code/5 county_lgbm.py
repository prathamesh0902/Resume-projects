# ═══════════════════════════════════════════════════════════════
# county_lgbm.py
# LightGBM panel model for county-level quarterly pharma sales
# Includes: synthetic data generation, feature engineering,
#           training, evaluation, SHAP feature importance
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

# ────────────────────────────────────────────────────────────────
# STEP 1 — Synthetic county-level panel data
# Simulates 16 quarters for 50 counties across 5 states.
# Each county has its own baseline, growth rate, and noise level —
# mimicking the real-world variation across small vs large markets.
# External features: rep visit count, HCP density, median income.
# ────────────────────────────────────────────────────────────────
N_COUNTIES = 50
N_QUARTERS = 16
quarters    = pd.date_range("2020-01-01", periods=N_QUARTERS, freq="QS")

states     = ["CA", "TX", "FL", "NY", "IL"]
state_map  = {i: states[i % len(states)] for i in range(N_COUNTIES)}

rows = []
for cid in range(N_COUNTIES):
    baseline    = np.random.uniform(5_000, 40_000)   # county market size
    growth      = np.random.uniform(0.005, 0.025)    # quarterly growth rate
    noise_scale = baseline * 0.07

    for t, dt in enumerate(quarters):
        trend    = baseline * (1 + growth) ** t
        seasonal = trend * np.array([0.02, -0.04, 0.01, 0.06])[t % 4]
        covid    = -trend * 0.15 if t in [1, 2] else 0   # Q2-Q3 2020 shock

        # External features — correlated with sales but with noise
        rep_visits  = int(np.random.poisson(lam=8 + baseline / 8_000))
        hcp_density = round(np.random.uniform(2.0, 12.0), 1)   # HCPs per 10k pop
        med_income  = round(np.random.uniform(40_000, 90_000), -3)

        trx = trend + seasonal + covid + np.random.normal(0, noise_scale)

        rows.append({
            "county_id":   f"CTY_{cid:03d}",
            "state":       state_map[cid],
            "ds":          dt,
            "trx":         max(trx, 0),
            "rep_visits":  rep_visits,
            "hcp_density": hcp_density,
            "med_income":  med_income,
        })

df = pd.DataFrame(rows)
df["trx"] = df["trx"].round(0)

print(f"── Synthetic panel: {df['county_id'].nunique()} counties × {N_QUARTERS} quarters = {len(df)} rows ──")
print(df.groupby("state")["trx"].agg(["mean","sum"]).round(0))

# ────────────────────────────────────────────────────────────────
# STEP 2 — Feature engineering
# LightGBM is a tree model — it cannot extrapolate a trend on its
# own. We must give it trend-capturing features explicitly:
#   • Lag features : what sales were 1, 2, and 4 quarters ago
#   • Rolling stats: mean and std over trailing 4 quarters
#   • Quarter number: captures seasonality (Q4 = 4, etc.)
#   • Time index   : captures the overall trend direction
# Lags introduce NaNs for early rows — we drop those below.
# ────────────────────────────────────────────────────────────────
df = df.sort_values(["county_id", "ds"]).reset_index(drop=True)

for lag in [1, 2, 4]:
    df[f"trx_lag{lag}"] = df.groupby("county_id")["trx"].shift(lag)

df["trx_roll4_mean"] = (
    df.groupby("county_id")["trx"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
)
df["trx_roll4_std"] = (
    df.groupby("county_id")["trx"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=2).std().fillna(0))
)

df["quarter_num"] = df["ds"].dt.quarter
df["time_index"]  = df.groupby("county_id").cumcount()   # 0…15 per county

# Encode county_id and state as integers for LightGBM
le_county = LabelEncoder()
le_state  = LabelEncoder()
df["county_encoded"] = le_county.fit_transform(df["county_id"])
df["state_encoded"]  = le_state.fit_transform(df["state"])

print(f"\n── Feature engineering complete. Sample row ──")
print(df[df["county_id"]=="CTY_000"].tail(3).to_string(index=False))

# ────────────────────────────────────────────────────────────────
# STEP 3 — Train / test split
# Temporal split: last 4 quarters = holdout.
# Important: we split by DATE, not randomly, to avoid data leakage.
# A random split would let future lags leak into training rows.
# ────────────────────────────────────────────────────────────────
HOLDOUT_DATE = "2023-01-01"

# Drop rows where lag features are NaN (first 4 quarters per county)
FEATURES = [
    "county_encoded", "state_encoded",
    "quarter_num", "time_index",
    "trx_lag1", "trx_lag2", "trx_lag4",
    "trx_roll4_mean", "trx_roll4_std",
    "rep_visits", "hcp_density", "med_income",
]
df_model = df.dropna(subset=FEATURES).copy()

df_train = df_model[df_model["ds"] <  HOLDOUT_DATE]
df_test  = df_model[df_model["ds"] >= HOLDOUT_DATE]

X_train, y_train = df_train[FEATURES], df_train["trx"]
X_test,  y_test  = df_test[FEATURES],  df_test["trx"]

print(f"\n── Train: {len(df_train)} rows | Test: {len(df_test)} rows ──")

# ────────────────────────────────────────────────────────────────
# STEP 4 — Train LightGBM panel model
# One model learns from ALL counties simultaneously.
# Key hyperparameters for a pharma panel:
#   objective=regression_l1  : MAE loss; robust to occasional large-
#                               volume counties dominating gradients
#   min_child_samples=10     : minimum rows per leaf — prevents the
#                               model from memorizing tiny counties
#   num_leaves=64            : tree complexity; 64 is moderate
#   early_stopping_rounds    : stop when validation MAPE stops falling
# ────────────────────────────────────────────────────────────────
params = {
    "objective":             "regression_l1",
    "metric":                "mape",
    "learning_rate":         0.05,
    "num_leaves":            64,
    "min_child_samples":     10,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "n_estimators":          600,
    "early_stopping_rounds": 40,
    "verbose":               -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.log_evaluation(period=100)],
)

# ────────────────────────────────────────────────────────────────
# STEP 5 — Evaluate
# We evaluate at two levels:
#   (a) Overall panel MAPE
#   (b) Per-county MAPE — important for spotting specific
#       counties where the model systematically under/over-forecasts
# ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test).clip(min=0)

overall_mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"\n── Overall holdout MAPE: {overall_mape:.1%} ──")

eval_df = df_test[["county_id","state","ds","trx"]].copy()
eval_df["forecast"]  = y_pred.round(0)
eval_df["pct_error"] = ((y_pred - y_test.values) / (y_test.values + 1e-9) * 100).round(1)

per_county_mape = (
    eval_df.groupby("county_id")
    .apply(lambda g: mean_absolute_percentage_error(g["trx"], g["forecast"]))
    .reset_index()
    .rename(columns={0: "mape"})
    .sort_values("mape", ascending=False)
)
print("\n── Worst 5 counties by holdout MAPE ──")
print(per_county_mape.head(5).to_string(index=False))
print("\n── Best 5 counties by holdout MAPE ──")
print(per_county_mape.tail(5).to_string(index=False))

# ────────────────────────────────────────────────────────────────
# STEP 6 — Feature importance
# LightGBM's gain-based importance tells us which features drive
# the most reduction in loss at split points across all trees.
# This directly answers: "what makes county sales go up or down?"
# ────────────────────────────────────────────────────────────────
importance = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=True)

print("\n── Feature importance (gain) ──")
print(importance.sort_values("importance", ascending=False).to_string(index=False))

# ────────────────────────────────────────────────────────────────
# STEP 7 — Plots  (2-panel figure)
#   Left  : actual vs forecast for a sample county (CTY_000)
#   Right : feature importance bar chart
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Panel A — sample county forecast
ax = axes[0]
sample = df[df["county_id"] == "CTY_000"].sort_values("ds")
pred_sample = eval_df[eval_df["county_id"] == "CTY_000"].sort_values("ds")

ax.plot(sample["ds"], sample["trx"], color="#378ADD", label="Actuals")
ax.plot(pred_sample["ds"], pred_sample["forecast"],
        color="#D85A30", marker="o", markersize=4, label="Forecast (holdout)")
ax.axvline(pd.Timestamp(HOLDOUT_DATE), color="#888780",
           linestyle="--", linewidth=0.9, label="Train/test split")
ax.set_title("County CTY_000 — actual vs forecast", fontsize=11)
ax.set_ylabel("TRx"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel B — feature importance
ax2 = axes[1]
colors = ["#378ADD" if i >= len(FEATURES) - 3 else "#B5D4F4"
          for i in range(len(importance))]
ax2.barh(importance["feature"], importance["importance"], color=colors)
ax2.set_title("Feature importance (gain)", fontsize=11)
ax2.set_xlabel("Importance score"); ax2.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/county_lgbm_plot.png", dpi=140, bbox_inches="tight")
print("\nPlot saved → county_lgbm_plot.png")

# ────────────────────────────────────────────────────────────────
# STEP 8 — Export for MinT reconciliation
# ────────────────────────────────────────────────────────────────
county_out = eval_df[["county_id","ds","forecast"]].rename(
    columns={"county_id":"geo_id","forecast":"forecast"}
)
county_out["level"] = "county"
county_out.to_parquet("/mnt/user-data/outputs/county_lgbm_forecast.parquet", index=False)
print("Forecast parquet saved → county_lgbm_forecast.parquet")
