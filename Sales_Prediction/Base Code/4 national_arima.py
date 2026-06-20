# ═══════════════════════════════════════════════════════════════
# national_arima.py
# ARIMA-X model for national-level quarterly pharma sales
# Includes: synthetic data generation, model fitting, evaluation
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

# ────────────────────────────────────────────────────────────────
# STEP 1 — Synthetic national data
# Simulates 16 quarters (Q1 2020 – Q4 2023) of national TRx sales
# Components:
#   • Upward trend (~+2% per quarter, organic growth)
#   • Quarterly seasonality (Q4 flu season bump, Q2 dip)
#   • A COVID shock in 2020 Q2-Q3 (HCP visits dropped)
#   • Promotional spend as an external regressor
#   • Gaussian noise
# ────────────────────────────────────────────────────────────────
quarters = pd.date_range("2020-01-01", periods=16, freq="QS")
n = len(quarters)

trend      = np.linspace(100_000, 135_000, n)
seasonality = np.tile([0.02, -0.04, 0.01, 0.06], 4)     # Q1-Q4 multipliers
covid_shock = np.array([0,-0.18,-0.10,0,0,0,0,0,0,0,0,0,0,0,0,0])  # Q2-Q3 2020

promo_spend = np.array([
    1.0,0.8,1.1,1.3, 1.0,0.9,1.2,1.4,
    1.1,1.0,1.3,1.5, 1.2,1.1,1.4,1.6
])  # normalized spend index; >1 = above baseline

noise = np.random.normal(0, 2_000, n)

trx = trend * (1 + seasonality + covid_shock) + promo_spend * 3_000 + noise
trx = trx.clip(min=0)

df_national = pd.DataFrame({
    "ds":          quarters,
    "trx":         trx.round(0),
    "promo_spend": promo_spend,
    "covid_flag":  (covid_shock != 0).astype(int),
})

print("── Synthetic national data (16 quarters) ──")
print(df_national.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# STEP 2 — Stationarity check (ADF test)
# ARIMA requires a stationary series (constant mean & variance).
# If the series is non-stationary we difference it (d=1).
# ADF null hypothesis: series has a unit root (non-stationary).
# p < 0.05 → reject null → series IS stationary.
# ────────────────────────────────────────────────────────────────
adf_result = adfuller(df_national["trx"], autolag="AIC")
print(f"\n── ADF test (raw TRx) ──")
print(f"  ADF statistic : {adf_result[0]:.4f}")
print(f"  p-value       : {adf_result[1]:.4f}")
print(f"  Conclusion    : {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary → will use d=1'}")

# ────────────────────────────────────────────────────────────────
# STEP 3 — Train / test split
# Hold out last 4 quarters as the validation window.
# This mirrors real deployment: train on Q1-2020 → Q4-2022,
# forecast Q1-2023 → Q4-2023, compare against known actuals.
# ────────────────────────────────────────────────────────────────
HOLDOUT = 4
df_train = df_national.iloc[:-HOLDOUT].copy()
df_test  = df_national.iloc[-HOLDOUT:].copy()

y_train   = df_train["trx"]
exog_train = df_train[["promo_spend", "covid_flag"]]
exog_test  = df_test[["promo_spend", "covid_flag"]]

# ────────────────────────────────────────────────────────────────
# STEP 4 — Fit SARIMAX model
# Order (p, d, q):
#   p=1  — one autoregressive lag (TRx depends on previous quarter)
#   d=1  — first-order differencing to remove trend non-stationarity
#   q=1  — one moving average term (smooths short-term shocks)
# Seasonal order (P, D, Q, s) with s=4 (quarterly):
#   P=1, D=0, Q=1 — mild seasonal AR and MA terms
# exog — include promo_spend and covid_flag as external regressors
# ────────────────────────────────────────────────────────────────
model = SARIMAX(
    endog=y_train,
    exog=exog_train,
    order=(1, 1, 1),               # (p, d, q)
    seasonal_order=(1, 0, 1, 4),   # (P, D, Q, s)
    enforce_stationarity=False,
    enforce_invertibility=False,
)

result = model.fit(disp=False)

print("\n── SARIMAX model summary ──")
print(result.summary().tables[1])  # coefficient table only

# ────────────────────────────────────────────────────────────────
# STEP 5 — Forecast on holdout
# get_forecast() returns point estimates + confidence intervals.
# We pass exog_test so the model can use known promo/covid values
# for the 4 forecast quarters.
# ────────────────────────────────────────────────────────────────
forecast_obj   = result.get_forecast(steps=HOLDOUT, exog=exog_test)
forecast_mean  = forecast_obj.predicted_mean
forecast_ci    = forecast_obj.conf_int(alpha=0.20)   # 80% CI

forecast_mean  = forecast_mean.clip(lower=0)

actuals   = df_test["trx"].values
predicted = forecast_mean.values
mape = np.mean(np.abs((actuals - predicted) / (actuals + 1e-9))) * 100

print(f"\n── Holdout results (last {HOLDOUT} quarters) ──")
results_df = pd.DataFrame({
    "quarter":   df_test["ds"].dt.to_period("Q").astype(str),
    "actual":    actuals.round(0),
    "forecast":  predicted.round(0),
    "pct_error": ((predicted - actuals) / actuals * 100).round(1),
    "lo_80":     forecast_ci.iloc[:, 0].clip(0).round(0).values,
    "hi_80":     forecast_ci.iloc[:, 1].round(0).values,
})
print(results_df.to_string(index=False))
print(f"\n  MAPE: {mape:.1f}%")

# ────────────────────────────────────────────────────────────────
# STEP 6 — Diagnostic plots
# Saves two plots:
#   (a) Forecast vs actuals with confidence band
#   (b) Residual diagnostics (for model validation)
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Panel A — forecast vs actuals
ax = axes[0]
ax.plot(df_train["ds"], df_train["trx"], color="#378ADD", label="Train actuals")
ax.plot(df_test["ds"],  df_test["trx"],  color="#378ADD", linestyle="--", label="Test actuals")
ax.plot(df_test["ds"],  predicted,       color="#D85A30", label="Forecast")
ax.fill_between(
    df_test["ds"],
    forecast_ci.iloc[:, 0].clip(0),
    forecast_ci.iloc[:, 1],
    color="#D85A30", alpha=0.15, label="80% CI"
)
ax.set_title("National TRx — ARIMA forecast vs actuals", fontsize=11)
ax.set_ylabel("TRx"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel B — residuals
residuals = result.resid
ax2 = axes[1]
ax2.plot(df_train["ds"], residuals, color="#888780")
ax2.axhline(0, color="#E24B4A", linewidth=0.8)
ax2.set_title("Model residuals (training period)", fontsize=11)
ax2.set_ylabel("Residual"); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/national_arima_plot.png", dpi=140, bbox_inches="tight")
print("\nPlot saved → national_arima_plot.png")

# ────────────────────────────────────────────────────────────────
# STEP 7 — Export for MinT reconciliation
# Standard schema: geo_id | ds | forecast | level
# ────────────────────────────────────────────────────────────────
national_out = pd.DataFrame({
    "geo_id":   "USA",
    "ds":       df_test["ds"],
    "forecast": predicted.round(0),
    "lo_80":    forecast_ci.iloc[:, 0].clip(0).round(0).values,
    "hi_80":    forecast_ci.iloc[:, 1].round(0).values,
    "level":    "national",
})
national_out.to_parquet("/mnt/user-data/outputs/national_arima_forecast.parquet", index=False)
print("Forecast parquet saved → national_arima_forecast.parquet")
