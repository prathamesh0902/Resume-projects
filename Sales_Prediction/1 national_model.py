# ─────────────────────────────────────────────────────────────
# national_model.py
# Prophet-based forecasting for national-level sales
# Inputs : quarterly Rx sales + promotional/seasonal regressors
# Outputs: 4-quarter ahead forecast with prediction intervals
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# ── 1. Load & prep ───────────────────────────────────────────
# Expected columns: ds (period-end date), y (TRx), plus regressor cols
df_national = pd.read_parquet("data/national_quarterly.parquet")

# Prophet requires 'ds' as datetime; convert quarter-end dates
df_national["ds"] = pd.to_datetime(df_national["ds"])
df_national = df_national.sort_values("ds").reset_index(drop=True)

# Flag COVID-disrupted quarters (down-weight, not drop)
df_national["covid_disruption"] = (
    (df_national["ds"] >= "2020-01-01") &
    (df_national["ds"] <= "2021-09-30")
).astype(int)

# ── 2. Train / test split (hold out last 4 quarters) ─────────
HOLDOUT = 4
df_train = df_national.iloc[:-HOLDOUT]
df_test  = df_national.iloc[-HOLDOUT:]

# ── 3. Build model ───────────────────────────────────────────
model_national = Prophet(
    growth="linear",
    yearly_seasonality=False,   # quarterly data — handle manually
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative",  # pharma sales scale with trend
    interval_width=0.80,                # 80% prediction interval
    changepoint_prior_scale=0.05,       # conservative — 16 quarters is short
)

# Quarterly seasonality (4 Fourier terms covers annual cycle)
model_national.add_seasonality(
    name="quarterly",
    period=365.25 / 4,
    fourier_order=4,
)

# External regressors — add any that are available at forecast time
REGRESSORS = [
    "promo_spend_normalized",   # normalized promotional $
    "flu_season_flag",          # 1 = Q4/Q1 peak cold/flu period
    "new_indication_launch",    # 1 = quarter of a label expansion
    "covid_disruption",
]
for reg in REGRESSORS:
    if reg in df_national.columns:
        model_national.add_regressor(reg, standardize=True)

# ── 4. Fit ───────────────────────────────────────────────────
model_national.fit(df_train[["ds", "y"] + REGRESSORS])

# ── 5. Forecast ──────────────────────────────────────────────
# Future dataframe must include regressor values for forecast horizon
future = model_national.make_future_dataframe(
    periods=HOLDOUT,
    freq="QS",          # quarter-start frequency
    include_history=True,
)

# Merge known/planned regressor values for future periods
future = future.merge(
    df_national[["ds"] + REGRESSORS],
    on="ds", how="left"
).fillna(0)

forecast_national = model_national.predict(future)

# ── 6. Evaluate on holdout ───────────────────────────────────
actuals   = df_test["y"].values
predicted = forecast_national.tail(HOLDOUT)["yhat"].values

mape = mean_absolute_percentage_error(actuals, predicted)
print(f"National holdout MAPE (4Q): {mape:.1%}")

# ── 7. Export for reconciliation ─────────────────────────────
# Keep only future-period rows; rename to standard schema
national_out = (
    forecast_national[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    .tail(HOLDOUT)
    .assign(level="national", geo_id="USA")
    .rename(columns={"yhat": "forecast", "yhat_lower": "lo_80", "yhat_upper": "hi_80"})
)

national_out.to_parquet("data/forecasts/national_base.parquet", index=False)
print(national_out)
