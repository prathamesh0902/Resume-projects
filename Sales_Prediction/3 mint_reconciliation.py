# ─────────────────────────────────────────────────────────────
# mint_reconciliation.py
# Hierarchical forecast reconciliation using MinT (Minimum Trace)
# Ensures: ZIP forecasts sum → county → state → national
#
# Uses the `hierarchicalforecast` library by Nixtla
#   pip install hierarchicalforecast
#
# Input:  base forecasts from national_model.py + zip_model.py
#         + state & county base forecasts (same schema)
# Output: reconciled forecasts at all geo levels
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
from hierarchicalforecast.utils import aggregate

# ── 1. Load base forecasts from all levels ───────────────────
# Schema for each: geo_id | ds | forecast | level
nat_fc    = pd.read_parquet("data/forecasts/national_base.parquet")
state_fc  = pd.read_parquet("data/forecasts/state_base.parquet")
county_fc = pd.read_parquet("data/forecasts/county_base.parquet")
zip_fc    = pd.read_parquet("data/forecasts/zip_base.parquet")

# ── 2. Build the hierarchy spec ──────────────────────────────
# geo_crosswalk maps every ZIP to its parent county, state, national
geo_xwalk = pd.read_csv("data/geo_crosswalk.csv")
# Required columns: zip_code, county_fips, state_abbr
# (national is the single root — we add it as a constant column)
geo_xwalk["national"] = "USA"

# hierarchicalforecast expects a list-of-lists defining each level
# Format: [[level_0_col], [level_1_col], ..., [leaf_col]]
# Each inner list = the columns that uniquely identify a node at that level
HIERARCHY_SPEC = [
    ["national"],
    ["state_abbr"],
    ["county_fips"],
    ["zip_code"],
]

# ── 3. Build the wide panel of actuals for covariance estimation ─
# MinT needs historical actuals (not just forecasts) to estimate
# error covariance across the hierarchy
actuals_zip = pd.read_parquet("data/zip_quarterly_panel.parquet")  # zip_code | ds | trx

# Aggregate actuals up the hierarchy using the crosswalk
actuals_zip = actuals_zip.merge(
    geo_xwalk[["zip_code", "county_fips", "state_abbr", "national"]],
    on="zip_code", how="left"
)

# Use hierarchicalforecast's `aggregate` utility to build
# a coherent wide panel: one row per (ds, unique_id), where
# unique_id encodes the full path (e.g. "USA/CA/06037/90210")
Y_df, S_df, tags = aggregate(
    df=actuals_zip,
    spec=HIERARCHY_SPEC,
)
# Y_df  : long-format actuals (unique_id | ds | y)
# S_df  : summing matrix — maps leaf nodes to all aggregates
# tags  : dict mapping level name → list of unique_ids at that level

print(f"Hierarchy nodes: {S_df.shape[1]} bottom, {S_df.shape[0]} total")
print("Level counts:")
for level, ids in tags.items():
    print(f"  {level:15s}: {len(ids):6,} nodes")

# ── 4. Stack base forecasts into the required format ─────────
# hierarchicalforecast expects: unique_id | ds | ModelName
# unique_id must match the format produced by `aggregate` above
# (full path string, e.g. "USA/CA/06037/90210")

def build_unique_id(df: pd.DataFrame, geo_col: str, level: str) -> pd.DataFrame:
    """Attach full-path unique_id from the crosswalk."""
    path_cols = {
        "national": ["national"],
        "state":    ["national", "state_abbr"],
        "county":   ["national", "state_abbr", "county_fips"],
        "zip":      ["national", "state_abbr", "county_fips", "zip_code"],
    }
    df = df.merge(geo_xwalk, on=geo_col, how="left")
    cols = path_cols[level]
    df["unique_id"] = df[cols].apply(lambda r: "/".join(r.astype(str)), axis=1)
    return df[["unique_id", "ds", "forecast"]]

nat_fc_fmt    = build_unique_id(nat_fc.rename(columns={"geo_id":"national"}),    "national",    "national")
state_fc_fmt  = build_unique_id(state_fc.rename(columns={"geo_id":"state_abbr"}),  "state_abbr",  "state")
county_fc_fmt = build_unique_id(county_fc.rename(columns={"geo_id":"county_fips"}),"county_fips", "county")
zip_fc_fmt    = build_unique_id(zip_fc.rename(columns={"geo_id":"zip_code"}),    "zip_code",    "zip")

# Combine all levels; rename 'forecast' → model name column
all_base = pd.concat([nat_fc_fmt, state_fc_fmt, county_fc_fmt, zip_fc_fmt])
all_base = all_base.rename(columns={"forecast": "BaseModel"})

# Pivot to wide: one column per model (here we have one: BaseModel)
Y_hat_df = all_base.pivot_table(
    index=["unique_id", "ds"],
    values="BaseModel",
    aggfunc="first"
).reset_index()

# ── 5. Run MinT reconciliation ───────────────────────────────
hrec = HierarchicalReconciliation(
    reconcilers=[
        MinTrace(method="mint_shrink"),   # shrinkage estimator — best for
                                           # large sparse hierarchies like ZIPs
        # Alternatives to try:
        # MinTrace(method="ols")          — ordinary least squares, faster
        # MinTrace(method="wls_struct")   — weighted by node size
        # BottomUp()                      — simple aggregation, no top signal
    ]
)

reconciled = hrec.reconcile(
    Y_hat_df=Y_hat_df,   # base forecasts, wide format
    Y_df=Y_df,           # historical actuals (for covariance estimation)
    S=S_df,              # summing matrix
    tags=tags,           # level metadata
)

# reconciled columns: unique_id | ds | BaseModel | BaseModel/MinTrace_mint_shrink
RECONCILED_COL = "BaseModel/MinTrace_mint_shrink"

# ── 6. Post-process & validate coherence ─────────────────────
# Floor negative reconciled values (can happen in very sparse ZIPs)
reconciled[RECONCILED_COL] = reconciled[RECONCILED_COL].clip(lower=0)

def check_coherence(df: pd.DataFrame, col: str, tol: float = 0.01) -> None:
    """
    Spot-check that ZIP forecasts sum to their parent county.
    Prints any violations above `tol` relative error.
    """
    df = df.copy()
    df["county_id"] = df["unique_id"].apply(lambda x: "/".join(x.split("/")[:3]))
    df["is_zip"]    = df["unique_id"].str.count("/") == 3

    zip_sum    = df[df["is_zip"]].groupby(["county_id", "ds"])[col].sum()
    county_val = df[~df["is_zip"] & (df["unique_id"].str.count("/") == 2)]\
                    .set_index(["unique_id", "ds"])[col]

    for (cid, ds), z_sum in zip_sum.items():
        try:
            c_val = county_val.loc[(cid, ds)]
            rel_err = abs(z_sum - c_val) / (c_val + 1e-9)
            if rel_err > tol:
                print(f"  WARN coherence gap: {cid} {ds} — ZIP sum={z_sum:.1f}, county={c_val:.1f}")
        except KeyError:
            pass

print("\nCoherence check (ZIP → county):")
check_coherence(reconciled, RECONCILED_COL)
print("  Done — no output means all within tolerance.")

# ── 7. Reshape to long output format ─────────────────────────
def extract_level(unique_id: str) -> str:
    depth = unique_id.count("/")
    return {0: "national", 1: "state", 2: "county", 3: "zip"}.get(depth, "unknown")

final = reconciled[["unique_id", "ds", RECONCILED_COL]].copy()
final = final.rename(columns={RECONCILED_COL: "reconciled_forecast"})
final["level"] = final["unique_id"].apply(extract_level)
final["geo_id"] = final["unique_id"].apply(lambda x: x.split("/")[-1])

# Summary
print("\nReconciled forecast counts by level:")
print(final["level"].value_counts())

final.to_parquet("data/forecasts/reconciled_all_levels.parquet", index=False)
print("\nOutput saved → data/forecasts/reconciled_all_levels.parquet")
print(final.head(10).to_string(index=False))
