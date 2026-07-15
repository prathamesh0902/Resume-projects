## Croston's method
The method predicts only the average interval between demands—not when the next demand will occur—and doesn't provide a probability distribution of sizes. Instead, it produces a flat line for a stable, realistic forecast.

https://learn.microsoft.com/en-us/dynamics365/supply-chain/demand-planning/croston-method#:~:text=Croston's%20method%20is%20a%20forecasting,periods%20with%20occasional%20nonzero%20demands.

---

## Concilliation Techniques
"Bottom-Up and Top-Down rely on a single direction of information flow, whereas MinT Shrinkage optimally combines forecasts from all hierarchy levels using forecast error covariance. This produces coherent forecasts with lower overall error and is considered the state-of-the-art statistical reconciliation method for hierarchical time series."

| Reconciler                         | Description                                              |
| ---------------------------------- | -------------------------------------------------------- |
| **BottomUp()**                     | Aggregate bottom-level forecasts upward.                 |
| **TopDown()**                      | Distribute top-level forecasts downward.                 |
| **MiddleOut()**                    | Forecast from a chosen intermediate level.               |
| **MinTrace(method="ols")**         | MinT using OLS covariance assumption.                    |
| **MinTrace(method="wls_var")**     | Weighted least squares using forecast variances.         |
| **MinTrace(method="mint_cov")**    | MinT using the full sample covariance matrix.            |
| **MinTrace(method="mint_shrink")** | MinT with shrinkage covariance estimation (recommended). |


https://chatgpt.com/share/6a4b6ea6-5e70-83ee-a8d0-0e3e90225db8

---

| Technique                             | Idea                                                                                                           | Pros                                                  | Cons                                      | When to Use                                     |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------- | ----------------------------------------------- |
| **Bottom-Up (BU)**                    | Forecast only the lowest level and aggregate upward.                                                           | Simple, preserves bottom-level detail.                | Errors at the bottom propagate upward.    | Bottom-level data is reliable.                  |
| **Top-Down (TD)**                     | Forecast the top level and distribute forecasts to lower levels using historical proportions.                  | Fast and simple.                                      | Lower-level forecasts can be inaccurate.  | Top-level forecasts are much more accurate.     |
| **Middle-Out (MO)**                   | Forecast at a middle level, aggregate upward and disaggregate downward.                                        | Balances BU and TD.                                   | Choosing the middle level is subjective.  | Large hierarchies with reliable mid-level data. |
| **OLS (Ordinary Least Squares)**      | Reconciles forecasts assuming equal forecast error variance across all series.                                 | Very fast and easy to compute.                        | Unrealistic assumption of equal variance. | Baseline reconciliation.                        |
| **WLS (Weighted Least Squares)**      | Gives different weights to series based on historical variances.                                               | Better than OLS when variances differ.                | Ignores covariance between series.        | Moderate-sized hierarchies.                     |
| **MinT (Minimum Trace)**              | Minimizes the variance of reconciled forecast errors using the covariance matrix of forecast errors.           | State-of-the-art statistical reconciliation.          | Requires covariance estimation.           | Recommended for production forecasting.         |
| **MinT Shrinkage**                    | Uses a shrinkage estimator for the covariance matrix to make MinT more stable when historical data is limited. | More robust, avoids overfitting covariance estimates. | Slightly more computationally expensive.  | **Most commonly recommended in practice.**      |
| **ERM (Empirical Risk Minimization)** | Learns reconciliation weights by minimizing forecasting loss directly.                                         | Can outperform statistical methods.                   | Requires more data and optimization.      | Research and advanced ML settings.              |
