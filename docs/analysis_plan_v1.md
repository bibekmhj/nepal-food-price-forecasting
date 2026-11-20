# Analysis plan v1 (frozen before any forecast error was computed)
Date: 19 November 2026. Project: forecasting retail food prices in Nepal (WFP/HDX file wfp_food_prices_npl.csv, 48,535 rows).
Only coverage, duplicates, units and counts were examined before this plan (script audit1.py). No forecast, error or model output had been produced.

## A. Panel definition (rule-based, from coverage only)
- Window: April 2019 to September 2026 (90 calendar months, index t = 0 to 89). July 2024 (t=63) and December 2024 (t=68) have no data for any market.
- Series = market x commodity. Core series: at least 60 observed months in the window. This yields markets with long coverage (24 expected) and 21 commodities (sugar, salt and mustard leaves excluded: 9 months).
- Target: natural log of retail price in NPR in the unit given in the file.

## B. Forecast task
- Origin t, horizon h in {1, 3, 6} months. Forecast log price at t+h using data up to and including t.
- A triple (series, origin, h) is scored only if the price at t and the price at t+h are both observed.
- First origin t = 35 (after 36 months). Last origin = 89 - h. All monthly origins are scored.
- Models are refit every 3rd origin (t = 35, 38, ...) for pooled models and reused for the two following origins. Direct strategy: one model per horizon.
- Statistical per-series models (ETS, SARIMA) are fit on every 3rd origin only; their comparison with other models uses only those origins (same triples for every model compared).

## C. Models (all parameters fixed here; no tuning on test periods)
1. Naive: log price at t.
2. Seasonal naive: log price at t+h-12; if missing use t+h-13 or t+h-11 (nearest); if still missing, fall back to naive (fallback share reported).
3. Seasonal change mean (SCM): log p_t plus the mean of past same-calendar-month changes (log p_{m+h} - log p_m) for pairs with target known by t; at least 2 pairs, otherwise naive.
4. ETS: damped additive trend, no seasonality, on observed log prices (statsmodels), fitted per series; fallback to naive if fitting fails (count reported).
5. SARIMA: order (1,1,1)(1,0,0,12) on log price, per series; fallback to naive if fitting fails.
6. Ridge: pooled, alpha = 1.0, same features as LightGBM, one-hot commodity and province, median imputation fitted on training rows.
7. LightGBM (main ML model): pooled over series, objective L1, 200 trees, learning rate 0.05, 15 leaves, min child samples 40, subsample 0.8 (every iteration), column sample 0.8, three seeds (11, 22, 33); predictions averaged over seeds; per-seed results also reported.
Target of pooled models: log p_{t+h} - log p_t. Training rows: all (series, s) with s + h <= t, target observed, price at s observed.

## D. Features (causal, computed from data up to the origin only)
Log price differences log p_s - log p_{s-k} for k = 1, 2, 3, 6, 12 (missing stays missing); rolling means of log price over 3 and 6 months relative to log p_s; standard deviation of monthly log changes over the last 6 months; SCM feature for horizon h; sine and cosine of calendar month of origin; commodity; province; latitude; longitude; cross-market mean 1-month and 3-month log change of the same commodity over all other markets observed at origin. No exogenous series are used.

## E. Metrics
- Scaled error: |forecast - actual| in log points divided by the in-sample mean absolute h-step change of the series using pairs known at the origin (MASE with an h-step naive scale). Series with fewer than 12 scale pairs are excluded from MASE (count reported).
- Primary summary: mean MASE over series at each origin, then mean over origins. Ratio to naive MASE reported.
- Secondary: MAE in log points; MAE in NPR (exp(forecast) vs actual); relative MAE (geometric mean over triples of absolute error ratio to naive; ties handled by adding 1e-6).
- Breakdowns: horizon, commodity category, province, and market isolation (mean distance to the 3 nearest other core markets from file coordinates; top tertile = high isolation).

## F. Unseen-market evaluation
- Leave-one-market-out (LOMO) over all core markets, and leave-one-province-out (LOPO). Held-out market data are never in training rows. Refit every 6th origin.
- Variant A (transfer with own history): the held-out market's own lags and rolling features are inputs, cross-market features exclude the held-out market; market-level identifiers (province, latitude, longitude) are kept as inputs. Compared with the same model trained with the market included (seen) on the same triples. Reported: MASE, ratio unseen/seen, difference by isolation tertile.
- Variant B (cold start): no own price history is used. Target: log p_{t+h} minus the mean log price of the same commodity over the other markets at t (reference). Features: reference level, province-excluded mean, mean of the 3 nearest other markets at t, latitude, longitude, commodity, month. Baselines: reference at t (B0), mean of the 3 nearest markets at t (B1). Scored by MAE in log points on the level of log price.

## G. Cleaning sensitivity
Rule (causal): a value is flagged if its absolute log change from the previous observed value exceeds log(1.5) and the following observed value (only if known at the time of cleaning) returns to within 15 percent of the value before the jump. The last observed month is never flagged. Flagged values are set to missing, not interpolated.
- Scenario A (primary): raw data.
- Scenario B: cleaned inputs and training data, raw targets.
- Scenario C: cleaned inputs, and flagged targets removed from scoring (flag determined using the target's neighbours; evaluation only).
Scenarios B and C run for naive, seasonal naive, SCM, Ridge and LightGBM at all horizons.

## H. Inference
- Diebold-Mariano test with Harvey small-sample correction on origin-level mean absolute scaled error differences, long-run variance with Newey-West lag h-1.
- Moving-block bootstrap over origins (block length 6, 2,000 resamples) for confidence intervals on MASE ratios.
- Holm correction across the pre-listed comparisons: LightGBM vs each of naive, seasonal naive, SCM, ETS, SARIMA, Ridge, at each horizon.
- Report all outcomes, whether or not they favour machine learning.

## I. Deviation policy
Any change made after test errors are seen is recorded in a deviations log with the reason, and results affected are labelled exploratory.
