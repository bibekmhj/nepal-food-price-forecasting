# Deviations from the frozen analysis plan

The plan (docs/analysis_plan_v1.md, SHA-256 87d2c5854987cc13d682c71f757273d1c65ea7caa56339cb399b348f4df28ac3)
was written on 19 November 2026 before any forecast error was computed. Every change made after test errors
were seen is listed here.

- D1  The leave-one-market-out (LOMO) and leave-one-province-out (LOPO) models use one random seed (11) instead of three, to limit computing cost.
- D2  ETS is fitted on linearly interpolated series inside the observed range because the statsmodels routine cannot take missing values. SARIMA uses missing values directly.
- D3  Divergent ETS and SARIMA forecasts (not finite, or more than 1.0 log point from the naive forecast) are replaced by the naive forecast. Shares are in results/fallbacks.json.
- D4  The Diebold-Mariano variance uses Bartlett weights with lag h - 1. Origins with no scoreable forecast (July and December 2024) are dropped without closing the gap in the time index. A first implementation was corrected before the final tables.
- D5  The plan's "relative MAE" was replaced by the plain ratio of mean absolute errors to the naive forecast.
- D6  Holm families were defined after some outputs existed: all origins (12 comparisons), ETS/SARIMA origins (18), unseen markets variant A, unseen markets variant B.
- D7  Scenario C removes flagged targets using next-observation information. This is for evaluation only and is as stated in the plan.
- D8  Market isolation is a distance proxy and not a measure of remoteness.
- D9  results/dm_tests.csv column n_origins shows scheduled origins (54, 52, 49; ETS/SARIMA sets 18, 18, 17). The tests used only origins with scored forecasts (50, 48, 45; 16, 16, 15) because origins whose own month or target month is missing have no scoreable triples. The manuscript reports both counts.
- D10 The cold-start model (variant B) uses the nine features of Equation 29 of the manuscript: the cross-market mean, the nearest-three mean minus the cross-market mean, the 3-month changes of both means, latitude, longitude, commodity, and sine and cosine of the calendar month. The plan (section F) listed a province-excluded mean and did not list the 3-month changes.
- D11 Post-hoc descriptive diagnostics (src/diagnostics.py; share of unchanged consecutive prices, MASE by origin year) were added after the main analysis in response to expected reviewer questions. They are not part of the plan and carry no tests.
