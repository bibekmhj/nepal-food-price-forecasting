# Forecasting retail food prices in Nepal with machine learning

Code, frozen analysis plan, deviation log, forecast outputs, result tables and figures for the manuscript
"Forecasting Retail Food Prices in Nepal with Machine Learning: A Rolling-Origin and Unseen-Market Evaluation"
(Bibek Maharjan). Manuscript status: in preparation. [Add the article DOI here once published.]

The study compares a pooled LightGBM model with naive, seasonal naive, seasonal change mean, ETS, SARIMA and
ridge forecasts of monthly retail food prices from the World Food Programme (WFP) price file for Nepal
(477 series, 24 markets, 21 commodities, April 2019 to September 2026), at horizons of 1, 3 and 6 months,
with rolling origins, leave-one-market-out and leave-one-province-out tests, and three data cleaning scenarios.

## Contents

| Folder | What it holds |
|---|---|
| `src/` | Python scripts (forecasting library `fc.py`, model runs, analysis, figures, diagnostics) |
| `data/` | Snapshot of the WFP Nepal price file used in the paper (see `NOTICE` for source, licence and SHA-256) |
| `results/` | Forecast arrays (`*.npz`) and every result table (`*.csv`) behind the manuscript |
| `figures/` | The six figures (300 dpi PNG) |
| `docs/` | Frozen analysis plan (v1) and the deviations log |

## Setup

Python 3.13 was used. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All scripts read and write paths relative to the repository root, so run them from there.

## Reproduce the results

Check that the data file is the one used in the paper:

```bash
sha256sum data/wfp_food_prices_npl.csv
# 7270e1463d195ddea8e37b8497ed854164d976c7d8ad699937aa94246bcc257b
```

Step 1, model runs (about 15 to 20 minutes in total on a 2-core machine; they overwrite the files in `results/`):

```bash
python src/run_main.py A naive,snaive,scm,ridge,lgbm   # primary scenario, all monthly origins
python src/run_main.py B naive,snaive,scm,ridge,lgbm   # cleaned inputs
python src/run_main.py C naive,snaive,scm,ridge,lgbm   # same forecasts as B, scored differently in step 2
python src/run_stat.py                                  # ETS and SARIMA on every third origin
python src/run_unseen.py LOMO                           # leave-one-market-out
python src/run_unseen.py LOPO                           # leave-one-province-out
```

Step 2, analysis and tables:

```bash
python src/analyze_main.py      # main_all_origins.csv, main_stat_origins.csv, dm_tests.csv, fallbacks.json
python src/analyze_break.py     # breakdown_A.csv (by category, province, isolation, commodity)
python src/analyze_clean.py     # cleaning_sensitivity.csv
python src/analyze_unseen.py    # unseen_summary.csv, unseen_tests.csv, unseen_groups.csv
python src/make_tables.py       # unseen_tests_holm.csv and panel summary tables
python src/diagnostics.py       # diagnostics_stickiness.csv, diagnostics_by_year.csv (post hoc)
python src/leakage_check.py     # confirms features use only data available at each origin
python src/figs.py              # figures/*.png
```

To check the analysis step without refitting the models, run step 2 only. The forecast arrays needed for it are
already in `results/`, and the output tables are identical to the committed ones.

## Notes

- Equations and the rolling-origin procedure are written out in Section 2 of the manuscript (Equations 1 to 29, Algorithm 1).
- Random seeds: LightGBM seeds 11, 22, 33 (main runs) and 11 (LOMO and LOPO).
- `src/profile.py`, `src/checks.py` and `src/audit1.py` are the data-audit scripts run before the plan was frozen. They take the CSV path as an argument.
- The WFP file is updated monthly. Downloading it later gives more months and will change the numbers.

## Licence

Code, results and figures: Apache License 2.0 (`LICENSE`). Data: WFP, CC BY-IGO, reproduced unchanged with attribution (`NOTICE`).
