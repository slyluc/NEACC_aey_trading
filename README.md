# Nitor Energy Case Competition — Minimal Solution

This repo contains my end-to-end pipeline to predict **hourly intraday electricity prices** across multiple markets (A–F).  
Primary evaluation metric: **RMSE**.

## Approach (high level)
- **Model:** LightGBM gradient-boosted trees
- **Market handling:** `market` is treated as a **categorical feature** (native LightGBM support)
- **Leakage-safe / causal feature engineering (computed within each market):**
  - Time features + cyclical encodings (hour / day-of-week / day-of-year / month)
  - “Physics/system” features:
    - `net_load_forecast = load_forecast - (solar_forecast + wind_forecast)`
    - degree-day proxies (cooling/heating)
    - wind shear approximation (80m − 10m)
    - wind direction sin/cos (circular feature)
  - Time-series structure:
    - lags (1, 2, 3, 6, 12, 24, 48, 168 hours)
    - rolling mean/std (6, 24, 48, 168)
    - EWM features (spans 24 and 168)
- **Validation:** fixed-horizon walk-forward CV (future block validation), stepped backwards in time
- **Recency weighting:** exponential weighting with tuned half-life (gives more weight to recent samples)
- **Hyperparameter tuning:** Optuna optimizing mean CV RMSE
- **Final predictions:** multi-seed ensemble (average predictions across multiple random seeds)

## Repo contents
- `FinalModel.ipynb` — full pipeline (features → CV → Optuna → final ensemble → submission)
- `final_submission_lgbm_improved.csv` — main submission output
- `submission_new_anton.csv` — alternative submission (used for stitching/ensembling)
- `aey_trading_submission.csv` — additional submission artifact

## How to run
1. Place `train.csv` and `test_for_participants.csv` in the project folder.
2. Open `FinalModel.ipynb` and **Run All**.
3. Output file is written as:
   - `final_submission_lgbm_improved.csv` (columns: `id`, `target`)

## Notes / guardrails
- No leakage: **no backward fill**. Only **forward fill within market** for raw weather inputs.
- LightGBM handles remaining NaNs naturally.
- The submission writer can be made fully generic by validating against `len(test_df)` and the IDs present in `test_for_participants.csv` (no hard-coded row counts or ID ranges).
