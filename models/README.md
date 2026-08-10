# Models — production artifacts (10 features)

The trained model files that the [inference API](../api/) serves in production. These are the **10-feature** artifacts (selected by ablation from an initial 78 — see notebook `06_feature_ablation`), deliberately published for this academic project.

## Files

| File | What it is | Format |
|---|---|---|
| `xgb_final_20260808.pkl` | XGBoost classifier (10 features) | joblib |
| `rf_final_20260808.pkl` | Random Forest (trained for comparison, **not** in the production ensemble) | joblib |
| `lstm_final_20260808.keras` | LSTM network — reads 30-step sequences of the 10 scaled features | Keras |
| `scaler_lstm_20260808.pkl` | `StandardScaler` fitted on the training features (for the LSTM only) | joblib |
| `feature_cols_selected.json` | the ordered list of the **10** features the models expect | JSON |
| `models_latest.json` | pointer file mapping logical names → the current filenames | JSON |
| `ml_config.json` | service config (threshold 0.97, etc.) used by the API | JSON |

> All model artifacts expect **10 features** (`xgb.n_features_in_ == 10`, `scaler.n_features_in_ == 10`). The order in `feature_cols_selected.json` is authoritative.

## The production model

- **Ensemble = `0.5 × XGBoost + 0.5 × LSTM`.** The Random Forest is included for comparison only.
- **XGBoost** takes the 10 **raw** (unscaled) features — it was trained on the raw scale; scaling would break its tree thresholds.
- **LSTM** takes 30-candle sequences of the **scaled** features (via `scaler_lstm.pkl`). For a row at time *i*, the sequence is the 30 candles **before** *i*, per symbol, sorted by time.
- **Decision threshold:** `0.97` on the ensemble probability.
- **Label the models were trained on:** triple-barrier — TP +0.95% / SL −0.5% / 1h horizon.

## How they are produced and served

Produced by the notebooks (`01`→`06`): feature engineering → EDA → model comparison → ablation (78→10) → LSTM → backtest. In production they are stored in a Supabase Storage bucket (`ml-models`) and downloaded by the API at start-up. Note: `feature_cols_selected.json` is uploaded to the bucket **as** `feature_cols.json`.

## Reproducing a prediction

To score the ensemble the way the API does:
1. Compute the 10 features from raw 5-min candles.
2. `xgb_prob = xgb.predict_proba(raw_features)[:, 1]`
3. Build 30-step sequences of `scaler.transform(features)`, then `lstm_prob = lstm.predict(seqs)`
4. `ensemble_prob = 0.5 * xgb_prob + 0.5 * lstm_prob`; act if `≥ 0.97`.

The dashboard (`app/app_streamlit.py`, function `compute_ensemble_probas`) is a working reference implementation of exactly this.
