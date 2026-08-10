# ML Inference API — Savvy Crypto

FastAPI service that serves the trading model (an **XGBoost + LSTM ensemble**) to the live trading engine. Given a stream of raw 5-minute candles, it computes the features itself, runs the ensemble, and returns a win probability for a candidate buy.

- **Framework:** FastAPI (served by Uvicorn)
- **Hosting:** Railway (Docker build)
- **Public URL:** `https://savvy-crypto-ml-production.up.railway.app`
- **Python:** 3.11 · TensorFlow 2.21.0 · XGBoost 2.0.3 · scikit-learn 1.7.2 · FastAPI 0.115.0 · Supabase 1.2.0

---

## Architecture

```
                       POST /predict (every 5 min, on a candidate buy)
Trading engine ───────── ~400 raw 5-min candles ─────────▶ FastAPI service (Railway)
      ▲                                                       1. build features from candles
      │                                                       2. ensemble = 0.5·XGBoost + 0.5·LSTM
      └──────────── win probability + signal ◀───────────────  3. compare to threshold (0.97)
      │
      └─▶ buy if signal, else fall back to the engine's own rules
```

The engine sends **raw OHLCV candles, not features** — all feature engineering happens inside the service, so training and serving use exactly the same code path. The model uses **10 features** (100% pure-5m, selected by ablation from 78). The ensemble is `0.5 × XGBoost + 0.5 × LSTM`; a Random Forest is loaded for comparison but **not** used in the decision. Decision threshold: **0.97**. Label target during training: triple-barrier (TP +0.95% / SL −0.5% / 1h horizon).

---

## Endpoints

### `POST /predict`
Runs the ensemble on a candidate buy.

**Request**
```json
{
  "symbol": "BTC-USD",
  "candles": [
    { "candle_time": "2026-08-08T09:00:00Z", "open_price": 61234.1, "high_price": 61290.0, "low_price": 61180.5, "close_price": 61255.7, "volume": 12.34 }
    // ... ~400 candles recommended (min ~350: EMA200 and other indicators need warm-up)
  ]
}
```

**Response**
```json
{
  "symbol": "BTC-USD",
  "signal": true,
  "ensemble_prob": 0.982,
  "xgb_prob": 0.990,
  "lstm_prob": 0.974,
  "stoch_k": 8.2,
  "rsi14": 27.4,
  "would_filter": false,
  "eda_signal": true,
  "threshold_used": 0.97,
  "timestamp": "2026-08-08T09:00:03Z"
}
```
`signal` is `true` when `ensemble_prob ≥ threshold_used`. `xgb_prob` / `lstm_prob` are the individual probabilities that make up the ensemble.

### `GET /healthz`
Service health — returns `models_loaded`, `feature_count` (should be `10`), `ensemble_threshold` (`0.97`).

### `GET /`
Service metadata.

---

## Deployment (Railway)

The service is built from the **Dockerfile** (`railway.toml` → `builder = "dockerfile"`), and started with `CMD ["python", "main.py"]`, which launches Uvicorn on `0.0.0.0:$PORT`.

**Environment variables** (names only — never commit values; see `.env.example`):

| Variable | Purpose |
|---|---|
| `SUPABASE_URL` | Supabase project hosting the `ml-models` bucket |
| `SUPABASE_KEY` | `service_role` key to download the bucket — **secret**, env only |
| `MODELS_DIR` | optional, defaults to `./models` |
| `PORT` | injected by Railway |

**Model loading at start-up.** On boot the service downloads 6 files from the Supabase Storage bucket `ml-models` and loads them into memory (`joblib` for `.pkl`, `keras.models.load_model` for `.keras`):

```
feature_cols.json   ml_config.json   xgb_final.pkl   rf_final.pkl   scaler_lstm.pkl   lstm_final.keras
```

There is **no hot-reload**: updating a model means uploading the new file to the bucket **and** restarting the Railway service. Note: the local `feature_cols_selected.json` (10 features) is uploaded to the bucket **as** `feature_cols.json` — that file fixes the number of features fed to the models.

**Run locally**
```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in your own Supabase values
python main.py                # serves on http://0.0.0.0:8000
```

---

## Security & limitations (stated honestly)

- **The `/predict` endpoint is public and unauthenticated.** It is the single buy-decision authority on the live path, so this is a real limitation — **adding an API key** (or network-level restriction) is a planned improvement.
- **Secrets are never hard-coded.** `SUPABASE_URL` / `SUPABASE_KEY` are read from environment variables; `.env` is git-ignored and `.env.example` ships only placeholder values.
- **Robust fallback.** The engine calls `/predict` with a 5-second timeout and a single attempt (no retry). If the service is unreachable, the engine falls back to its own rules and keeps trading — a model outage never stops the system.

---

## How the model is trained

The training pipeline (feature engineering, EDA, model comparison, ablation 78→10, LSTM, backtest) is in the project notebooks **01–06**. This service only performs **inference** with the artifacts produced there. The published model files are in [`../models/`](../models/).
