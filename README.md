# AI for Trading — an ML filter on top of a signal-based crypto engine

Final project · **Jedha Bootcamp — Data Science Fullstack & Lead**

An ML layer that sits on top of a live, rule-based crypto trading engine and decides **when to buy** — by predicting the probability that a given buy will win, and letting through only the most confident ones.

**Interactive dashboard (Streamlit):** https://jedha.carlos-isturiz.fr

> The trained **models** are included in [`models/`](models/) and the inference **API** in [`api/`](api/). The large raw datasets are not committed (size), but the notebooks show how they are built. Notebooks are meant to be read, not re-run as-is.

---

## The problem

I run a **live crypto trading engine**. Every 5 minutes it looks at the 10 markets it actively trades (out of 19 tracked), fuses technical signals (trend, momentum, volatility) with external ones (whale movements, crypto news, a fear & greed index) into a single score, and decides to **buy, hold or sell**.

On its own, the engine is **right less than half the time**: its signal win rate is **~45.7%**. It stays profitable only because its winners are about **twice as large** as its losers. The question of this project: **can a machine-learning layer raise the win rate of its buys?**

## The solution

A supervised model scores every candidate buy from 0 to 100% (its probability of winning). The engine only executes a buy when that probability clears a confidence bar (**0.97**). Selling stays 100% rule-based (take-profit, stop-loss, trailing stop) — the AI lifts the *win rate*, the classic rules protect the *amounts*.

```
SIGNALS ──▶ ENGINE ──▶ [ AI LAYER: "is this really a good moment?" ] ──▶ DECISION
           (score)         blocks weak buys                              buy / hold / sell
```

The AI **confirms or vetoes** — it never creates a trade on its own.

## The data

The running engine continuously logs everything it does to a database (Supabase). That **operational logbook** is what powers live monitoring — it is built for *watching the system run*, not for training a model. So, by design, I keep two separate data worlds: the operational logbook on one side, and a dedicated **training dataset built from the raw 5-minute price history** on the other. A model needs clean, consistent, leak-free labels over the whole period, which is exactly what a purpose-built dataset guarantees. It is assembled in three steps:

1. **Collect & clean** — one tidy row per 5-minute moment, per crypto.
2. **Enrich** — **10** indicators describing the recent past (trend, momentum, volatility, oversold-ness…), kept from an initial 78 by ablation (see below).
3. **Label** — a 0/1 answer via the **triple-barrier method**: pretend to buy, watch the next hour (12 candles); hit **+0.95%** first = win (1), hit **−0.5%** first (or nothing happens) = loss (0).

Result: **500k+ labeled moments**, split **by time** (~80% train / ~20% test): the model learns on the past and is judged only on later months it never saw. A temporal split, not random — otherwise near-identical neighbouring candles would let it "see the future".

## Exploratory findings

The strongest, most stable pattern is **mean reversion**: win rate is highest right after a price drop (deeply oversold). It reaches **~63%** at very low Stochastic, and combining conditions sharpens it further (e.g. `RSI<30 & Stoch<20` ≈ **65%**). The edge holds month after month over the training period, but cryptos are heterogeneous and the effect fades in strong bear markets — so the model has to learn and combine these signals rather than rely on a single hand-written rule.

## Models & results

Trained and compared on the **out-of-sample test set**: **Logistic Regression → Random Forest → XGBoost → LSTM**, plus their ensemble.

| Model | AUC (test) |
|---|---|
| Logistic Regression | 0.767 |
| Random Forest | **0.870** — best single model |
| XGBoost | 0.866 |
| LSTM | 0.855 |
| **Ensemble (XGBoost + LSTM)** | **0.897** |

Because only ~10% of moments are winners, a lazy model that predicts "loss" every time is right 90% of the time — a useless 90% accuracy (F1 = 0). Models are therefore judged on **F1 and AUC**, not raw accuracy. Random Forest is the best *single* model, but the **XGBoost + LSTM ensemble** does better (0.897) and is what runs in production. A **data-leakage check** confirms the model doesn't cheat: removing the current-candle features drops AUC only from 0.86 to 0.82.

### Fewer features, same power (ablation)

The model was **cut from 78 features down to 10** with virtually no loss of quality (AUC 0.866 vs 0.872). The reason is a production constraint: the live service only receives 5-minute candles, so features that needed outside data (higher-timeframe history, a fear & greed index) can't be reproduced in real time. The 10 kept features are **100% pure-5m** — so training and production see exactly the same inputs, with no train/serving skew.

### Does it work?

Three win rates that **must not be stacked** (different targets and definitions): **~10%** buying blindly · **45.7%** the engine's signal · **72%** the AI filter — the ensemble at its **0.97** confidence bar. The confidence filter is the mechanism: the higher the bar, the higher the win rate and the fewer the trades (from **46%** at 0.80 to **81%** at 0.98). Measured on the **full test period** — the 72% holds because the ensemble stays *selective* when the market turns (it trades less, not badly).

### What it could earn — and on how little capital

On the test set, **€500 per trade**, net of ~0.18% costs (target Aerodrome routing):

| Model | Win rate | Trades | Net P&L |
|---|---|---|---|
| XGBoost | 49% | 27,477 | +€30,100 |
| Random Forest | 43% | 29,516 | +€21,300 |
| LSTM | 42% | 23,957 | +€14,400 |
| **Ensemble** | **72%** | **14,514** | **+€33,600** |

Every model is profitable, but the **ensemble earns the most with far fewer trades** (edge per trade ~0.46%, about double any single model — it only acts when XGBoost and the LSTM agree). And it runs on **modest capital**: trades are short (~17 min), so rarely more than ~1 runs at once (6-month peak: 19). So **~€9,500 covers every trade**, and €4,500 still captures 98% of them (a +747% return on that capital), at ~81 trades/day. A backtest projection under the target low-cost routing — promising, not a guarantee.

## Honest caveats

- **The model looks better in cross-validation than it really is.** In cross-validation the score is very high (~0.996), but on a genuinely unseen later period the ensemble scores **~0.90** (0.897). The gap comes from consecutive 5-minute candles being almost identical, which lets a little information "leak" between neighbouring slices without an embargo. The number to trust is the one measured on the later, out-of-sample **test** period (0.897) — still strong.
- **The backtest simplifies reality in three ways.** (1) The model is trained assuming a trade closes within about an hour, whereas the live engine actually holds a position ~7h on average (it lets winners run with a trailing stop). (2) The training target (+0.95% / −0.5%) is slightly different from the engine's live settings (+0.8% / −0.4%). (3) The raw returns are before costs; the ~0.18% on-chain cost per trade is subtracted afterwards, at evaluation. None of these break the results — but they mean the backtest is a solid **estimate**, not a guarantee.
- Results are measured on **past data** — promising, not proven. Real validation requires small-stake live trading.

## Deployment

**How the model is served and called.** The model runs as a small **FastAPI** service on **Railway**, exposing a `/predict` endpoint. Every 5 minutes, when the engine is about to buy, it sends the service the **last 400 raw 5-minute candles** (OHLCV). The service itself rebuilds the technical indicators, runs an **ensemble of XGBoost + LSTM** (50/50), and returns the **win probability** in a few milliseconds. The engine compares that probability to its confidence bar and buys above it — and, crucially, **falls back to its own rule-based decision if the service is unreachable** (5-second timeout), so a model outage never stops trading. The models themselves live in a Supabase Storage bucket and are downloaded by the service at start-up.

```
        every 5 min, on a candidate buy
ENGINE ───────── 400 raw 5-min candles ─────────▶ FastAPI service (Railway)
   ▲                                                 builds features
   │                                                 XGBoost + LSTM ensemble
   └───────────────── win probability (0–1) ◀────────────────────┘
   │
   └─▶ buy if probability ≥ confidence bar   (else fall back to rules)
```

**Known limitation (stated honestly):** the `/predict` endpoint is currently a **public URL with no authentication** — it is the single decision authority on the live path, so adding an API key (or network-level restriction) is a planned security improvement.

**The two pieces:**
- **ML service** — the ensemble, wrapped in FastAPI, deployed on **Railway**; called by the engine on every candidate buy. → [`api/`](api/)
- **Dashboard** — this project's analytics front-end: **Streamlit** behind **nginx** with HTTPS on a **VPS** → https://jedha.carlos-isturiz.fr

Full step-by-step deployment (Railway service + VPS dashboard) is documented in [`deployment/`](deployment/). The published production models are in [`models/`](models/).

## Next steps

1. **Make it reliable** — finish the live engine↔AI pipeline in production.
2. **Prove it for real** — validate carefully with small real money.
3. **Extend to selling** — today the AI only decides buys.
4. **Ride the trends** — a prudent simulation suggests that letting winners run (trailing stop) instead of exiting at +0.95% could lift the net gain per trade materially; the next model should be trained for that longer-horizon exit.

## Tech stack

Python · pandas · scikit-learn · XGBoost · TensorFlow/Keras (LSTM) · Streamlit · FastAPI · Railway · nginx · Supabase

## Repository structure

## Repository structure

```
├── api/                      # FastAPI inference service (served on Railway)
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── models/                   # published production models (10 features)
│   ├── xgb_final_20260808.pkl · rf_final_20260808.pkl · lstm_final_20260808.keras
│   ├── scaler_lstm_20260808.pkl · feature_cols_selected.json · models_latest.json · ml_config.json
│   └── README.md
├── deployment/
│   └── README.md             # full deployment: Railway service + VPS dashboard
├── app/
│   └── app_streamlit.py      # the interactive dashboard (technical + simplified views)
├── notebooks/                # 01 feature engineering · 02 EDA · 03 ML models
│   └── ...                   # 04 backtest · 05 deep learning · 06 feature ablation
├── requirements.txt
└── README.md
```

## Data privacy (RGPD)

This project uses **market data only** (crypto prices and derived technical indicators) — **no personal data** is collected or processed, so it falls outside the scope of RGPD personal-data obligations. The trained models are included; only **credentials/secrets** and the large raw datasets are deliberately kept out of this repository.

---

*Solo final project · Jedha Bootcamp · 2026.*
