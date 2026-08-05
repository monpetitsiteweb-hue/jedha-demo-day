# AI for Trading — an ML filter on top of a signal-based crypto engine

Final project · **Jedha Bootcamp — Data Science Fullstack & Lead**

An ML layer that sits on top of a live, rule-based crypto trading engine and decides **when to buy** — by predicting the probability that a given buy will win, and letting through only the most confident ones.

**Interactive dashboard (Streamlit):** https://jedha.carlos-isturiz.fr

> ⚠️ The raw datasets and trained models are **not** included in this repository (they are proprietary and are the core value of the running system). The notebooks show the full pipeline and results; they are meant to be read, not re-run as-is.

---

## The problem

I run a **live crypto trading engine**. Every 5 minutes it looks at the 10 markets it actively trades (out of 19 tracked), fuses technical signals (trend, momentum, volatility) with external ones (whale movements, crypto news, a fear & greed index) into a single score, and decides to **buy, hold or sell**.

On its own, the engine is **right less than half the time**: its signal win rate is **~45.7%**. It stays profitable only because its winners are about **twice as large** as its losers. The question of this project: **can a machine-learning layer raise the win rate of its buys?**

## The solution

A supervised model scores every candidate buy from 0 to 100% (its probability of winning). The engine only executes a buy when that probability clears a confidence bar (**0.94**). Selling stays 100% rule-based (take-profit, stop-loss, trailing stop) — the AI lifts the *win rate*, the classic rules protect the *amounts*.

```
SIGNALS ──▶ ENGINE ──▶ [ AI LAYER: "is this really a good moment?" ] ──▶ DECISION
           (score)         blocks weak buys                              buy / hold / sell
```

The AI **confirms or vetoes** — it never creates a trade on its own.

## The data

The running engine continuously logs everything it does to a database (Supabase). That **operational logbook** is what powers live monitoring — it is built for *watching the system run*, not for training a model. So, by design, I keep two separate data worlds: the operational logbook on one side, and a dedicated **training dataset built from the raw 5-minute price history** on the other. A model needs clean, consistent, leak-free labels over the whole period, which is exactly what a purpose-built dataset guarantees. It is assembled in three steps:

1. **Collect & clean** — one tidy row per 5-minute moment, per crypto.
2. **Enrich** — 78 indicators describing the recent past (trend, momentum, volatility, oversold-ness…). 77 are computed from prices; 1 is the external fear & greed index.
3. **Label** — a 0/1 answer via the **triple-barrier method**: pretend to buy, watch the next hour (12 candles); hit **+0.95%** first = win (1), hit **−0.5%** first (or nothing happens) = loss (0).

Result: **500k+ labeled moments**, split into a training set and a later, unseen test set (~213k rows) — no future information leaks into any feature.

## Exploratory findings

The strongest, most stable pattern is **mean reversion**: win rate is highest right after a price drop (deeply oversold). It reaches **~63%** at very low Stochastic, and combining conditions sharpens it further (e.g. `RSI<30 & Stoch<20` ≈ **65%**). The edge holds month after month over the training period, but cryptos are heterogeneous and the effect fades in strong bear markets — so the model has to learn and combine these signals rather than rely on a single hand-written rule.

## Models & results

Trained and compared on the out-of-sample test set: **Logistic Regression → Random Forest → XGBoost → LSTM**, plus an ensemble.

| Model | Quality score (0.5 = random, 1 = perfect) |
|---|---|
| Logistic Regression | 0.70 |
| Random Forest | **0.88** |
| XGBoost | 0.86 |
| LSTM | 0.85 |

Because only ~10% of moments are winners, a lazy model that simply predicts "loss" **every single time** is right **90% of the time** — a 90% accuracy that is completely useless, since it never catches a single winner (its F1-score is 0). This is why the models are judged on **F1 and AUC**, which reward actually finding the rare winners, rather than on raw accuracy.

**The confidence filter is the core idea.** The model gives every candidate buy a score from 0 to 100% — how confident it is that the trade will win. We then set a **bar** and act only on buys that score above it. The higher the bar, the more selective we are: the win rate goes up, but fewer trades qualify. At our **0.94** bar (the model must be at least 94% sure):

| | Win rate | Trades (test) | Net / trade (after ~0.18% costs) |
|---|---|---|---|
| **XGBoost** | 52% | 25,851 | +0.18% |
| **Random Forest** | 75% | 11,135 | +0.39% |

Three win rates that **must not be stacked** (they use different targets and definitions): **~10%** buying blindly · **45.7%** the engine's signal · **52–75%** the AI filter.

## Honest caveats

- **The model looks better in cross-validation than it really is.** In cross-validation the score is very high (~0.997), but on a genuinely unseen later period it drops to ~0.87. The gap comes from consecutive 5-minute candles being almost identical, which lets a little information "leak" between neighbouring slices. The number to trust is the one measured on the later, out-of-sample **test** period (~0.87) — which is still strong.
- **The backtest simplifies reality in three ways.** (1) The model is trained assuming a trade closes within about an hour, whereas the live engine actually holds a position ~7h on average (it lets winners run with a trailing stop). (2) The training target (+0.95% / −0.5%) is slightly different from the engine's live settings (+0.8% / −0.4%). (3) The raw returns are before costs; the ~0.18% on-chain cost per trade is subtracted afterwards, at evaluation. None of these break the results — but they mean the backtest is a solid **estimate**, not a guarantee.
- Results are measured on **past data** — promising, not proven. Real validation requires small-stake live trading.

## Deployment

**How the model is served and called.** The trained model is wrapped in a small **FastAPI** web service and deployed on **Railway**, exposing a single prediction endpoint. Every 5 minutes, when the engine is about to buy, it sends the 78 features of the current moment to that endpoint over an authenticated HTTPS request (the engine holds a secret API key that the service checks, so no one else can call it). The service loads the model once at start-up, runs the prediction, and returns the win probability in a few milliseconds. The engine then executes the buy only if that probability clears the 0.94 bar — and, crucially, **falls back to its own rule-based decision if the ML service is unreachable**, so a model outage never stops trading.

```
        every 5 min, on a candidate buy
ENGINE ───────── 78 features (HTTPS + API key) ─────────▶ FastAPI model service (Railway)
   ▲                                                              │
   └───────────────── win probability (0–1) ◀────────────────────┘
   │
   └─▶ buy only if probability ≥ 0.94   (else fall back to rules)
```

**The two pieces:**
- **ML service** — the model, wrapped in FastAPI, deployed on **Railway**; called by the engine on every candidate buy.
- **Dashboard** — this project's analytics front-end: **Streamlit** behind **nginx** with HTTPS on a **VPS** → https://jedha.carlos-isturiz.fr

## Next steps

1. **Make it reliable** — finish the live engine↔AI pipeline in production.
2. **Prove it for real** — validate carefully with small real money.
3. **Extend to selling** — today the AI only decides buys.
4. **Ride the trends** — a prudent simulation suggests that letting winners run (trailing stop) instead of exiting at +0.95% could lift net gain/trade from ~0.2% to **+1.3–2.4%**; the next model should be trained for that longer-horizon exit.

## Tech stack

Python · pandas · scikit-learn · XGBoost · TensorFlow/Keras (LSTM) · Streamlit · FastAPI · Railway · nginx · Supabase

## Repository structure

```
├── app/
│   └── app_streamlit.py      # the interactive dashboard (technical + simplified views)
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_backtest.ipynb
│   └── 05_deep_learning.ipynb
├── requirements.txt
└── README.md
```

## Data privacy (RGPD)

This project uses **market data only** (crypto prices and derived technical indicators) — **no personal data** is collected or processed, so it falls outside the scope of RGPD personal-data obligations. The raw datasets, trained models and any credentials are deliberately kept out of this repository.

---

*Solo final project · Jedha Bootcamp · 2026.*