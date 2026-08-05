"""
==============================================================================
  AI FOR TRADING — Jedha final project (deck + dashboard)
==============================================================================
Tabs navigation · optional slide-by-slide "Presentation mode".
Plotly on key charts · matplotlib elsewhere.

RUN :
    pip install streamlit pandas numpy scikit-learn xgboost matplotlib joblib
    pip install plotly
    streamlit run app_streamlit.py

Theme -> .streamlit/config.toml (provided) · Screenshot -> assets/config_panel.png
==============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ---- config ----------------------------------------------------------------
# Local (Windows PC) and server (VPS Linux) paths — pick whichever exists
_LOCAL_PROCESSED = r"C:\Users\casir\Desktop\CRYPTOS\data\processed"
_LOCAL_MODELS    = r"C:\Users\casir\Desktop\CRYPTOS\models"
_VPS_PROCESSED   = "/opt/jedha-app/data/processed"
_VPS_MODELS      = "/opt/jedha-app/models"
DEFAULT_PROCESSED = _LOCAL_PROCESSED if os.path.isdir(_LOCAL_PROCESSED) else _VPS_PROCESSED
DEFAULT_MODELS    = _LOCAL_MODELS if os.path.isdir(_LOCAL_MODELS) else _VPS_MODELS
GITHUB_URL = "https://github.com/<your-username>/ai-for-trading"   # TODO: replace
SCREENSHOT = "assets/config_panel.png"
BREAKEVEN = 41.4

ENGINE = dict(
    winrate_signal=45.74, n=4233, window="Mar–Jul 2026",
    pnl_mean=0.107, pnl_median=-0.01, tp=0.8, sl=0.4,
    avg_win_pct=0.583, avg_loss_pct=-0.294,    # full data (decision_outcomes.realized_pnl_pct)
    reward_risk=1.98, decisions=686487, candles=749990, candles_symbols=19, candles_months=13,
    live_reward_risk=1.60,
    p90=0.60, p95=1.08, p99=2.99, max_win=15.26, max_loss=-2.71,
    mock=3966, live=267, real_executed=7, double_true=549,
    by_month=[("Mar", 834, 33.93), ("Apr", 3094, 49.13), ("May", 78, 60.26),
              ("Jun", 121, 33.88), ("Jul", 106, 42.45)],
    by_symbol=[("BTC", 651, 49.77), ("AVAX", 182, 51.65), ("SOL", 818, 47.19),
               ("ETH", 857, 46.44), ("XRP", 811, 44.76), ("ADA", 914, 40.59)],
)

METRICS = pd.DataFrame({
    "Model":     ["Naive (always LOSS)", "Logistic Regression", "Random Forest", "XGBoost", "LSTM", "Ensemble"],
    "Accuracy":  [0.896, 0.702, 0.883, 0.863, np.nan, np.nan],
    "Precision": [0.000, 0.229, 0.455, 0.407, np.nan, np.nan],
    "Recall":    [0.000, 0.791, 0.661, 0.704, np.nan, np.nan],
    "F1":        [0.000, 0.355, 0.539, 0.516, np.nan, np.nan],
    "AUC (test)":[0.500, 0.705, 0.884, 0.865, 0.854, 0.893],
}).set_index("Model")

st.set_page_config(page_title="AI for Trading — Jedha", layout="wide")

# ---- global styling: bigger fonts everywhere -------------------------------
st.markdown('''<style>
@import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");
.bi { color:#3BA99C; }
/* --- paragraph / list text (robust targeting) --- */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 1.28rem !important;
    line-height: 1.75 !important;
}
[data-testid="stMarkdownContainer"] p { margin-bottom: 0.7rem !important; }
/* captions a bit bigger too */
[data-testid="stCaptionContainer"] p { font-size: 1.02rem !important; }
/* tabs */
button[data-baseweb="tab"] { padding: 12px 22px; }
button[data-baseweb="tab"] p { font-size: 1.25rem !important; font-weight: 700 !important; }
/* metric numbers */
[data-testid="stMetricValue"] { font-size: 2.3rem !important; }
/* headers */
h1 { font-size: 2.3rem !important; }
h2 { font-size: 1.9rem !important; }
h3 { font-size: 1.5rem !important; }
.block-container { padding-top: 2rem; max-width: 1300px; }
</style>''', unsafe_allow_html=True)


# ---- loaders ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_test(processed_dir):
    latest = load_json(os.path.join(processed_dir, "latest.json"))
    test = pd.read_csv(os.path.join(processed_dir, latest["test"]))
    feats = load_json(os.path.join(processed_dir, latest["feature_cols"]))
    if isinstance(feats, dict):
        feats = feats.get("feature_cols", list(feats.values())[0])
    return test, feats, latest


@st.cache_resource(show_spinner=False)
def load_model(models_dir, filename):
    import joblib
    return joblib.load(os.path.join(models_dir, filename))


@st.cache_data(show_spinner=False)
def load_train(processed_dir):
    latest = load_json(os.path.join(processed_dir, "latest.json"))
    path = os.path.join(processed_dir, latest["train"])
    want = ["symbol", "candle_time", "rsi14", "stoch_k", "bb_pos", "price_vs_ema50", "label"]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in want if c in header]
    return pd.read_csv(path, usecols=cols)


@st.cache_data(show_spinner="Scoring the test set...")
def compute_test_probas(processed_dir, models_dir):
    import joblib
    t, feats, _ = load_test(processed_dir)
    ml = load_json(os.path.join(models_dir, "models_latest.json"))
    xgb = joblib.load(os.path.join(models_dir, ml["xgb"]))
    rf = joblib.load(os.path.join(models_dir, ml["rf"]))
    X = t[feats]
    return (xgb.predict_proba(X)[:, 1], rf.predict_proba(X)[:, 1], t["label"].values)


# ---- UI helpers ------------------------------------------------------------
def kpi_card(container, value, label, desc):
    container.markdown(
        f"""<div style="padding:8px 0 14px 0">
        <div style="font-size:2.7rem;font-weight:800;color:#3BA99C;line-height:1.05">{value}</div>
        <div style="font-size:1.12rem;font-weight:700;margin-top:6px">{label}</div>
        <div style="font-size:1.0rem;opacity:.72;margin-top:3px;line-height:1.4">{desc}</div>
        </div>""", unsafe_allow_html=True)


def winrate_by_zone(df, col, bins):
    d = df[[col, "label"]].dropna().copy()
    d["_z"] = pd.cut(d[col], bins=bins)
    return d.groupby("_z", observed=True)["label"].mean() * 100


def show_winrate(container, wr_or_labels, values=None, xlabel="", title="", rotate=45):
    if values is None:
        labels, values = list(wr_or_labels.index), list(wr_or_labels.values)
    else:
        labels = wr_or_labels
    labels = [str(l) for l in labels]
    colors = ["#3BA99C" if v >= BREAKEVEN else "#D98880" for v in values]
    if HAS_PLOTLY:
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
        fig.add_hline(y=BREAKEVEN, line_dash="dash", line_color="grey")
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Win rate (%)",
                          height=320, margin=dict(l=10, r=10, t=45, b=10))
        container.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.bar(labels, values, color=colors); ax.axhline(BREAKEVEN, color="grey", ls="--", lw=1)
        ax.set_ylabel("Win rate (%)"); ax.set_xlabel(xlabel); ax.set_title(title)
        plt.xticks(rotation=rotate, ha="right"); plt.tight_layout()
        container.pyplot(fig)


# ---- sidebar ---------------------------------------------------------------
st.sidebar.header("Configuration")
processed_dir = st.sidebar.text_input("/processed folder", DEFAULT_PROCESSED)
models_dir    = st.sidebar.text_input("/models folder", DEFAULT_MODELS)
st.sidebar.divider()
simple = st.sidebar.toggle("Simplified version (Demoday)", value=True,
                           help="A simple, story-driven version for a non-technical audience")

data_ok = False
try:
    test, FEATURE_COLS, latest = load_test(processed_dir)
    data_ok = True
    st.sidebar.success(f"Data loaded: {latest.get('date', '?')}")
except Exception as e:
    st.sidebar.error("Data not found. Check /processed.")
    st.sidebar.caption(f"Detail: {e}")

train_df = None
if data_ok:
    try:
        train_df = load_train(processed_dir)
    except Exception:
        train_df = None

xgb_model, rf_model, models_latest = None, None, None
if data_ok:
    try:
        models_latest = load_json(os.path.join(models_dir, "models_latest.json"))
        xgb_model = load_model(models_dir, models_latest["xgb"])
        st.sidebar.success("XGBoost model loaded.")
    except Exception as e:
        st.sidebar.warning("Model not loaded (live prediction limited).")
        st.sidebar.caption(f"Detail: {e}")
    try:
        rf_model = load_model(models_dir, models_latest["rf"])
    except Exception:
        rf_model = None
if not HAS_PLOTLY:
    st.sidebar.caption("Tip: `pip install plotly` for interactive charts.")


# ============================================================================
#  SECTIONS
# ============================================================================
def render_overview():
    st.header("Description of the project")
    with st.container(border=True):
        st.markdown(
            "I run a **live crypto trading engine**. Every **5 minutes** it looks at 10 crypto markets and "
            "decides whether to **buy, hold or sell**. Its decisions are driven by **technical signals** — "
            "trend (moving averages), momentum (RSI, MACD), volatility — combined with external signals like "
            "**whale movements**, **crypto news** and a **fear & greed** index. It **fuses** all of these into "
            "a single score and acts on it. Crucially, it has been **recording every signal, decision and "
            "outcome since launch** into a database."
        )
        st.markdown(
            "That gave me **two ideas**:\n\n"
            "1. **Add an AI layer to optimise the win rate** — a model that confirms or blocks the engine's buys.\n"
            "2. Since I have now **accumulated a large dataset**, I might have **enough data to train** such a "
            "model and plug it on top of the decision engine.\n\n"
            "**This project explores exactly that.**"
        )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("The engine today (before the AI layer)")
        a, b = st.columns(2)
        kpi_card(a, f"{ENGINE['winrate_signal']} %", "Signal win rate",
                 "Share of the engine's BUY signals that turned out profitable (realized P&L &gt; 0), "
                 "measured in shadow evaluation.")
        kpi_card(b, f"{ENGINE['tp']}% / {ENGINE['sl']}%", "Take-Profit / Stop-Loss",
                 "TP = the gain that closes a trade in profit. SL = the loss that closes it. "
                 "Here +0.8% / \u22120.4% \u2014 the engine aims to win about twice what it risks.")
        c, d = st.columns(2)
        kpi_card(c, f"{ENGINE['reward_risk']}×", "Reward / risk ratio",
                 "The key number: winners are on average ~2× bigger than losers "
                 f"(avg win +{ENGINE['avg_win_pct']}% vs avg loss {ENGINE['avg_loss_pct']}%). "
                 "This is what makes the engine viable despite a win rate below 50%.")
        kpi_card(d, f"+{ENGINE['avg_win_pct']}% / {ENGINE['avg_loss_pct']}%", "Avg win / avg loss",
                 "Average % move (vs entry price) of a winning signal versus a losing one. "
                 "The engine loses more often than it wins, but its wins are twice as large.")

        st.subheader("How the overlay works")
        st.markdown(
            "1. Every **5 minutes**, the engine pulls the last 300 candles and computes its signals.\n"
            "2. Safety **guards** can block everything (whale inflow, extreme fear/greed).\n"
            "3. The engine proposes a **BUY** via its rule-based fusion score.\n"
            "4. The **ML model** (Railway) returns a **probability of winning**.\n"
            "5. The BUY executes **only if that probability ≥ 0.97** — otherwise blocked. Fallback to the "
            "engine's own rules if the ML service is down.\n\n"
            "**The engine proposes, the ML confirms or vetoes** — it never creates a trade on its own."
        )
        st.markdown(f"<i class='bi bi-rocket-takeoff'></i> **Deployed on Railway**  &nbsp;·&nbsp;  <i class='bi bi-github'></i> [Project code on GitHub]({GITHUB_URL})", unsafe_allow_html=True)
        st.caption("(replace the GitHub link with your repo once created)")
    with right:
        if os.path.exists(SCREENSHOT):
            st.image(SCREENSHOT, caption="The engine's strategy configuration — the system that produces my data")
        else:
            st.info(f"Put your screenshot at `{SCREENSHOT}` to show the engine's config panel here.")

    st.subheader("How much data we have collected")
    e, f, g, h = st.columns(4)
    kpi_card(e, f"{ENGINE['decisions']/1000:.0f}k", "Decisions recorded",
             "Total engine decisions logged since launch (buy / hold / sell).")
    kpi_card(f, f"{ENGINE['candles']/1000:.0f}k", "Price candles",
             f"5-min price bars collected over ~{ENGINE['candles_months']} months — the raw material for the model.")
    kpi_card(g, f"{ENGINE['n']:,}", "Signals evaluated",
             "Signals with a measured outcome — the sample behind the win rate.")
    kpi_card(h, f"{ENGINE['candles_symbols']}", "Cryptos tracked",
             "Distinct coins recorded in the price history.")


def render_data():
    st.header("Where the data comes from")
    st.markdown("There are **two very different sources of data**, and they must not be confused. "
                "One is what the running engine writes down; the other is what we build ourselves for the AI.")

    # ---------------- WORLD A ----------------
    with st.container(border=True):
        st.subheader("Source A — the engine's logbook (stored in a Supabase database)")
        st.markdown(
            "As the engine runs, it **writes down everything it does** into an online database (Supabase). "
            "That logbook is split into four tables, all connected by a shared decision ID:"
        )
        st.table(pd.DataFrame({
            "Table": ["live_signals", "decision_events", "decision_snapshots", "decision_outcomes"],
            "What one row is": [
                "one signal the engine received",
                "one decision the engine took",
                "a full photo of the market at that moment",
                "what actually happened after the decision",
            ],
            "What's inside": [
                "the signal's type and strength, and where it came from",
                "which crypto, buy/hold/sell, how confident, entry price",
                "the combined score, the individual signals, the safety checks",
                "the profit or loss, and whether the target/stop was reached",
            ],
        }))
        st.markdown("**Why we do NOT use this logbook to train the model — it is unreliable:**")
        st.markdown(
            "- **It only imagines the trades.** Most of these results are *what-if* calculations — the engine "
            "asked \"would this have worked?\" **without actually placing the trade**. Only **7** results come "
            "from real, executed trades. So this tells us how good the *signal* is, not how much money was "
            "really made.\n"
            "- **Some results are broken.** In **549** rows, the record says the trade hit **both** its profit "
            "target **and** its loss limit at the same time — which is impossible. So we ignore those flags and "
            "trust only the **actual profit/loss** number.\n"
            "- **It's mostly a test account.** Almost all results (**3,966**) come from a demo/test account, and "
            "only **267** from the real one.\n"
            "- **The record has holes.** It was well filled in March–April 2026, then almost stopped. So only "
            "**March–April 2026** is trustworthy.\n"
            "- **Most decisions are 'do nothing'.** About **65 %** of decisions are \"hold\" (no trade), so they "
            "can't be scored. Out of everything, only about **4,233** signals were actually evaluated."
        )
        st.info("In short: this logbook is great to *watch the engine live*, but too incomplete and messy to "
                "*teach a model*. So for the AI, we rebuild our own clean data from scratch (Source B).")

    # ---------------- WORLD B ----------------
    with st.container(border=True):
        st.subheader("Source B — the clean dataset we build for the AI (from raw prices)")
        st.markdown(
            "Instead of the messy logbook, we start again from the **raw price history** — the simple record "
            "of crypto prices every 5 minutes. From it we build one clean table in three steps:"
        )
        st.markdown(
            "**Step 1 — collect & clean.** *Collect* = gather every 5-minute price bar for each crypto. "
            "*Clean* = remove duplicates, sort them in time order, and check none are missing. "
            "Result: one tidy row per 5-minute moment.\n\n"
            "**Step 2 — enrich.** For each moment, we compute 78 extra numbers that describe the market "
            "(explained just below).\n\n"
            "**Step 3 — label.** We add one final column, the **answer** we want to predict, written as "
            "**0 or 1**."
        )

        st.markdown("#### What does 'label 0/1' mean?")
        st.markdown(
            "It's simply a **yes/no answer** attached to each moment: **1 = 'buying here would have been a "
            "win'**, **0 = 'it would have been a loss'**. The model's whole job is to look at the 78 numbers "
            "and guess this 0 or 1. This is called **classification** (sorting each case into one of two boxes)."
        )

        st.markdown("#### How we decide if a moment is a 1 or a 0")
        st.markdown(
            "We pretend to buy at that moment, then we watch the **next hour** (the next 12 five-minute bars) "
            "and see what happens first:\n\n"
            "- If the price rises to our **profit target** (+0.95 %) first → it's a **win (1)**.\n"
            "- If the price drops to our **safety limit** (−0.5 %) first → it's a **loss (0)**.\n"
            "- If **nothing** big happens within the hour → we also count it as a **loss (0)**.\n\n"
            "This method (watch for a target vs a limit within a time window) is known as the "
            "**triple-barrier method** — three ways out: hit the top, hit the bottom, or run out of time."
        )

        st.markdown("#### The choices behind this, in plain words")
        st.markdown(
            "- **We aim higher than we risk** (+0.95 % target vs −0.5 % limit). Winners are rarer this way, "
            "but each winner is worth more than each loser.\n"
            "- **We only look one hour ahead** — we're after quick moves, matching the engine's 5-minute rhythm.\n"
            "- **'Nothing happened' counts as a loss** — if there's no clear move, we'd rather not have traded.\n"
            "- **We never peek into the future.** Every number for a given moment uses only past prices, and "
            "each crypto is handled separately. (If we accidentally used future information, the model would "
            "look brilliant in testing and fail in real life — a classic mistake called *data leakage*.)"
        )

        st.markdown("#### How we 'enrich' — in simple terms")
        st.markdown(
            "A raw price on its own says nothing (*is 58,500 € expensive or cheap? going up or down?*). So for "
            "each moment we compute 78 indicators that summarise the **recent past**: is the price above or "
            "below its recent average? is it rising or falling? is it unusually calm or volatile? has it "
            "dropped a lot lately? These turn 5 raw price numbers into 78 meaningful clues the model can learn "
            "from. **77 of them we calculate ourselves from the prices; only 1 comes from outside** (the "
            "market's *fear & greed* mood index)."
        )

    # ---------------- SNAPSHOT ----------------
    st.subheader("Dataset snapshot — where we really are")
    if data_ok:
        with st.container(border=True):
            a, b, c, d = st.columns(4)
            kpi_card(a, f"{len(test):,}", "Rows (test set)",
                     "Number of 5-minute moments kept aside to test the model on unseen data.")
            kpi_card(b, f"{test['symbol'].nunique()}", "Cryptos",
                     "Number of coins tracked (BTC, ETH, SOL, …).")
            kpi_card(c, f"{len(FEATURE_COLS)}", "Features",
                     "Number of indicators describing each moment.")
            kpi_card(d, f"{test['label'].mean()*100:.1f} %", "Blind baseline",
                     "If you bought at EVERY moment, this is how often you'd win. The bar the model must beat.")
            if "candle_time" in test.columns:
                st.caption(f"Test window {test['candle_time'].min()} → {test['candle_time'].max()} · "
                           "past used for training, this later period used only for testing.")

        st.markdown("#### Three different win rates — and why they are NOT the same thing")
        w1, w2, w3 = st.columns(3)
        kpi_card(w1, "~9 %", "Buying blindly",
                 "Win rate if you bought at every single moment, no selection. This is just how hard the "
                 "problem is — not anyone's real performance.")
        kpi_card(w2, f"{ENGINE['winrate_signal']} %", "The engine's signal",
                 "How often the engine's own signals were right (measured on imagined trades). This is the "
                 "level the AI is trying to improve on.")
        kpi_card(w3, "52–75 %", "The AI filter",
                 "How often the AI is right when it only keeps its most confident calls. Higher, but on a "
                 "slightly different definition — so it can't be compared one-to-one with the engine.")
        st.warning("These three numbers live in **different setups** (different profit targets, different ways "
                   "of counting a win). So we present them side by side, but we **never stack them** into a "
                   "single '9 % → 75 %' story. Being clear about this is the honest, rigorous conclusion.")
    else:
        st.info("Load the data in the sidebar to see the snapshot.")


def render_eda():
    st.header("Exploratory Data Analysis")
    st.markdown("Slicing the data every way to find **which situations produce winners**. Green = above break-even.")
    if not data_ok:
        st.info("Load the data in the sidebar to compute the charts.")
        return

    with st.container(border=True):
        st.subheader("1 · Target balance")
        vc = test["label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 1.9))
        ax.barh(["Loss (0)", "Win (1)"], [vc.get(0, 0), vc.get(1, 0)], color=["#D98880", "#3BA99C"])
        for i, v in enumerate([vc.get(0, 0), vc.get(1, 0)]):
            ax.text(v, i, f" {v:,} ({v/len(test)*100:.1f}%)", va="center")
        plt.tight_layout(); st.pyplot(fig)
        st.caption("**Shows:** class balance. **Says:** only ~10 % winners → imbalanced; never trust Accuracy alone.")

    with st.container(border=True):
        st.subheader("2 · Win rate across indicator zones (interactive)")
        cA, cB, cC = st.columns(3)
        if "rsi14" in test.columns:
            show_winrate(cA, winrate_by_zone(test, "rsi14", list(range(0, 101, 10))), xlabel="RSI", title="By RSI")
        if "stoch_k" in test.columns:
            show_winrate(cB, winrate_by_zone(test, "stoch_k", list(range(0, 101, 10))), xlabel="Stochastic", title="By Stochastic")
        if "bb_pos" in test.columns:
            show_winrate(cC, winrate_by_zone(test, "bb_pos", [x/10 for x in range(-2, 13)]), xlabel="Bollinger", title="By Bollinger")
        st.caption("**Shows:** win rate by indicator level. **Says:** wins concentrate at **low** RSI/Stochastic "
                   "and the **lower** Bollinger band → **mean reversion**.")

    with st.container(border=True):
        st.subheader("3 · Temporal & per-crypto patterns")
        cD, cE, cF = st.columns(3)
        if "hour_utc" in test.columns:
            show_winrate(cD, test.groupby("hour_utc")["label"].mean() * 100, xlabel="Hour UTC", title="By hour", rotate=0)
        if "dow" in test.columns:
            show_winrate(cE, test.groupby("dow")["label"].mean() * 100, xlabel="Day (0=Mon)", title="By weekday", rotate=0)
        show_winrate(cF, (test.groupby("symbol")["label"].mean() * 100).sort_values(ascending=False),
                     xlabel="Symbol", title="By crypto")
        st.caption("**Shows:** win rate by time and crypto. **Says:** time effect is weak; cryptos are "
                   "**heterogeneous** → stay cautious.")

    with st.container(border=True):
        st.subheader("4 · Which features explain the outcome?")
        corr = test[FEATURE_COLS].corrwith(test["label"]).dropna().sort_values()
        top = pd.concat([corr.head(7), corr.tail(7)])
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.barh(top.index, top.values, color=["#D98880" if v < 0 else "#3BA99C" for v in top.values])
        ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("correlation with 'win'")
        plt.tight_layout(); st.pyplot(fig)
        st.caption("**Shows:** feature/label correlation. **Says:** oversold-type features correlate most with "
                   "winning → confirms mean reversion, guides selection.")

    if "rsi14" in test.columns and "stoch_k" in test.columns:
        with st.container(border=True):
            st.subheader("5 · Crossing signals: RSI × Stochastic (interactive)")
            d = test[["rsi14", "stoch_k", "label"]].dropna().copy()
            d["r"] = pd.cut(d["rsi14"], range(0, 101, 20))
            d["s"] = pd.cut(d["stoch_k"], range(0, 101, 20))
            piv = d.pivot_table(index="r", columns="s", values="label", aggfunc="mean", observed=True) * 100
            piv.index = piv.index.astype(str)      # Interval -> text (avoids Plotly JSON error)
            piv.columns = piv.columns.astype(str)
            if HAS_PLOTLY:
                fig = px.imshow(piv, color_continuous_scale="RdYlGn", zmin=0, zmax=100, text_auto=".0f",
                                aspect="auto", labels=dict(x="Stochastic", y="RSI", color="win %"))
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(6.5, 3.4))
                im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
                ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([str(c) for c in piv.columns], rotation=45, ha="right")
                ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([str(i) for i in piv.index])
                plt.colorbar(im, ax=ax, label="win %"); plt.tight_layout(); st.pyplot(fig)
            st.caption("**Shows:** win rate per RSI×Stochastic cell. **Says:** best = **double oversold** "
                       "(low RSI + low Stochastic) → combining signals sharpens the edge.")

    with st.container(border=True):
        st.subheader("6 · The engine's signal — P&L asymmetry (from Supabase)")
        st.markdown(f"Win rate **{ENGINE['winrate_signal']} %**, but mean P&L **+{ENGINE['pnl_mean']} %** vs "
                    f"median **{ENGINE['pnl_median']} %**.")
        cG, cH = st.columns(2)
        show_winrate(cG, [m[0] for m in ENGINE["by_month"]], [m[2] for m in ENGINE["by_month"]],
                     xlabel="Month 2026", title="Engine WR by month", rotate=0)
        show_winrate(cH, [s[0] for s in ENGINE["by_symbol"]], [s[2] for s in ENGINE["by_symbol"]],
                     xlabel="Symbol", title="Engine WR by crypto", rotate=0)
        st.caption(
            "**Shows:** the engine signal's win rate over time and per crypto. **Says:** the win rate is "
            f"below 50 % ({ENGINE['winrate_signal']} %), but the signal still works because of the "
            "**reward/risk asymmetry** — winners are ~2× bigger than losers.")
        st.markdown(
            f"**The asymmetry, proven by the distribution of individual results:** the *median* result is "
            f"slightly **negative** (−0.01 %) yet the *average* is **positive** (+0.11 %). The best signal "
            f"reached **+{ENGINE['max_win']} %** while the worst was only **{ENGINE['max_loss']} %** — a few "
            "big winners on the right carry the whole performance (the top 1 % of signals exceed "
            f"+{ENGINE['p99']} %). That long right tail is the real story.")
        st.caption("Honesty note: these are **signals evaluated in shadow** (imagined trades), not executed "
                   f"trades. On the real (LIVE) account the ratio drops to ~{ENGINE['live_reward_risk']} "
                   f"(N={ENGINE['live']}) — the real world is a bit less favourable than the shadow test.")

    st.success(
        "**EDA conclusion** \u2014 (1) hard, imbalanced problem (~10 % winners); (2) exploitable signal = "
        "**mean reversion** (oversold), sharpened by combining indicators; (3) heterogeneous across cryptos, "
        "weakly time-dependent; (4) engine signal profitable but asymmetric. → Let a model **learn and "
        "combine** these signals instead of hand-written rules."
    )


def render_models():
    st.header("From simplest to most complex")
    st.markdown("Logistic Regression → Random Forest → XGBoost, then LSTM. All on the **out-of-sample test**.")
    with st.container(border=True):
        st.subheader("Model comparison (test set)")
        st.dataframe(METRICS.style.format("{:.3f}", na_rep="—"), use_container_width=True)
        st.markdown("- Naive = 90 % Accuracy but **F1 = 0** → Accuracy lies on imbalanced data; judge by F1/AUC.\n"
                    "- Tree models win; the **ensemble** is best (AUC 0.89).")
    st.warning("**Honesty:** cross-validation shows ~0.997 vs test ~0.87 — candle autocorrelation (no embargo "
               "between folds). We rely on the out-of-sample test.")
    with st.container(border=True):
        st.subheader("Adding temporality: the LSTM (sheet → cube)")
        st.markdown("Trees see **one row** (a photo); the LSTM sees a **window of 30 candles** (a movie).")
        c1, c2 = st.columns(2)
        c1.metric("Trees — flat (2D)", "(N, 78)")
        c2.metric("LSTM — sequences (3D)", "(N, 30, 78)")
        st.caption("`create_sequences` re-slices the flat table into sliding windows. The LSTM (0.854) does "
                   "**not** beat the trees — features already encode temporality. Honest, mature finding.")


def render_results():
    st.header("Three win rates — three different frames")
    with st.container(border=True):
        st.markdown("The key honesty slide: these numbers **do not stack**.")
        st.table(pd.DataFrame({
            "Frame": ["Blind (every candle)", "Engine signal (shadow)", "ML filter (high conf.)"],
            "Win rate": ["~9 %", f"{ENGINE['winrate_signal']} %", "52–75 %"],
            "Target": ["0.95 / 0.5", "0.8 / 0.4", "0.95 / 0.5"],
            "Measure": ["TP before SL", "P&L > 0", "TP before SL"],
        }))
        st.caption("Different targets & definitions → not directly comparable.")

    st.subheader("Live prediction")
    with st.container(border=True):
        if data_ok and xgb_model is not None:
            seuil = 0.94
            try:
                mlcfg = load_json(os.path.join(models_dir, "ml_config.json"))
                seuil = float(mlcfg.get("threshold_xgb", mlcfg.get("threshold", 0.94)))
            except Exception:
                pass
            symbols = sorted(test["symbol"].unique().tolist())
            cA, cB = st.columns([1, 1])
            sym = cA.selectbox("Crypto", symbols)
            sub = test[test["symbol"] == sym].reset_index(drop=True)
            if cB.button("Random candle"):
                st.session_state["idx"] = int(np.random.randint(0, len(sub)))
            idx = st.slider("Position in history", 0, len(sub) - 1, st.session_state.get("idx", len(sub) // 2))
            row = sub.iloc[[idx]]
            proba = float(xgb_model.predict_proba(row[FEATURE_COLS])[:, 1][0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Win probability", f"{proba*100:.1f} %")
            m2.metric(f"Decision (thr {seuil:.2f})", "BUY" if proba >= seuil else "skip")
            m3.metric("Actual outcome", "WIN" if int(row["label"].iloc[0]) == 1 else "loss")
            if "candle_time" in row.columns:
                st.caption(f"{sym} · {row['candle_time'].iloc[0]}")
            show = [c for c in ["rsi14", "stoch_k", "bb_pos", "atr_pct", "ret_12", "fear_greed_value"] if c in row.columns]
            st.dataframe(row[show], use_container_width=True)
        else:
            st.info("Load data AND model (sidebar) to enable live prediction.")

    st.subheader("Deployment & next steps")
    with st.container(border=True):
        st.markdown(
            "- **Already live** on **Railway** (FastAPI): every 5 min the engine asks, gets a probability, "
            "buys only above threshold; fallback if down.\n"
            "- **Next:** align targets (0.7/0.7 vs 0.95/0.5); model slippage; embargo inside CV; repair "
            "shadow eval; rotate exposed keys.\n"
            "- **Goal:** a data-driven filter that durably improves the engine's win rate — validated first "
            "in paper trading."
        )


# ============================================================================
#  SIMPLIFIED VERSION (Demoday — clear & pedagogical, slide by slide)
# ============================================================================
def big(text, size="2.4rem", color="#FAFAFA", weight=800, opacity=1.0):
    st.markdown(f"<div style='text-align:center;font-size:{size};font-weight:{weight};"
                f"color:{color};opacity:{opacity};line-height:1.3;margin:0.35rem 0'>{text}</div>",
                unsafe_allow_html=True)


def head(icon, text):
    st.markdown(f"<div style='text-align:center;font-size:1.9rem;font-weight:800;color:#3BA99C;"
                f"margin-bottom:.5rem'><i class='bi {icon}'></i>&nbsp; {text}</div>", unsafe_allow_html=True)


def stat(col, num, label, desc="", color="#3BA99C"):
    col.markdown(
        f"<div style='text-align:center;padding:6px 4px'>"
        f"<div style='font-size:2.6rem;font-weight:800;color:{color};line-height:1'>{num}</div>"
        f"<div style='font-size:1.25rem;font-weight:700;margin-top:4px'>{label}</div>"
        f"<div style='font-size:1.05rem;opacity:.72;margin-top:2px;line-height:1.4'>{desc}</div></div>",
        unsafe_allow_html=True)


def glossary(items):
    rows = "".join(f"<div style='margin:3px 0'><b style='color:#3BA99C'>{t}</b> \u2014 {d}</div>" for t, d in items)
    st.markdown(
        "<div style='background:rgba(90,100,114,.12);border:1px solid #3A4150;border-radius:10px;"
        "padding:10px 16px;font-size:.95rem;margin-top:.8rem'>"
        "<div style='font-size:.72rem;letter-spacing:.08em;opacity:.55;margin-bottom:4px'>GLOSSARY</div>"
        f"{rows}</div>", unsafe_allow_html=True)


ENGINE_ONLY_SVG = """
<div style="display:flex;justify-content:center">
<svg viewBox="0 0 700 180" style="width:100%;max-width:700px">
  <defs><marker id="a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#8A94A6" stroke-width="1.6" stroke-linecap="round"/></marker></defs>
  <rect x="6" y="10" width="180" height="160" rx="12" fill="#1A1F2B" stroke="#5A6472"/>
  <text x="96" y="34" text-anchor="middle" fill="#E6E6E6" font-size="14" font-weight="700">SIGNALS IT WATCHES</text>
  <text x="96" y="66" text-anchor="middle" fill="#C9CDD6" font-size="12.5">Technical indicators</text>
  <text x="96" y="92" text-anchor="middle" fill="#C9CDD6" font-size="12.5">Whale movements</text>
  <text x="96" y="118" text-anchor="middle" fill="#C9CDD6" font-size="12.5">Crypto news</text>
  <text x="96" y="144" text-anchor="middle" fill="#C9CDD6" font-size="12.5">Fear &amp; greed</text>
  <line x1="186" y1="90" x2="236" y2="90" stroke="#8A94A6" stroke-width="2" marker-end="url(#a1)"/>
  <rect x="236" y="50" width="170" height="80" rx="12" fill="#1A1F2B" stroke="#5A6472"/>
  <text x="321" y="84" text-anchor="middle" fill="#E6E6E6" font-size="14" font-weight="700">THE ENGINE</text>
  <text x="321" y="107" text-anchor="middle" fill="#C9CDD6" font-size="12">combines them into a score</text>
  <line x1="406" y1="90" x2="456" y2="90" stroke="#8A94A6" stroke-width="2" marker-end="url(#a1)"/>
  <rect x="456" y="50" width="180" height="80" rx="12" fill="#1A1F2B" stroke="#5A6472"/>
  <text x="546" y="84" text-anchor="middle" fill="#E6E6E6" font-size="14" font-weight="700">DECISION</text>
  <text x="546" y="107" text-anchor="middle" fill="#C9CDD6" font-size="12">buy / hold / sell</text>
</svg></div>"""

ENGINE_AI_SVG = """
<div style="display:flex;justify-content:center">
<svg viewBox="0 0 940 300" style="width:100%;max-width:940px">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#8A94A6" stroke-width="1.6" stroke-linecap="round"/></marker>
    <marker id="aht" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#3BA99C" stroke-width="1.8" stroke-linecap="round"/></marker>
  </defs>
  <rect x="10" y="70" width="190" height="180" rx="12" fill="#1A1F2B" stroke="#5A6472"/>
  <text x="105" y="95" text-anchor="middle" fill="#E6E6E6" font-size="15" font-weight="700">SIGNALS IT WATCHES</text>
  <text x="105" y="128" text-anchor="middle" fill="#C9CDD6" font-size="13">Technical indicators</text>
  <text x="105" y="156" text-anchor="middle" fill="#C9CDD6" font-size="13">Whale movements</text>
  <text x="105" y="184" text-anchor="middle" fill="#C9CDD6" font-size="13">Crypto news</text>
  <text x="105" y="212" text-anchor="middle" fill="#C9CDD6" font-size="13">Fear &amp; greed</text>
  <line x1="200" y1="160" x2="250" y2="160" stroke="#8A94A6" stroke-width="2" marker-end="url(#ah)"/>
  <rect x="252" y="118" width="180" height="86" rx="12" fill="#1A1F2B" stroke="#5A6472"/>
  <text x="342" y="150" text-anchor="middle" fill="#E6E6E6" font-size="15" font-weight="700">THE ENGINE</text>
  <text x="342" y="175" text-anchor="middle" fill="#C9CDD6" font-size="12.5">turns them into a score</text>
  <line x1="432" y1="160" x2="560" y2="160" stroke="#8A94A6" stroke-width="2" marker-end="url(#ah)"/>
  <rect x="430" y="14" width="300" height="78" rx="12" fill="rgba(59,169,156,.16)" stroke="#3BA99C" stroke-width="2"/>
  <text x="580" y="42" text-anchor="middle" fill="#3BA99C" font-size="15" font-weight="800">THE AI LAYER \u2014 my project</text>
  <text x="580" y="66" text-anchor="middle" fill="#DDF3EF" font-size="12.5">"is this really a good moment?" \u2192 blocks weak buys</text>
  <line x1="524" y1="92" x2="524" y2="150" stroke="#3BA99C" stroke-width="2.2" marker-end="url(#aht)"/>
  <rect x="562" y="118" width="168" height="86" rx="12" fill="#1A1F2B" stroke="#5A6472"/>
  <text x="646" y="150" text-anchor="middle" fill="#E6E6E6" font-size="15" font-weight="700">DECISION</text>
  <text x="646" y="175" text-anchor="middle" fill="#C9CDD6" font-size="12.5">buy / hold / sell</text>
  <line x1="730" y1="160" x2="770" y2="160" stroke="#8A94A6" stroke-width="2" marker-end="url(#ah)"/>
  <rect x="772" y="122" width="158" height="78" rx="12" fill="rgba(59,169,156,.16)" stroke="#3BA99C"/>
  <text x="851" y="152" text-anchor="middle" fill="#3BA99C" font-size="15" font-weight="800">Better</text>
  <text x="851" y="176" text-anchor="middle" fill="#3BA99C" font-size="15" font-weight="800">win rate</text>
</svg></div>"""


def s_hook():
    head("bi-robot", "A robot that trades crypto on its own \u2014 24/7")
    big("Running <b>since June 2025</b> \u2014 about <b>13 months</b> of non-stop data.",
        size="1.6rem", weight=650, opacity=0.9)
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    stat(a, "686k", "Decisions recorded", "Every buy / hold / sell it ever made.")
    stat(b, "750k", "Price snapshots", "One every 5 minutes, over ~13 months.")
    stat(c, "19", "Cryptos tracked", "Bitcoin, Ethereum, Solana, and more.")
    d, e = st.columns(2)
    stat(d, "4,233", "Trades measured", "Signals we could check the result of.")
    stat(e, "0.8% / 0.4%", "Target / safety limit", "Sell at +0.8% gain, or cut at \u22120.4% loss.")


def s_engine():
    head("bi-gear-wide-connected", "How it decides today (no AI yet)")
    st.markdown(ENGINE_ONLY_SVG, unsafe_allow_html=True)
    big("It listens to signals, turns them into a score, and buys, holds or sells.",
        size="1.5rem", weight=600, opacity=0.9)
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    stat(a, "~10%", "Buying at random", "Only ~1 moment in 10 would be a winner.", color="#D98880")
    stat(b, "45.7%", "With the robot's signals", "It wins less than half the <b>times</b> \u2014 "
         "but this only counts how OFTEN it wins, not how MUCH.")
    big("And when it wins, it wins about <b>2\u00d7 more</b> than it loses \u2014 so overall it "
        "stays profitable.", size="1.45rem", weight=650, opacity=0.9)
    glossary([("Signal", "a clue about the market (e.g. price rising, big money moving)"),
              ("Win rate", "the share of trades that end in profit")])


def s_idea():
    head("bi-lightbulb", "The idea")
    st.markdown(ENGINE_AI_SVG, unsafe_allow_html=True)
    big("Today the engine buys from its signals. My idea: add an <span style='color:#3BA99C'>AI layer</span> "
        "that confirms or blocks each buy.", size="1.3rem", weight=650)
    big("Goal: lift the <b>win rate</b> (how often we win) on buying \u2014 while the classic rules keep "
        "protecting the <b>amounts</b> (how much we win or lose) on selling.", size="1.2rem",
        weight=600, opacity=0.85)
    glossary([("TP \u2014 Take-Profit", "auto-sell once a trade is +0.8% in profit"),
              ("SL \u2014 Stop-Loss", "auto-sell once a trade is \u22120.4% down, to cap the loss"),
              ("TSL \u2014 Trailing Stop-Loss", "a moving safety net that rises with the price to lock in gains"),
              ("AI does the BUYING side", "selling stays 100% handled by these classic rules \u2014 no AI")])


def s_data():
    head("bi-database", "The data we feed the AI")
    big("For every 5-minute moment, we turn 5 raw prices into "
        "<span style='color:#3BA99C'>78 clues</span> about the market\u2026", size="1.6rem", weight=700)
    big("\u2026then we mark each moment: would buying here have <b>won</b> or <b>lost</b>?",
        size="1.6rem", weight=700)
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    stat(a, "78", "Clues per moment", "Trend, momentum, volatility, and more.")
    stat(b, "500k+", "Labelled examples", "Past moments, each marked win or loss.")
    stat(c, "1 or 0", "The answer to learn", "1 = would have won \u00b7 0 = would have lost.")
    glossary([("Clue (a 'feature')", "a number describing the market, computed from past prices"),
              ("Label", "the right answer we want the AI to predict: win (1) or loss (0)")])


# ---- EDA gallery charts (regenerated live from the TRAIN data) --------------
#   EDA is done on the TRAINING set (never on the test set) — this matches the
#   original analysis and is the methodologically correct choice.
def _base():
    return train_df["label"].mean() * 100


def _zone_bar(col, bins, xlabel, title, note):
    d = train_df[[col, "label"]].dropna().copy()
    d["_z"] = pd.cut(d[col], bins=bins)
    wr = d.groupby("_z", observed=True)["label"].mean() * 100
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    colors = ["#3BA99C" if v >= BREAKEVEN else "#E8825A" for v in wr.values]
    ax.bar([str(i) for i in wr.index], wr.values, color=colors)
    ax.axhline(BREAKEVEN, color="#222", ls="--", lw=1.2, label="Break-even (41%)")
    ax.axhline(_base(), color="grey", ls=":", lw=1, label="Blind baseline")
    for i, v in enumerate(wr.values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("Win rate (%)"); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    return fig, note


def eda_stoch():
    return _zone_bar("stoch_k", list(range(0, 101, 10)), "Stochastic zone (left = oversold)",
                     "Win rate by Stochastic level",
                     "When the Stochastic is very low (deeply oversold), the win rate reaches ~63%. "
                     "As soon as it rises, it collapses. The strongest single signal.")


def eda_rsi():
    d = train_df[["rsi14", "label"]].dropna().copy()
    d["_z"] = pd.cut(d["rsi14"], list(range(0, 101, 10)))
    wr = d.groupby("_z", observed=True)["label"].mean() * 100
    share = d.groupby("_z", observed=True).size() / len(d) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.3))
    colors = ["#3BA99C" if v >= BREAKEVEN else "#E8825A" for v in wr.values]
    ax1.bar([str(i) for i in wr.index], wr.values, color=colors)
    ax1.axhline(BREAKEVEN, color="#222", ls="--", lw=1.2)
    for i, v in enumerate(wr.values):
        ax1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax1.set_title("Win rate by RSI level"); ax1.set_ylabel("Win rate (%)")
    ax1.tick_params(axis="x", rotation=45)
    ax2.bar([str(i) for i in share.index], share.values, color="#7FA8C9")
    ax2.set_title("% of all moments in each RSI zone"); ax2.set_ylabel("% of dataset")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, ("Same story with RSI: ~45% win rate when oversold, near 0% when overbought. "
                 "The right panel shows most moments sit in the middle zones \u2014 so winning setups are rare.")


def eda_bb():
    bins = [-np.inf, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, np.inf]
    labels = ["<0", "0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5-.6", ".6-.7", ".7-.8", ".8-.9", ".9-1", ">1"]
    d = train_df[["bb_pos", "label"]].dropna().copy()
    d["_z"] = pd.cut(d["bb_pos"], bins=bins, labels=labels)
    wr = d.groupby("_z", observed=True)["label"].mean() * 100
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    colors = ["#3BA99C" if v >= BREAKEVEN else "#E8825A" for v in wr.values]
    ax.bar([str(i) for i in wr.index], wr.values, color=colors)
    ax.axhline(BREAKEVEN, color="#222", ls="--", lw=1.2)
    for i, v in enumerate(wr.values):
        ax.text(i, v + .5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("Win rate (%)"); ax.set_xlabel("Position in Bollinger Bands (0 = low, 1 = high)")
    ax.set_title("Win rate by Bollinger position")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    return fig, ("Glued to the LOWER band (price abnormally low) \u2192 ~24% win rate; near the top \u2192 "
                 "almost never. A third angle confirming: we win after drops.")


def eda_combos():
    t = train_df
    conds = [
        ("Stoch < 10", t["stoch_k"] < 10),
        ("RSI<30 & Stoch<20", (t["rsi14"] < 30) & (t["stoch_k"] < 20)),
        ("RSI<35 & Stoch<20", (t["rsi14"] < 35) & (t["stoch_k"] < 20)),
        ("RSI<30 & Stoch<30", (t["rsi14"] < 30) & (t["stoch_k"] < 30)),
        ("RSI<40 & Stoch<20", (t["rsi14"] < 40) & (t["stoch_k"] < 20)),
        ("Stoch < 15", t["stoch_k"] < 15),
        ("RSI<40 & Stoch<30", (t["rsi14"] < 40) & (t["stoch_k"] < 30)),
        ("RSI < 20", t["rsi14"] < 20),
        ("RSI < 25", t["rsi14"] < 25),
    ]
    if "bb_pos" in t.columns:
        conds.append(("RSI<30 & bb_pos<0.2", (t["rsi14"] < 30) & (t["bb_pos"] < 0.2)))
    rows = []
    for lab, m in conds:
        n = int(m.sum())
        if n < 50:
            continue
        wr = t.loc[m, "label"].mean() * 100
        ev = wr / 100 * 0.95 - (1 - wr / 100) * 0.5
        rows.append((lab, wr, n, ev))
    rows.sort(key=lambda r: r[1])
    labs = [f"{r[0]} (n={r[2]:,})" for r in rows]
    wrs = [r[1] for r in rows]
    evs = [r[3] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.barh(labs, wrs, color=["#3BA99C" if v >= BREAKEVEN else "#E8825A" for v in wrs])
    ax1.axvline(BREAKEVEN, color="#222", ls="--", lw=1.2)
    ax1.set_xlabel("Win rate (%)"); ax1.set_title("Win rate of rule combinations")
    ax2.barh(labs, evs, color=["#3BA99C" if v >= 0 else "#E8825A" for v in evs])
    ax2.axvline(0, color="#222", lw=1)
    ax2.set_xlabel("Est. gain per trade (%)"); ax2.set_title("Expected value per trade (estimated)")
    ax2.set_yticklabels([])
    plt.tight_layout()
    return fig, ("Combining conditions sharpens the edge: RSI<30 AND Stoch<20 reaches ~65%. The right panel "
                 "estimates the average gain per trade \u2014 the more selective the rule, the more it pays. "
                 "(Estimated from the +0.95% / -0.5% target.)")


def eda_stability():
    d = train_df[["candle_time", "rsi14", "stoch_k", "label"]].dropna().copy()
    d = d[(d["rsi14"] < 35) & (d["stoch_k"] < 20)]
    d["month"] = pd.to_datetime(d["candle_time"], utc=True, errors="coerce").dt.to_period("M").astype(str)
    g = d.groupby("month")["label"].agg(["mean", "size"])
    g["mean"] *= 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(g.index, g["mean"], marker="o", color="#3BA99C")
    ax1.fill_between(range(len(g)), BREAKEVEN, g["mean"], where=(g["mean"] >= BREAKEVEN),
                     color="#3BA99C", alpha=.18)
    ax1.axhline(BREAKEVEN, color="#222", ls="--", lw=1.2, label="Break-even")
    ax1.set_ylabel("Win rate (%)"); ax1.set_title("Filter 'RSI<35 & Stoch<20' \u2014 win rate over time (training period)")
    ax1.legend(fontsize=8); ax1.set_ylim(0, 100)
    ax2.bar(g.index, g["size"], color="#7FA8C9")
    ax2.set_ylabel("Trades"); ax2.set_xlabel("Month")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, ("The signal is not a fluke: month after month, the same rule stays above break-even. "
                 "It holds over the whole training period.")


def s_eda():
    head("bi-search", "What we found in the data")
    big("Before any AI, we explored the data by hand. One clear pattern:", size="1.45rem", weight=650)
    big("we win most often <span style='color:#3BA99C'>right after the price has dropped</span> "
        "(when the market is 'oversold').", size="1.7rem", weight=750)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    glos = [("Oversold", "the price has fallen sharply and tends to bounce back up"),
            ("RSI", "a 0\u2013100 gauge: low = oversold, high = overbought"),
            ("Stochastic", "a 0\u2013100 gauge of where the price sits in its recent range; low = near its recent low"),
            ("Bollinger position", "where the price sits between a low band (0) and a high band (1)"),
            ("Expected value", "the average gain (or loss) you'd expect per trade")]
    if not (data_ok and train_df is not None and "stoch_k" in train_df.columns):
        st.info("Load the data in the sidebar to see the EDA charts.")
        glossary(glos)
        return

    GALLERY = [("Stochastic", eda_stoch), ("RSI", eda_rsi), ("Bollinger", eda_bb),
               ("Combinations", eda_combos), ("Stability over time", eda_stability)]
    n = len(GALLERY)
    if "eda_idx" not in st.session_state:
        st.session_state.eda_idx = 0
    st.session_state.eda_idx %= n
    i = st.session_state.eda_idx
    c1, c2, c3 = st.columns([1, 3, 1])
    if c1.button("\u2039 Prev chart", key="eda_prev", use_container_width=True):
        st.session_state.eda_idx = (i - 1) % n; st.rerun()
    c2.markdown(f"<div style='text-align:center;font-size:1.2rem;font-weight:700;padding-top:6px'>"
                f"{GALLERY[i][0]} <span style='opacity:.5'>({i+1}/{n})</span></div>", unsafe_allow_html=True)
    if c3.button("Next chart \u203a", key="eda_next", use_container_width=True):
        st.session_state.eda_idx = (i + 1) % n; st.rerun()
    fig, note = GALLERY[i][1]()
    st.pyplot(fig)
    st.markdown(f"<div style='text-align:center;font-size:1.1rem;margin-top:.2rem'>{note}</div>",
                unsafe_allow_html=True)
    st.caption("Blue bars = win rate above break-even (profitable zone) \u00b7 orange = below.")
    glossary(glos)


# ---- MODELS gallery charts -------------------------------------------------
def mdl_compare():
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    names = ["Logistic\nRegression", "Random\nForest", "XGBoost", "LSTM"]
    vals = [0.705, 0.884, 0.865, 0.854]
    ax.bar(names, vals, color=["#8A94A6", "#3BA99C", "#3BA99C", "#7FB8D6"])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("quality score (0.5 = random, 1 = perfect)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_title("How good is each model at telling winners from losers? (test set)")
    plt.tight_layout()
    return fig, ("From the simplest (Logistic Regression) to the most advanced (LSTM). The tree models win. "
                 "Careful: a lazy model that ALWAYS says 'loss' scores 90% accuracy yet is useless \u2014 so we "
                 "judge on this score and the F1, never on accuracy.")


def mdl_confusion():
    mats = {"Logistic Regression": [[132276, 58908], [4611, 17468]],
            "Random Forest":        [[173723, 17461], [7482, 14597]],
            "XGBoost":              [[168528, 22656], [6533, 15546]]}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (name, m) in zip(axes, mats.items()):
        m = np.array(m)
        ax.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > m.max() / 2 else "#222", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["LOSS", "WIN"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["LOSS", "WIN"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    plt.tight_layout()
    return fig, ("At the standard 0.5 threshold: Logistic Regression shouts 'WIN' far too often (58,908 false "
                 "alarms). The tree models are much more precise \u2014 far fewer false alarms, cleaner calls.")


def mdl_roc():
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    px, pr, y = compute_test_probas(processed_dir, models_dir)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.7))
    for p, name, col in [(px, "XGBoost", "#4C78A8"), (pr, "Random Forest", "#E8825A")]:
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=col, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    a1.plot([0, 1], [0, 1], "--", color="grey")
    a1.set_xlabel("False-alarm rate"); a1.set_ylabel("True-win rate")
    a1.set_title("ROC \u2014 test set"); a1.legend(fontsize=8)
    for p, name, col in [(px, "XGBoost", "#4C78A8"), (pr, "Random Forest", "#E8825A")]:
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=col, label=name)
    a2.axhline(0.414, ls="--", color="black", label="Break-even")
    a2.set_xlabel("Recall (share of winners caught)"); a2.set_ylabel("Precision (= win rate)")
    a2.set_title("Precision-Recall \u2014 test set"); a2.legend(fontsize=8)
    plt.tight_layout()
    return fig, ("The models clearly separate winners from losers (AUC 0.87\u20130.89). The right panel is the "
                 "key one: if we act only on the most confident calls (left side), the precision \u2014 which IS "
                 "the win rate \u2014 stays high (80%+).")


def mdl_threshold():
    px, pr, y = compute_test_probas(processed_dir, models_dir)
    which = st.radio("Model", ["XGBoost", "Random Forest"], horizontal=True, key="mdl_thr")
    proba = px if which == "XGBoost" else pr
    ts, wr, nn, ev = [], [], [], []
    for t in np.arange(0.05, 0.98, 0.01):
        m = proba >= t
        c = int(m.sum())
        if c < 30:
            break
        w = float(y[m].mean())
        ts.append(t); wr.append(w * 100); nn.append(c); ev.append(w * 0.85 - (1 - w) * 0.6)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    a1.bar(ts, wr, width=0.01, color=["#4C78A8" if v >= BREAKEVEN else "#E8825A" for v in wr])
    a1.axhline(BREAKEVEN, ls="--", color="black", label="Break-even (41%)")
    a1.axvline(0.94, ls="--", color="green", lw=2, label="Our threshold (0.94)")
    a1.set_ylabel("Win rate (%)"); a1.legend(fontsize=8)
    a2.bar(ts, nn, width=0.01, color="#9CC2E0"); a2.axvline(0.94, ls="--", color="green", lw=2)
    a2.set_ylabel("# trades")
    a3.bar(ts, ev, width=0.01, color=["#4C78A8" if e >= 0 else "#E8825A" for e in ev])
    a3.axhline(0, color="black"); a3.axvline(0.94, ls="--", color="green", lw=2)
    a3.set_ylabel("Gain per trade (%)"); a3.set_xlabel("Confidence threshold required")
    a1.set_title(f"The confidence filter on the test set \u2014 {which}")
    plt.tight_layout()
    return fig, ("This is the heart of the method. The more confidence we require (x-axis), the higher the win "
                 "rate (top) and the more positive the average gain per trade (bottom) \u2014 at the cost of "
                 "fewer trades (middle). At our threshold (green line, 0.94): win rate ~52% for XGBoost, "
                 "~79% for Random Forest.")


def s_models():
    head("bi-cpu", "The models we tested")
    big("From the simplest to the most advanced \u2014 and how we judge them honestly.",
        size="1.45rem", weight=650)
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    glos = [("Logistic Regression / Random Forest / XGBoost", "models, simple to advanced, that learn rules from examples"),
            ("LSTM", "a neural network that reads a sequence of moments, like a short film"),
            ("Accuracy", "% of correct answers \u2014 misleading when winners are rare"),
            ("F1-score", "an honest score that rewards actually finding the rare winners"),
            ("Confusion matrix", "a table of correct vs wrong predictions, both ways"),
            ("ROC / AUC", "how well a model separates winners from losers (0.5 = random, 1 = perfect)"),
            ("Precision", "among the trades it calls 'win', how many really win (= win rate)"),
            ("Confidence threshold", "how sure the model must be before it says 'buy' (higher = fewer, safer trades)")]

    GALLERY = [("Model comparison", mdl_compare), ("How they get it wrong", mdl_confusion),
               ("How well they separate", mdl_roc), ("The confidence filter", mdl_threshold)]
    n = len(GALLERY)
    if "mdl_idx" not in st.session_state:
        st.session_state.mdl_idx = 0
    st.session_state.mdl_idx %= n
    i = st.session_state.mdl_idx
    c1, c2, c3 = st.columns([1, 3, 1])
    if c1.button("\u2039 Prev chart", key="mdl_prev", use_container_width=True):
        st.session_state.mdl_idx = (i - 1) % n; st.rerun()
    c2.markdown(f"<div style='text-align:center;font-size:1.2rem;font-weight:700;padding-top:6px'>"
                f"{GALLERY[i][0]} <span style='opacity:.5'>({i+1}/{n})</span></div>", unsafe_allow_html=True)
    if c3.button("Next chart \u203a", key="mdl_next", use_container_width=True):
        st.session_state.mdl_idx = (i + 1) % n; st.rerun()

    try:
        fig, note = GALLERY[i][1]()
        st.pyplot(fig)
        st.markdown(f"<div style='text-align:center;font-size:1.1rem;margin-top:.2rem'>{note}</div>",
                    unsafe_allow_html=True)
    except Exception as e:
        st.info("This chart needs the data AND both models (XGBoost + Random Forest) loaded. "
                "Check the /models folder in the sidebar.")
        st.caption(f"Detail: {e}")
    glossary(glos)


def s_demo():
    head("bi-play-circle", "See it in action")
    big("Pick a moment \u2192 the AI gives its confidence \u2192 see what really happened.",
        size="1.3rem", weight=600, opacity=0.85)
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    if data_ok and xgb_model is not None:
        seuil = 0.94
        try:
            mlcfg = load_json(os.path.join(models_dir, "ml_config.json"))
            seuil = float(mlcfg.get("threshold_xgb", mlcfg.get("threshold", 0.94)))
        except Exception:
            pass
        symbols = sorted(test["symbol"].unique().tolist())
        cc1, cc2 = st.columns([2, 1])
        sym = cc1.selectbox("Which crypto?", symbols)
        sub = test[test["symbol"] == sym].reset_index(drop=True)
        if cc2.button("Pick a random moment", use_container_width=True):
            st.session_state["idx"] = int(np.random.randint(0, len(sub)))
        idx = st.session_state.get("idx", len(sub) // 2)
        row = sub.iloc[[idx]]
        proba = float(xgb_model.predict_proba(row[FEATURE_COLS])[:, 1][0])
        buy = proba >= seuil
        won = int(row["label"].iloc[0]) == 1
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        stat(m1, f"{proba*100:.0f}%", "AI confidence", "How sure it is this is a winner.")
        stat(m2, "BUY" if buy else "skip", "The AI's call", "It only buys when very confident.",
             color="#3BA99C" if buy else "#C9CDD6")
        stat(m3, "WIN" if won else "loss", "What really happened", "The real outcome of that moment.",
             color="#3BA99C" if won else "#D98880")
    else:
        st.info("Load the data and the model in the sidebar to run the live demo.")


def s_results():
    head("bi-graph-up-arrow", "Does it work?")
    big("On <b>past data it had never seen</b>, keeping only the AI's most confident calls, "
        "the win rate climbs:", size="1.3rem", weight=650)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    stat(a, "~10%", "Buying at random", "The starting point.", color="#D98880")
    stat(b, "45.7%", "The engine's signals", "Rules only, no AI.")
    stat(c, "52\u201375%", "With the AI filter", "From a low confidence bar (52%) to a high one (75%).")
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    big("<b>How the confidence bar works:</b> the AI scores each moment from 0 to 100%. We only buy "
        "above a bar we choose (e.g. 94%). A higher bar = fewer trades, but a higher win rate.",
        size="1.15rem", weight=600, opacity=0.9)
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    big("<b>How we know:</b> we trained the AI on the past, then tested it on later months it had never "
        "seen \u2014 with safeguards so no future information could leak in.", size="1.12rem",
        weight=600, opacity=0.88)
    st.markdown(
        "<div style='text-align:center;background:rgba(217,136,128,.12);border:1px solid #D98880;"
        "border-radius:10px;padding:10px 14px;font-size:1.05rem'>"
        "Honest caveat: this is measured on <b>past data</b>, not real trading yet. It is <b>promising, "
        "not proven</b> \u2014 and the three numbers are measured slightly differently, so read them as a "
        "direction, not one single ladder.</div>", unsafe_allow_html=True)
    glossary([("Train / test", "learn on the past, then check on unseen later data \u2014 like real life"),
              ("Leakage", "accidentally using future info; we ran a check to rule it out"),
              ("Confidence threshold", "how sure the AI must be before it says 'buy' (higher = fewer, safer trades)")])


def s_earn():
    head("bi-cash-coin", "What could it earn?")
    big("A small edge per trade, repeated thousands of times, adds up \u2014 but on-chain, "
        "<b>trading costs decide everything</b>.", size="1.35rem", weight=650)
    if not (data_ok and xgb_model is not None):
        st.info("Load the data and models in the sidebar to run the simulator.")
        return
    try:
        px, pr, y = compute_test_probas(processed_dir, models_dir)
    except Exception as e:
        st.info("This simulator needs both models (XGBoost + Random Forest) loaded.")
        st.caption(f"Detail: {e}")
        return

    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Confidence bar required", 0.50, 0.97, 0.94, 0.01, key="earn_thr")
    size = c2.select_slider("Money per trade (\u20ac)", options=[100, 250, 500, 1000], value=500, key="earn_size")
    cost = c3.slider("Cost per trade %  (fees + slippage)", 0.0, 0.40, 0.18, 0.01, key="earn_cost")
    cost_frac = cost / 100.0
    has_ret = "exit_ret" in test.columns
    ret_all = test["exit_ret"].values if has_ret else None

    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for col, name, proba in [(cols[0], "XGBoost", px), (cols[1], "Random Forest", pr)]:
        m = proba >= thr
        n = int(m.sum())
        if n == 0:
            col.markdown(f"<div style='text-align:center;font-size:1.4rem;font-weight:800;"
                         f"color:#3BA99C;margin-bottom:8px'>{name}</div>", unsafe_allow_html=True)
            col.warning("No trades at this bar.")
            continue
        wr = float(y[m].mean()) * 100
        if has_ret:
            net = ret_all[m] - cost_frac
            avg = float(np.nanmean(net)) * 100
            pnl = float(np.nansum(net)) * size
        else:
            avg = (wr / 100 * 0.85 - (1 - wr / 100) * 0.6) - cost
            pnl = avg / 100 * size * n
        pnl_color = "#3BA99C" if pnl >= 0 else "#D98880"
        avg_color = "#3BA99C" if avg >= 0 else "#D98880"
        col.markdown(
            f"<div style='border:1px solid rgba(59,169,156,.35);border-radius:16px;"
            f"padding:22px 18px 20px;background:rgba(59,169,156,.06)'>"
            f"<div style='text-align:center;font-size:1.35rem;font-weight:800;color:#3BA99C;"
            f"margin-bottom:14px'>{name}</div>"
            f"<div style='text-align:center;font-size:2.9rem;font-weight:800;color:{pnl_color};"
            f"line-height:1'>{pnl:+,.0f} \u20ac</div>"
            f"<div style='text-align:center;font-size:1.0rem;opacity:.6;margin-top:8px'>"
            f"estimated P&amp;L (after costs)</div>"
            f"<div style='display:flex;justify-content:space-around;text-align:center;"
            f"margin-top:22px;padding-top:16px;border-top:1px solid rgba(59,169,156,.20)'>"
            f"<div><div style='font-size:1.5rem;font-weight:700'>{wr:.0f}%</div>"
            f"<div style='font-size:.9rem;opacity:.65'>win rate</div></div>"
            f"<div><div style='font-size:1.5rem;font-weight:700'>{n:,}</div>"
            f"<div style='font-size:.9rem;opacity:.65'>trades</div></div>"
            f"<div><div style='font-size:1.5rem;font-weight:700;color:{avg_color}'>{avg:+.2f}%</div>"
            f"<div style='font-size:.9rem;opacity:.65'>net / trade</div></div>"
            f"</div></div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    big("<b>Costs matter a lot on-chain.</b> Each round-trip costs ~0.18% (fees + slippage), "
        "which we subtract from every trade. A thin per-trade margin means keeping costs low is essential.",
        size="1.15rem", weight=600, opacity=0.92)
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    big("<b>Precision vs volume:</b> Random Forest wins more often (higher win rate, thicker margin per "
        "trade \u2014 more robust to costs); XGBoost trades far more, so it can win on total P&L when costs "
        "are low. The best model depends on the goal.", size="1.12rem", weight=600, opacity=0.88)
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;background:rgba(217,136,128,.12);border:1px solid #D98880;"
        "border-radius:10px;padding:14px 18px;font-size:1.0rem;line-height:1.5'>"
        "<b>Costs are included.</b> Every trade carries ~0.18% in costs (platform fees + slippage), "
        "already subtracted from the figures above. Estimated on <b>past data</b> \u2014 an order of "
        "magnitude, not a promise.</div>",
        unsafe_allow_html=True)
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    glossary([("Confidence bar", "the minimum certainty the AI must have before we buy (higher = fewer, safer trades)"),
              ("Round-trip cost", "everything paid to open AND close one trade: fees + slippage (~0.18%)"),
              ("Net / trade", "the average gain per trade once costs are subtracted"),
              ("Estimated", "measured on past data \u2014 an order of magnitude, not a guarantee")])


def s_next():
    head("bi-rocket-takeoff", "What's next")
    big("The AI is already <span style='color:#3BA99C'>live online</span> (on Railway), checking the "
        "robot's buys <b>every 5 minutes</b>.", size="1.35rem", weight=700)
    big("But it is <b>not yet working as planned in production</b> \u2014 the blocker is on the "
        "engine side, not the AI. I am fixing it.", size="1.2rem", weight=600, opacity=0.88)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    stat(a, "Fix", "The live pipeline", "Make the engine\u2013AI connection reliable.")
    stat(b, "Prove", "In real conditions", "Validate carefully with real (small) money.")
    stat(c, "Extend", "AI for selling too", "Today the AI only buys \u2014 next, let it help sell.")
    stat(d, "Grow", "Beyond quick wins", "Let gains run with the trailing stop instead of exiting at "
         "+0.95%.")
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;background:rgba(59,169,156,.10);border:1px solid #3BA99C;"
        "border-radius:10px;padding:10px 16px;font-size:1.05rem'>"
        "<b>A first test on our data:</b> letting winners run lifts the net gain per trade from "
        "<b>~0.2\u20130.3%</b> (fixed 1h) to <b>+1.3% (XGBoost)</b> and <b>+2.4% (Random Forest)</b>, "
        "with trades lasting minutes to a few hours. A promising direction \u2014 the next model should "
        "be trained for this longer-horizon exit, not a fixed 1h target. "
        "<span style='opacity:.7'>(Prudent simulation; does not yet model market gaps, so extreme losses are understated.)</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;font-size:1.15rem;margin-top:.6rem'>"
                f"<i class='bi bi-rocket-takeoff'></i> Deployed on Railway &nbsp;\u00b7&nbsp; "
                f"<i class='bi bi-github'></i> <a href='{GITHUB_URL}'>Project code on GitHub</a></div>",
                unsafe_allow_html=True)


def render_simplified():
    SLIDES = [("The robot", s_hook), ("How it works today", s_engine), ("The idea", s_idea),
              ("The data", s_data), ("What we found (EDA)", s_eda), ("The models", s_models),
              ("Does it work?", s_results), ("What could it earn?", s_earn),
              ("What's next", s_next)]
    if "ssec" not in st.session_state:
        st.session_state.ssec = 0
    n = len(SLIDES)
    st.session_state.ssec = max(0, min(st.session_state.ssec, n - 1))
    c1, c2, c3 = st.columns([1, 8, 1])
    if c1.button("Prev", disabled=st.session_state.ssec == 0, use_container_width=True):
        st.session_state.ssec -= 1; st.rerun()
    with c2:
        st.progress((st.session_state.ssec + 1) / n)
    if c3.button("Next", disabled=st.session_state.ssec == n - 1, use_container_width=True):
        st.session_state.ssec += 1; st.rerun()
    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        SLIDES[st.session_state.ssec][1]()
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    st.caption(f"Slide {st.session_state.ssec + 1} / {n} \u2014 {SLIDES[st.session_state.ssec][0]}")


SECTIONS = ["1 \u00b7 Overview", "2 \u00b7 The data", "3 \u00b7 EDA", "4 \u00b7 Models", "5 \u00b7 Results & demo"]
RENDER = [render_overview, render_data, render_eda, render_models, render_results]
if "sec" not in st.session_state:
    st.session_state.sec = 0

st.title("AI for Trading — an ML filter on top of a signal-based crypto engine")
st.caption("Jedha final project · Data Science Fullstack & Lead")

if simple:
    render_simplified()
else:
    tabs = st.tabs(SECTIONS)
    for tab, render in zip(tabs, RENDER):
        with tab:
            render()