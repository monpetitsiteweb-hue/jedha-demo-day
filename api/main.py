"""
Savvy Crypto Agent — ML Inference Service
FastAPI service serving XGBoost + LSTM + Ensemble predictions
Deploy on Railway
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Savvy Crypto ML Service", version="1.0.0")

# ─── Load models at startup ───────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "./models")

xgb_model = None
rf_model = None
lstm_model = None
scaler_lstm = None
feature_cols = None
ml_config = None


@app.on_event("startup")
async def load_models():
    global xgb_model, rf_model, lstm_model, scaler_lstm, feature_cols, ml_config

    try:
        logger.info("Downloading models from Supabase Storage...")
        
        from supabase import create_client
        SUPABASE_URL = os.environ["SUPABASE_URL"]   # à configurer (variable d'env Railway)
        SUPABASE_KEY = os.environ["SUPABASE_KEY"]   # service_role key — à configurer, jamais en dur
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        files = ['feature_cols.json', 'ml_config.json', 'xgb_final.pkl', 
                 'rf_final.pkl', 'scaler_lstm.pkl', 'lstm_final.keras']
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        for f in files:
            path = f"{MODELS_DIR}/{f}"
            if True:  # Toujours re-télécharger pour garantir la version Supabase
                logger.info(f"Downloading {f}...")
                data = sb.storage.from_('ml-models').download(f)
                with open(path, 'wb') as out:
                    out.write(data)
                logger.info(f"✅ {f} downloaded")

        with open(f"{MODELS_DIR}/feature_cols.json") as f:
            feature_cols = json.load(f)
        with open(f"{MODELS_DIR}/ml_config.json") as f:
            ml_config = json.load(f)

        xgb_model = joblib.load(f"{MODELS_DIR}/xgb_final.pkl")
        logger.info("✅ XGBoost loaded")
        rf_model = joblib.load(f"{MODELS_DIR}/rf_final.pkl")
        logger.info("✅ Random Forest loaded")
        scaler_lstm = joblib.load(f"{MODELS_DIR}/scaler_lstm.pkl")

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from tensorflow import keras
        lstm_model = keras.models.load_model(f"{MODELS_DIR}/lstm_final.keras")
        logger.info("✅ LSTM loaded")

        logger.info(f"✅ All models ready — threshold={ml_config['ensemble_threshold']}")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise


# ─── Feature Engineering (from notebook 01) ──────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def add_features(df_sym: pd.DataFrame) -> pd.DataFrame:
    """Compute 62 pure-5m features — exact replica of notebook 01. The model uses a 10-feature subset selected via feature_cols.json."""
    d = df_sym.copy().sort_values('candle_time').reset_index(drop=True)
    c = d['close_price']
    h = d['high_price']
    l = d['low_price']
    o = d['open_price']
    v = d['volume']

    # ── TREND ─────────────────────────────────────────────────────────────────
    d['ema9']   = c.ewm(span=9,   adjust=False).mean()
    d['ema21']  = c.ewm(span=21,  adjust=False).mean()
    d['ema50']  = c.ewm(span=50,  adjust=False).mean()
    d['ema200'] = c.ewm(span=200, adjust=False).mean()

    d['price_vs_ema9']   = (c - d['ema9'])   / d['ema9']   * 100
    d['price_vs_ema21']  = (c - d['ema21'])  / d['ema21']  * 100
    d['price_vs_ema50']  = (c - d['ema50'])  / d['ema50']  * 100
    d['price_vs_ema200'] = (c - d['ema200']) / d['ema200'] * 100

    d['ema21_slope']  = d['ema21'].pct_change(3)  * 100
    d['ema50_slope']  = d['ema50'].pct_change(6)  * 100

    d['ema_aligned_bull'] = ((d['ema9'] > d['ema21']) & (d['ema21'] > d['ema50'])).astype(int)
    d['ema_aligned_bear'] = ((d['ema9'] < d['ema21']) & (d['ema21'] < d['ema50'])).astype(int)

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    d['rsi14'] = compute_rsi(c, 14)
    d['rsi7']  = compute_rsi(c, 7)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['macd']        = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist']   = d['macd'] - d['macd_signal']
    d['macd_cross_bull'] = ((d['macd'] > d['macd_signal']) & (d['macd'].shift(1) <= d['macd_signal'].shift(1))).astype(int)
    d['macd_cross_bear'] = ((d['macd'] < d['macd_signal']) & (d['macd'].shift(1) >= d['macd_signal'].shift(1))).astype(int)

    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    d['stoch_k'] = (c - low14) / (high14 - low14 + 1e-10) * 100
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()

    d['ret_1']   = c.pct_change(1)   * 100
    d['ret_3']   = c.pct_change(3)   * 100
    d['ret_12']  = c.pct_change(12)  * 100
    d['ret_48']  = c.pct_change(48)  * 100
    d['ret_288'] = c.pct_change(288) * 100

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    d['atr14']   = compute_atr(h, l, c, 14)
    d['atr_pct'] = d['atr14'] / c * 100

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d['bb_upper'] = bb_mid + 2 * bb_std
    d['bb_lower'] = bb_mid - 2 * bb_std
    d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / bb_mid * 100
    d['bb_pos']   = (c - d['bb_lower']) / (d['bb_upper'] - d['bb_lower'] + 1e-10)

    d['vol_realized_1h'] = d['ret_1'].rolling(12).std()

    # ── VOLUME ────────────────────────────────────────────────────────────────
    vol_ma20 = v.rolling(20).mean()
    d['volume_ratio']  = v / (vol_ma20 + 1e-10)
    d['buy_pressure']  = (c - l) / (h - l + 1e-10)
    d['sell_pressure'] = (h - c) / (h - l + 1e-10)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d['obv_slope'] = obv.pct_change(12) * 100

    # ── CANDLE STRUCTURE ──────────────────────────────────────────────────────
    body        = (c - o).abs()
    candle_range = (h - l + 1e-10)
    d['body_ratio']       = body / candle_range
    d['upper_wick_ratio'] = (h - c.clip(upper=o + body)) / candle_range
    d['lower_wick_ratio'] = (c.clip(lower=o - body) - l) / candle_range
    d['is_bullish_candle'] = (c > o).astype(int)

    # ── TIME CONTEXT ──────────────────────────────────────────────────────────
    d['hour_utc']    = d['candle_time'].dt.hour
    d['dow']         = d['candle_time'].dt.dayofweek
    d['is_weekend']  = (d['dow'] >= 5).astype(int)
    d['session_asia']   = d['hour_utc'].between(0, 8).astype(int)
    d['session_europe'] = d['hour_utc'].between(8, 16).astype(int)
    d['session_us']     = d['hour_utc'].between(13, 22).astype(int)

    # ── REGIME ────────────────────────────────────────────────────────────────

    def compute_adx(high, low, close, period=14):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(com=period-1, min_periods=period).mean()

        up_move   = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di  = 100 * pd.Series(plus_dm,  index=close.index).ewm(com=period-1, min_periods=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(com=period-1, min_periods=period).mean() / atr

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(com=period-1, min_periods=period).mean()
        return adx, plus_di, minus_di

    adx, plus_di, minus_di = compute_adx(h, l, c, 14)
    d['adx_14']      = adx
    d['plus_di']     = plus_di
    d['minus_di']    = minus_di
    d['is_trending'] = (adx > 25).astype(int)
    d['is_ranging']  = (adx < 20).astype(int)

    d['ema200_slope_12']  = d['ema200'].pct_change(12)  * 100
    d['ema200_slope_48']  = d['ema200'].pct_change(48)  * 100
    d['ema200_slope_288'] = d['ema200'].pct_change(288) * 100
    d['trend_direction']  = np.sign(d['ema200_slope_48'])

    d['atr_ratio_short'] = d['atr14'] / (d['atr14'].rolling(48).mean()  + 1e-10)
    d['atr_ratio_long']  = d['atr14'] / (d['atr14'].rolling(288).mean() + 1e-10)

    d['higher_high_12'] = (h > h.shift(12)).astype(int)
    d['lower_low_12']   = (l < l.shift(12)).astype(int)
    d['hh_ll_bull']     = ((d['higher_high_12'] == 1) & (d['lower_low_12'] == 0)).astype(int)
    d['hh_ll_bear']     = ((d['higher_high_12'] == 0) & (d['lower_low_12'] == 1)).astype(int)

    return d


# ─── API Models ───────────────────────────────────────────────────────────────

class Candle(BaseModel):
    candle_time: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float


class PredictRequest(BaseModel):
    symbol: str
    candles: list[Candle]  # Last 400+ candles (need 288 for ema200_slope_288 warmup)


class PredictResponse(BaseModel):
    symbol: str
    signal: bool              # True = BUY signal
    ensemble_prob: float      # Ensemble probability [0,1]
    xgb_prob: float
    lstm_prob: float
    stoch_k: float
    rsi14: float
    would_filter: bool        # True = ML would block this trade
    eda_signal: bool          # True = mean reversion signal detected
    threshold_used: float
    timestamp: str


# ─── Prediction endpoint ──────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not all([xgb_model, rf_model, lstm_model, feature_cols, ml_config]):
        raise HTTPException(status_code=503, detail="Models not loaded")

    if len(request.candles) < 350:
        raise HTTPException(status_code=400,
            detail=f"Need at least 350 candles, got {len(request.candles)}")

    try:
        # Build DataFrame
        df = pd.DataFrame([c.dict() for c in request.candles])
        df['candle_time'] = pd.to_datetime(df['candle_time'], utc=True)
        df = df.sort_values('candle_time').reset_index(drop=True)

        # Compute features
        df_feat = add_features(df)

        # Get last row (current candle)
        last = df_feat.iloc[-1]

        # Debug — compter les features à 0 ou NaN
        X_values = df_feat[feature_cols].iloc[[-1]]
        zero_count = (X_values == 0).sum(axis=1).values[0]
        nan_count = X_values.isnull().sum(axis=1).values[0]
        logger.info(f"[{request.symbol}] features zero={zero_count} nan={nan_count} "
                    f"stoch_k={X_values['stoch_k'].values[0]:.2f} "
                    f"ema200={X_values['ema200'].values[0]:.2f}")

        # Appliquer ffill sur tout le dataset avant extraction
        # (propage les dernières valeurs connues ; fillna(0) reste en dernier recours)
        df_feat[feature_cols] = df_feat[feature_cols].ffill().bfill().fillna(0)

        # Extract feature vector
        X = df_feat[feature_cols].iloc[[-1]].fillna(0)

        # ── XGBoost prediction ────────────────────────────────────────────────
        xgb_prob = float(xgb_model.predict_proba(X)[0][1])

        logger.info(f"[{request.symbol}] XGB_DEBUG "
                    f"xgb_prob={xgb_prob:.4f} "
                    f"stoch_k={float(last.get('stoch_k',0)):.2f} "
                    f"rsi14={float(last.get('rsi14',50)):.2f} "
                    f"lower_wick={float(last.get('lower_wick_ratio',0)):.4f} "
                    f"buy_pressure={float(last.get('buy_pressure',0)):.4f} "
                    f"zero_count={int((df_feat[feature_cols].iloc[[-1]]==0).sum(axis=1).values[0])}")

        # ── Random Forest prediction ──────────────────────────────────────────
        rf_prob = float(rf_model.predict_proba(X)[0][1])

        # ── LSTM prediction ───────────────────────────────────────────────────
        SEQ_LEN = 30
        if len(df_feat) >= SEQ_LEN:
            X_seq = df_feat[feature_cols].iloc[-SEQ_LEN:].fillna(0).values
            X_scaled = scaler_lstm.transform(X_seq)
            X_lstm = X_scaled.reshape(1, SEQ_LEN, len(feature_cols))
            lstm_prob = float(lstm_model.predict(X_lstm, verbose=0)[0][0])
        else:
            lstm_prob = 0.0

        # ── Ensemble (LSTM + XGB 50/50, matches training notebook 05) ────────
        ensemble_prob = 0.5 * lstm_prob + 0.5 * xgb_prob

        threshold = ml_config["ensemble_threshold"]  # 0.97
        signal = ensemble_prob >= threshold

        # EDA signal = mean reversion conditions (Stoch K < 10, RSI < 30)
        stoch_k = float(last.get('stoch_k', 0) or 0)
        rsi14   = float(last.get('rsi14', 50) or 50)
        eda_signal = stoch_k < 10 and rsi14 < 30

        # would_filter = ML would block this trade
        would_filter = not signal

        logger.info(f"[{request.symbol}] XGB={xgb_prob:.3f} RF={rf_prob:.3f} LSTM={lstm_prob:.3f} "
                    f"Ensemble={ensemble_prob:.3f} signal={signal} "
                    f"stoch_k={stoch_k:.1f} rsi14={rsi14:.1f}")

        return PredictResponse(
            symbol=request.symbol,
            signal=signal,
            ensemble_prob=round(ensemble_prob, 4),
            xgb_prob=round(xgb_prob, 4),
            lstm_prob=round(lstm_prob, 4),
            stoch_k=round(stoch_k, 2),
            rsi14=round(rsi14, 2),
            would_filter=would_filter,
            eda_signal=eda_signal,
            threshold_used=threshold,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/healthz")
async def health():
    models_ok = all([xgb_model, rf_model, lstm_model])
    return {
        "status": "ok" if models_ok else "degraded",
        "models_loaded": {
            "xgboost": xgb_model is not None,
            "random_forest": rf_model is not None,
            "lstm": lstm_model is not None,
        },
        "feature_count": len(feature_cols) if feature_cols else 0,
        "ensemble_threshold": ml_config.get("ensemble_threshold") if ml_config else None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    return {"service": "Savvy Crypto ML Service", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
