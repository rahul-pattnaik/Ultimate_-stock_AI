# ai/random_forest_signal.py
# ─────────────────────────────────────────────────────────────────────────────
# Random Forest Trading Signal
# Features : RSI · MACD · MAs · Bollinger · Volume · ATR · Returns · Momentum
# Output   : BUY/SELL prediction + probability + feature importance
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features from raw OHLCV data.
    Returns a clean DataFrame ready for ML.
    """
    df = df.copy()
    close  = df["Close"]
    volume = df["Volume"]
    high   = df["High"]
    low    = df["Low"]

    # Moving Averages
    for w in [5, 10, 20, 50, 200]:
        df[f"ma{w}"] = close.rolling(w).mean()
        df[f"ma{w}_ratio"] = close / df[f"ma{w}"]   # price / MA ratio

    # EMA
    for span in [9, 21]:
        df[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()

    # RSI
    for period in [9, 14]:
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        df[f"rsi{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # Bollinger Bands
    bb_mean         = close.rolling(20).mean()
    bb_std          = close.rolling(20).std()
    df["bb_upper"]  = bb_mean + 2 * bb_std
    df["bb_lower"]  = bb_mean - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mean
    df["bb_pos"]    = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ATR (volatility)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr14"]     = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / close

    # Volume
    df["vol_ma20"]   = volume.rolling(20).mean()
    df["vol_ratio"]  = volume / df["vol_ma20"]
    df["vol_spike"]  = (df["vol_ratio"] > 1.5).astype(int)

    # Returns
    for n in [1, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = close.pct_change(n)

    # Momentum
    df["roc5"]  = close.pct_change(5)
    df["roc10"] = close.pct_change(10)

    # Candle body
    df["body"]       = (close - df["Open"]).abs() / (high - low + 1e-9)
    df["upper_wick"] = (high - close.clip(upper=close)) / (high - low + 1e-9)

    # Target: will next-day close be HIGHER?
    df["target"] = (close.shift(-1) > close).astype(int)

    return df


FEATURE_COLS = [
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma50_ratio",
    "ema9", "ema21",
    "rsi9", "rsi14",
    "macd", "macd_sig", "macd_hist",
    "bb_width", "bb_pos",
    "atr_ratio",
    "vol_ratio", "vol_spike",
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "roc5", "roc10",
    "body",
]


# ── Training ──────────────────────────────────────────────────────────────────

def train_rf(df: pd.DataFrame) -> tuple[RandomForestClassifier, StandardScaler, dict]:
    """
    Train a Random Forest on engineered features using TimeSeriesSplit CV.

    Returns:
        model    : trained RandomForestClassifier
        scaler   : fitted StandardScaler
        report   : accuracy metrics + feature importances
    """
    feat_df = build_features(df)
    feat_df = feat_df[FEATURE_COLS + ["target"]].dropna()

    X = feat_df[FEATURE_COLS].values
    y = feat_df["target"].values

    if len(X) < 100:
        raise ValueError("Need at least 100 rows after feature engineering")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TimeSeriesSplit — never leak future data into training
    tscv    = TimeSeriesSplit(n_splits=5)
    cv_accs = []

    for train_idx, val_idx in tscv.split(X_scaled):
        Xtr, Xval = X_scaled[train_idx], X_scaled[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        m = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,            # prevent overfitting
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        m.fit(Xtr, ytr)
        cv_accs.append(accuracy_score(yval, m.predict(Xval)))

    # Final model on all data
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    # Feature importance
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    report = {
        "cv_accuracy_mean": round(float(sum(cv_accs) / len(cv_accs)), 4),
        "cv_accuracy_std":  round(float(pd.Series(cv_accs).std()), 4),
        "top_features": [{"feature": f, "importance": round(float(i), 4)}
                         for f, i in importances[:10]],
        "training_samples": len(X),
    }

    return model, scaler, report


# ── Prediction ────────────────────────────────────────────────────────────────

def rf_signal(df: pd.DataFrame,
              model: RandomForestClassifier = None,
              scaler: StandardScaler = None) -> dict:
    """
    Train (if no model given) and predict signal for the latest row.

    Returns:
        signal      : "BUY" | "SELL"
        probability : confidence of prediction
        report      : training metrics
    """
    if model is None or scaler is None:
        model, scaler, report = train_rf(df)
    else:
        report = {}

    feat_df  = build_features(df)
    last_row = feat_df[FEATURE_COLS].dropna().iloc[[-1]]
    X_last   = scaler.transform(last_row.values)

    proba  = model.predict_proba(X_last)[0]
    pred   = int(model.predict(X_last)[0])

    signal = "BUY 📈" if pred == 1 else "SELL 📉"
    conf   = round(float(max(proba)), 4)

    return {
        "signal":      signal,
        "probability": f"{conf * 100:.1f}%",
        "buy_prob":    f"{proba[1] * 100:.1f}%",
        "sell_prob":   f"{proba[0] * 100:.1f}%",
        "training":    report,
    }
