# technical/volatility.py
# ─────────────────────────────────────────────────────────────────────────────
# Volatility Engine (no external `ta` dependency)
# Metrics: ATR · Historical Volatility · Bollinger Width · Keltner Channel
#          Z-Score · Volatility Regime
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── Core Calculations ─────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"].shift(1)
    tr    = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs()
    ], axis=1).max(axis=1)
    return tr


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).rolling(period).mean()


def _historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualised historical volatility using log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252) * 100   # in %


def _bollinger_width(close: pd.Series, period: int = 20) -> pd.Series:
    mean = close.rolling(period).mean()
    std  = close.rolling(period).std()
    return (4 * std / mean) * 100   # width as % of mid band


def _keltner_channel(df: pd.DataFrame, ema_period: int = 20,
                     atr_mult: float = 2.0) -> tuple:
    ema   = df["Close"].ewm(span=ema_period, adjust=False).mean()
    atr   = _atr_series(df, 14)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, ema, lower


# ── Public API ────────────────────────────────────────────────────────────────

def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility columns to DataFrame — backward compatible.
    Replaces `ta` dependency with pure pandas/numpy implementation.

    Columns added:
        atr         : Average True Range (14)
        atr_pct     : ATR as % of close
        hist_vol_20 : 20-day annualised historical volatility
        bb_width    : Bollinger Band width %
        kc_upper    : Keltner Channel upper
        kc_lower    : Keltner Channel lower
    """
    df = df.copy()

    df["atr"]         = _atr_series(df, 14)
    df["atr_pct"]     = df["atr"] / df["Close"] * 100
    df["hist_vol_20"] = _historical_volatility(df["Close"], 20)
    df["bb_width"]    = _bollinger_width(df["Close"], 20)

    kc_up, kc_mid, kc_lo = _keltner_channel(df)
    df["kc_upper"] = kc_up
    df["kc_lower"] = kc_lo

    return df


def volatility_report(df: pd.DataFrame) -> dict:
    """
    Full volatility snapshot for the latest bar.

    Returns:
        atr, atr_pct, hist_vol, bb_width, regime, z_score, keltner
    """
    if df is None or len(df) < 25:
        return {"error": "Need at least 25 rows"}

    df_v  = add_volatility(df)
    close = df["Close"].dropna()
    price = float(close.iloc[-1])

    atr      = round(float(df_v["atr"].iloc[-1]), 4)
    atr_pct  = round(float(df_v["atr_pct"].iloc[-1]), 2)
    hv20     = round(float(df_v["hist_vol_20"].iloc[-1]), 2)
    bw       = round(float(df_v["bb_width"].iloc[-1]), 2)

    # Volatility z-score: how extreme is current vol vs last 60 days?
    hv_series = df_v["hist_vol_20"].dropna().iloc[-60:]
    z_score   = round(float(
        (hv20 - hv_series.mean()) / (hv_series.std() + 1e-9)
    ), 2)

    # Regime classification
    if   hv20 < 15:                 regime = "Low Volatility 😴"
    elif hv20 < 30:                 regime = "Normal Volatility 🟡"
    elif hv20 < 50:                 regime = "High Volatility ⚠️"
    else:                           regime = "Extreme Volatility 🔴"

    if   z_score > 2:               z_label = "Unusually High Vol"
    elif z_score > 1:               z_label = "Above Average Vol"
    elif z_score < -1:              z_label = "Below Average Vol"
    else:                           z_label = "Normal Vol Range"

    # Keltner Channel position
    kc_up = round(float(df_v["kc_upper"].iloc[-1]), 2)
    kc_lo = round(float(df_v["kc_lower"].iloc[-1]), 2)

    if   price > kc_up: kc_pos = "Above Keltner (breakout)"
    elif price < kc_lo: kc_pos = "Below Keltner (breakdown)"
    else:               kc_pos = "Inside Keltner (normal)"

    return {
        "current_price":  round(price, 2),
        "ATR_14":         atr,
        "ATR_pct":        f"{atr_pct}%",
        "hist_vol_20d":   f"{hv20}%",
        "bb_width_pct":   f"{bw}%",
        "volatility_regime": regime,
        "z_score":        z_score,
        "z_label":        z_label,
        "keltner_upper":  kc_up,
        "keltner_lower":  kc_lo,
        "keltner_position": kc_pos,
    }
