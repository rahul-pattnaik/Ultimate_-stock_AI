# technical/supertrend.py
# ─────────────────────────────────────────────────────────────────────────────
# Supertrend Indicator
# Formula : midpoint ± (multiplier × ATR)
# Output  : trend direction, signal, line values, flip events
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    prev  = df["Close"].shift(1)
    tr    = pd.concat([
        high - low,
        (high - prev).abs(),
        (low  - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def supertrend(df: pd.DataFrame,
               period: int = 10,
               multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute the Supertrend indicator.

    Args:
        df         : yfinance OHLCV DataFrame
        period     : ATR period (default 10)
        multiplier : ATR band multiplier (default 3.0)

    Returns original df with added columns:
        supertrend      : the actual Supertrend line value
        supertrend_dir  : 1 = uptrend, -1 = downtrend
        supertrend_signal: 'BUY' | 'SELL' | 'HOLD'
        supertrend_flip : True where direction changed
    """
    if df is None or len(df) < period + 5:
        return df

    df   = df.copy()
    atr  = _atr_series(df, period)
    hl2  = (df["High"] + df["Low"]) / 2

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    trend = pd.Series(index=df.index, dtype=int)
    st    = pd.Series(index=df.index, dtype=float)

    close = df["Close"]

    for i in range(period, len(df)):
        idx      = df.index[i]
        prev_idx = df.index[i - 1]

        # Upper band: only lower if needed
        upper.iloc[i] = min(
            upper_basic.iloc[i],
            upper.iloc[i-1]
        ) if close.iloc[i-1] <= upper.iloc[i-1] else upper_basic.iloc[i]

        # Lower band: only raise if needed
        lower.iloc[i] = max(
            lower_basic.iloc[i],
            lower.iloc[i-1]
        ) if close.iloc[i-1] >= lower.iloc[i-1] else lower_basic.iloc[i]

        # Determine trend direction
        prev_trend = trend.iloc[i-1] if i > period else 1
        if prev_trend == -1 and close.iloc[i] > upper.iloc[i]:
            trend.iloc[i] = 1
        elif prev_trend == 1 and close.iloc[i] < lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend

        st.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]

    df["supertrend"]     = st.round(2)
    df["supertrend_dir"] = trend
    df["supertrend_flip"]= trend != trend.shift(1)

    # Signal
    conditions = [
        df["supertrend_flip"] & (df["supertrend_dir"] == 1),
        df["supertrend_flip"] & (df["supertrend_dir"] == -1),
    ]
    choices = ["BUY", "SELL"]
    df["supertrend_signal"] = np.select(conditions, choices, default="HOLD")

    return df


def supertrend_signal(df: pd.DataFrame,
                      period: int = 10,
                      multiplier: float = 3.0) -> dict:
    """
    Latest Supertrend reading.

    Returns: direction, signal, supertrend line value, distance from price.
    """
    if df is None or len(df) < period + 5:
        return {"error": "Insufficient data"}

    df_st = supertrend(df, period, multiplier)
    latest = df_st.iloc[-1]

    direction = "UPTREND 📈"  if latest["supertrend_dir"] == 1 else "DOWNTREND 📉"
    signal    = latest["supertrend_signal"]
    st_val    = round(float(latest["supertrend"]), 2)
    price     = round(float(latest["Close"]), 2)
    dist_pct  = round((price - st_val) / st_val * 100, 2)

    # Recent flip events
    flips = df_st[df_st["supertrend_flip"]].tail(3)
    recent_flips = [
        {"date": str(idx.date()), "direction": "BUY" if row["supertrend_dir"] == 1 else "SELL"}
        for idx, row in flips.iterrows()
    ]

    return {
        "direction":         direction,
        "signal":            signal,
        "supertrend_line":   st_val,
        "current_price":     price,
        "distance_pct":      f"{dist_pct:+.2f}%",
        "recent_flips":      recent_flips,
        "parameters":        {"period": period, "multiplier": multiplier},
    }
