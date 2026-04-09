# technical/trend_detection.py
# ─────────────────────────────────────────────────────────────────────────────
# Trend Detection Engine
# Methods: MA cross · ADX (trend strength) · Linear slope · Multi-timeframe
# Output : label (backward compat) + detailed trend dict
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average Directional Index — measures STRENGTH of trend (0-100).
    >25 = strong trend, <20 = weak/sideways.
    """
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.flatten().astype(float)

    n = len(close)
    if n < period + 2:
        return 0.0

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_arr   = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i]  - high[i-1]
        l_diff = low[i-1] - low[i]
        plus_dm[i]  = h_diff if h_diff > l_diff and h_diff > 0 else 0
        minus_dm[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0
        tr_arr[i]   = max(high[i] - low[i],
                          abs(high[i] - close[i-1]),
                          abs(low[i]  - close[i-1]))

    def smooth(arr, p):
        out = [arr[:p+1].sum()]
        for v in arr[p+1:]:
            out.append(out[-1] - out[-1]/p + v)
        return np.array(out)

    s_tr    = smooth(tr_arr[1:],    period)
    s_plus  = smooth(plus_dm[1:],   period)
    s_minus = smooth(minus_dm[1:],  period)

    di_plus  = 100 * s_plus  / (s_tr + 1e-9)
    di_minus = 100 * s_minus / (s_tr + 1e-9)
    dx       = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)

    adx_arr = smooth(dx, period)
    return float(adx_arr[-1])


def _linear_slope(close: np.ndarray, window: int = 20) -> float:
    """Slope of best-fit line over `window` bars, normalised by price."""
    if len(close) < window:
        return 0.0
    y = close[-window:]
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    return slope / close[-window]   # normalise: slope as % of price per bar


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    k, out = 2 / (span + 1), [arr[0]]
    for v in arr[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return np.array(out)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame) -> str:
    """
    Classic trend label — backward compatible with main_terminal.py.
    Uses MA50 vs MA200 cross with ADX confirmation.
    """
    if df is None or len(df) < 50:
        return "Insufficient Data"

    close = df["Close"].dropna().squeeze().astype(float)

    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])
    adx   = _adx(df)

    if   ma50 > ma200 and adx > 20: return "Strong Uptrend 📈"
    elif ma50 > ma200:               return "Weak Uptrend 📈"
    elif ma50 < ma200 and adx > 20: return "Strong Downtrend 📉"
    elif ma50 < ma200:               return "Weak Downtrend 📉"
    else:                            return "Sideways ➡️"


def trend_analysis(df: pd.DataFrame) -> dict:
    """
    Full multi-timeframe trend analysis.

    Covers:
        Short-term  : EMA9 vs EMA21 (days)
        Medium-term : MA20 vs MA50  (weeks)
        Long-term   : MA50 vs MA200 (months)
        Trend strength: ADX
        Trend slope   : linear regression angle
        Price position: % above/below key MAs

    Returns: trend label, strength, direction, ADX, slope, all MA values.
    """
    if df is None or len(df) < 50:
        return {"error": "Need at least 50 rows"}

    close = df["Close"].dropna().squeeze().values.astype(float)
    price = close[-1]
    s     = pd.Series(close)

    # MAs
    ma20  = float(s.rolling(20).mean().iloc[-1])
    ma50  = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(min(200, len(close))).mean().iloc[-1])
    ema9  = float(_ema(close, 9)[-1])
    ema21 = float(_ema(close, 21)[-1])
    ema50 = float(_ema(close, 50)[-1])

    # Signals per timeframe
    short_bull  = ema9  > ema21
    medium_bull = ma20  > ma50
    long_bull   = ma50  > ma200

    bulls = sum([short_bull, medium_bull, long_bull])

    adx   = _adx(df)
    slope = _linear_slope(close, window=20)

    # Trend label
    if   bulls == 3 and adx > 25: trend = "Strong Uptrend 📈"
    elif bulls >= 2:               trend = "Uptrend 📈"
    elif bulls == 0 and adx > 25: trend = "Strong Downtrend 📉"
    elif bulls <= 1:               trend = "Downtrend 📉"
    else:                          trend = "Sideways ➡️"

    # Trend strength category
    if   adx >= 40: strength = "Very Strong"
    elif adx >= 25: strength = "Strong"
    elif adx >= 20: strength = "Moderate"
    else:           strength = "Weak / Sideways"

    # Distance from key MAs (%)
    def pct_from(ma): return round((price - ma) / ma * 100, 2)

    return {
        "trend":          trend,
        "strength":       strength,
        "adx":            round(adx, 2),
        "slope_pct":      round(slope * 100, 4),
        "timeframes": {
            "short_term":  "Bullish ✅" if short_bull  else "Bearish ❌",
            "medium_term": "Bullish ✅" if medium_bull else "Bearish ❌",
            "long_term":   "Bullish ✅" if long_bull   else "Bearish ❌",
        },
        "moving_averages": {
            "EMA9":  round(ema9, 2),  "EMA21": round(ema21, 2),
            "MA20":  round(ma20, 2),  "MA50":  round(ma50, 2),
            "MA200": round(ma200, 2), "EMA50": round(ema50, 2),
        },
        "price_vs_ma": {
            "vs_MA20":  f"{pct_from(ma20):+.2f}%",
            "vs_MA50":  f"{pct_from(ma50):+.2f}%",
            "vs_MA200": f"{pct_from(ma200):+.2f}%",
        },
        "current_price": round(price, 2),
    }
