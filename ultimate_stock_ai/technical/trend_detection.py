# technical/trend_detection.py
# ─────────────────────────────────────────────────────────────────────────────
# Trend Detection Engine
# FIXES:
#   ✅ ADX ValueError ("setting an array element with a sequence")
#      Root cause: yfinance MultiIndex → close was 2D. Fixed with .squeeze()
#      and explicit scalar extraction at each loop step.
#   ✅ detect_trend() now uses squeeze + flatten everywhere
#   ✅ trend_analysis() full multi-timeframe report
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_1d(arr) -> np.ndarray:
    """Guarantee a 1-D float numpy array (fixes MultiIndex side effects)."""
    if isinstance(arr, pd.Series):
        arr = arr.values
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average Directional Index — measures trend STRENGTH (0-100).
    > 25 = strong trend, < 20 = weak/sideways.

    FIX: all intermediate arrays use explicit scalar indexing to avoid the
    "setting an array element with a sequence" ValueError that occurs when
    the Close column is 2-D (yfinance MultiIndex artefact).
    """
    high  = _to_1d(df["High"])
    low   = _to_1d(df["Low"])
    close = _to_1d(df["Close"])

    n = len(close)
    if n < period + 2:
        return 0.0

    plus_dm  = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    tr_arr   = np.zeros(n, dtype=float)

    for i in range(1, n):
        # Explicit float() conversion guarantees scalar assignment
        h_curr  = float(high[i]);   h_prev = float(high[i - 1])
        l_curr  = float(low[i]);    l_prev = float(low[i - 1])
        c_prev  = float(close[i - 1])

        h_diff = h_curr - h_prev
        l_diff = l_prev - l_curr

        plus_dm[i]  = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
        tr_arr[i]   = max(h_curr - l_curr,
                          abs(h_curr - c_prev),
                          abs(l_curr - c_prev))

    def _smooth(arr: np.ndarray, p: int) -> np.ndarray:
        out = [float(arr[:p + 1].sum())]
        for v in arr[p + 1:]:
            out.append(out[-1] - out[-1] / p + float(v))
        return np.array(out, dtype=float)

    s_tr    = _smooth(tr_arr[1:],   period)
    s_plus  = _smooth(plus_dm[1:],  period)
    s_minus = _smooth(minus_dm[1:], period)

    di_plus  = 100.0 * s_plus  / (s_tr + 1e-9)
    di_minus = 100.0 * s_minus / (s_tr + 1e-9)
    dx       = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)

    adx_arr = _smooth(dx, period)
    return float(adx_arr[-1])


def _linear_slope(close: np.ndarray, window: int = 20) -> float:
    """Slope of best-fit line over `window` bars, normalised by price."""
    close = _to_1d(close)
    if len(close) < window:
        return 0.0
    y = close[-window:]
    x = np.arange(window, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / float(close[-window])


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    arr = _to_1d(arr)
    k   = 2.0 / (span + 1)
    out = [float(arr[0])]
    for v in arr[1:]:
        out.append(float(v) * k + out[-1] * (1.0 - k))
    return np.array(out, dtype=float)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame) -> str:
    """
    Classic trend label — backward compatible with main_terminal.py.
    Uses MA50 vs MA200 cross with ADX confirmation.
    """
    if df is None or len(df) < 50:
        return "Insufficient Data"

    # .squeeze() converts any 2-D single-column DataFrame/array to 1-D
    close = pd.Series(_to_1d(df["Close"])).dropna()

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
    Returns: trend label, strength, ADX, slope, MA values, timeframes.
    """
    if df is None or len(df) < 50:
        return {"error": "Need at least 50 rows"}

    close = _to_1d(df["Close"])
    close = close[~np.isnan(close)]
    price = float(close[-1])
    s     = pd.Series(close)

    ma20  = float(s.rolling(20).mean().iloc[-1])
    ma50  = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(min(200, len(close))).mean().iloc[-1])
    ema9  = float(_ema(close, 9)[-1])
    ema21 = float(_ema(close, 21)[-1])
    ema50 = float(_ema(close, 50)[-1])

    short_bull  = ema9  > ema21
    medium_bull = ma20  > ma50
    long_bull   = ma50  > ma200
    bulls = sum([short_bull, medium_bull, long_bull])

    adx   = _adx(df)
    slope = _linear_slope(close, window=20)

    if   bulls == 3 and adx > 25: trend = "Strong Uptrend 📈"
    elif bulls >= 2:               trend = "Uptrend 📈"
    elif bulls == 0 and adx > 25: trend = "Strong Downtrend 📉"
    elif bulls <= 1:               trend = "Downtrend 📉"
    else:                          trend = "Sideways ➡️"

    if   adx >= 40: strength = "Very Strong"
    elif adx >= 25: strength = "Strong"
    elif adx >= 20: strength = "Moderate"
    else:           strength = "Weak / Sideways"

    def pct_from(ma): return round((price - ma) / ma * 100, 2)

    return {
        "trend":    trend,
        "strength": strength,
        "adx":      round(adx, 2),
        "slope_pct": round(slope * 100, 4),
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
