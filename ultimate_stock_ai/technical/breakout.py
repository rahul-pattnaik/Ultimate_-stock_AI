# technical/breakout.py
# ─────────────────────────────────────────────────────────────────────────────
# Breakout Detection Engine
# Types: Price breakout · Volume breakout · Bollinger squeeze · ATR expansion
# Output: signal bool (backward compat) + detailed dict
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h = df["High"].values[-(period + 1):]
    l = df["Low"].values[-(period + 1):]
    c = df["Close"].values[-(period + 1):]
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(tr.mean())


def _bollinger_squeeze(close: pd.Series, period: int = 20) -> bool:
    """Bollinger Bands squeeze — low bandwidth signals coiled spring."""
    mean = close.rolling(period).mean()
    std  = close.rolling(period).std()
    bw   = ((mean + 2*std) - (mean - 2*std)) / mean  # bandwidth
    # Squeeze: current BW in lowest 20% of last 60 bars
    bw_window = bw.dropna().iloc[-60:]
    if len(bw_window) < 20:
        return False
    return float(bw.iloc[-1]) < float(bw_window.quantile(0.20))


def _volume_confirmed(df: pd.DataFrame, multiplier: float = 1.5) -> bool:
    """Volume must be above average to confirm a breakout."""
    avg_vol  = float(df["Volume"].iloc[-20:].mean())
    last_vol = float(df["Volume"].iloc[-1])
    return last_vol > avg_vol * multiplier


# ── Public API ────────────────────────────────────────────────────────────────

def breakout_signal(df: pd.DataFrame) -> bool:
    """
    Simple breakout signal — backward compatible with main_terminal.py.
    Returns True if price broke above 20-day high resistance.
    """
    if df is None or len(df) < 22:
        return False

    last_close      = float(df["Close"].iloc[-1])
    resistance_20d  = float(df["High"].rolling(20).max().iloc[-2])   # yesterday's high
    return last_close > resistance_20d


def breakout_analysis(df: pd.DataFrame) -> dict:
    """
    Comprehensive breakout analysis across multiple dimensions.

    Checks:
        1. 20-day price breakout (classic)
        2. 52-week high breakout
        3. Volume surge
        4. Bollinger squeeze + expansion
        5. ATR expansion (momentum burst)
        6. EMA crossover momentum

    Returns signal strength (0-100) + type labels + details.
    """
    if df is None or len(df) < 30:
        return {"error": "Need at least 30 rows"}

    close = df["Close"].dropna()
    price = float(close.iloc[-1])

    signals  = []
    score    = 0
    details  = {}

    # 1. 20-day price breakout
    res_20d = float(df["High"].rolling(20).max().iloc[-2])
    bo_20d  = price > res_20d
    details["price_breakout_20d"] = {"triggered": bo_20d,
                                     "resistance": round(res_20d, 2)}
    if bo_20d:
        score += 25
        signals.append("20-day price breakout 🚀")

    # 2. 52-week high
    high_52w = float(df["High"].rolling(min(252, len(df))).max().iloc[-1])
    near_52w = price >= 0.98 * high_52w
    details["near_52w_high"] = {"triggered": near_52w,
                                 "52w_high": round(high_52w, 2)}
    if near_52w:
        score += 20
        signals.append("Near / at 52-week high 💪")

    # 3. Volume surge
    vol_confirmed = _volume_confirmed(df, multiplier=1.5)
    avg_vol  = float(df["Volume"].iloc[-20:].mean())
    last_vol = float(df["Volume"].iloc[-1])
    vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0
    details["volume_surge"] = {"triggered": vol_confirmed,
                                "ratio": vol_ratio}
    if vol_confirmed:
        score += 20
        signals.append(f"Volume surge {vol_ratio}x avg 🔥")

    # 4. Bollinger squeeze (coiled spring)
    squeeze = _bollinger_squeeze(close)
    details["bollinger_squeeze"] = {"triggered": squeeze}
    if squeeze:
        score += 15
        signals.append("Bollinger squeeze — breakout potential ⚡")

    # 5. ATR expansion (volatility burst)
    atr_now  = _atr(df, 7)    # short-term ATR
    atr_norm = _atr(df, 20)   # normal ATR
    atr_expansion = atr_now > atr_norm * 1.3
    details["atr_expansion"] = {
        "triggered": atr_expansion,
        "atr_7d": round(atr_now, 2),
        "atr_20d": round(atr_norm, 2),
    }
    if atr_expansion:
        score += 10
        signals.append("ATR expansion — momentum burst 💥")

    # 6. EMA fast/slow crossover
    ema9  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_cross = ema9 > ema21
    prev_ema9  = float(close.iloc[:-1].ewm(span=9,  adjust=False).mean().iloc[-1])
    prev_ema21 = float(close.iloc[:-1].ewm(span=21, adjust=False).mean().iloc[-1])
    fresh_cross = ema_cross and (prev_ema9 <= prev_ema21)
    details["ema_crossover"] = {"triggered": fresh_cross,
                                 "ema9": round(ema9, 2),
                                 "ema21": round(ema21, 2)}
    if fresh_cross:
        score += 10
        signals.append("Fresh EMA 9/21 bullish crossover ✅")

    score = min(100, score)

    if   score >= 75: label = "STRONG BREAKOUT 🚀"
    elif score >= 50: label = "BREAKOUT 📈"
    elif score >= 25: label = "POTENTIAL BREAKOUT ⚡"
    else:             label = "NO BREAKOUT ➡️"

    return {
        "signal":       label,
        "score":        score,
        "triggered":    score >= 50,
        "signals_hit":  signals,
        "details":      details,
        "current_price": round(price, 2),
    }
