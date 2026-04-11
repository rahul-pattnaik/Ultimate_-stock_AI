# ai/signal_model.py
# ─────────────────────────────────────────────────────────────────────────────
# AI Signal Generator
# Uses: RSI · MACD · Bollinger Bands · EMA crossover · Volume · ATR
# Returns: signal label, numeric score, full reasoning
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    k, out = 2 / (span + 1), [arr[0]]
    for v in arr[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return np.array(out)


def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains  = np.where(deltas > 0,  deltas, 0.0).mean()
    losses = np.where(deltas < 0, -deltas, 0.0).mean()
    if losses == 0:
        return 100.0
    return float(100 - 100 / (1 + gains / losses))


def _macd(close: np.ndarray):
    line   = _ema(close, 12) - _ema(close, 26)
    signal = _ema(line, 9)
    return float(line[-1]), float(signal[-1]), float(line[-1] - signal[-1])


def _bollinger(close: np.ndarray, period: int = 20):
    window = close[-period:]
    mean, std = window.mean(), window.std()
    return mean + 2 * std, mean, mean - 2 * std  # upper, mid, lower


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h = df["High"].values[-(period + 1):]
    l = df["Low"].values[-(period + 1):]
    c = df["Close"].values[-(period + 1):]
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(tr.mean())


# ── Signal Engine ─────────────────────────────────────────────────────────────

def ai_signal(df: pd.DataFrame) -> dict:
    """
    Multi-indicator confluence signal.

    Scoring breakdown (total 100 pts):
        RSI            25 pts
        MACD           25 pts
        Bollinger      20 pts
        EMA crossover  20 pts
        Volume         10 pts

    Returns:
        signal, score (0-100), all indicator values, reasoning list
    """
    if df is None or len(df) < 50:
        return {"error": "Need at least 50 days of data"}

    df    = df.copy()
    close = df["Close"].dropna().values.flatten().astype(float)
    price = close[-1]

    rsi                     = _rsi(close)
    macd_line, sig_line, hist = _macd(close)
    upper_bb, mid_bb, lower_bb = _bollinger(close)
    ema9  = float(_ema(close, 9)[-1])
    ema21 = float(_ema(close, 21)[-1])
    avg_vol  = float(df["Volume"].iloc[-20:].mean())
    last_vol = float(df["Volume"].iloc[-1])
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
    atr = _atr(df)

    score   = 0
    reasons = []

    # ── RSI (25 pts) ─────────────────────────────────────────────────
    if rsi < 30:
        score += 25; reasons.append(f"RSI {rsi:.1f} → Oversold — strong buy zone 🟢")
    elif rsi < 40:
        score += 18; reasons.append(f"RSI {rsi:.1f} → Mildly oversold")
    elif rsi < 55:
        score += 12; reasons.append(f"RSI {rsi:.1f} → Neutral")
    elif rsi < 70:
        score += 5;  reasons.append(f"RSI {rsi:.1f} → Approaching overbought ⚠️")
    else:
        score += 0;  reasons.append(f"RSI {rsi:.1f} → Overbought — sell pressure 🔴")

    # ── MACD (25 pts) ────────────────────────────────────────────────
    if macd_line > sig_line and hist > 0:
        score += 25; reasons.append("MACD: bullish crossover + positive histogram ✅")
    elif macd_line > sig_line:
        score += 15; reasons.append("MACD: above signal line (weak bull)")
    elif macd_line < sig_line and hist < 0:
        score += 0;  reasons.append("MACD: bearish crossover + negative histogram ❌")
    else:
        score += 8;  reasons.append("MACD: below signal line (weak bear)")

    # ── Bollinger Bands (20 pts) ─────────────────────────────────────
    bb_pos = (price - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5
    if price <= lower_bb:
        score += 20; reasons.append("Price at/below lower BB → Oversold bounce zone 🔥")
    elif bb_pos < 0.35:
        score += 14; reasons.append("Price in lower BB zone (buy region)")
    elif bb_pos < 0.65:
        score += 8;  reasons.append("Price mid-BB (neutral)")
    elif price >= upper_bb:
        score += 0;  reasons.append("Price at/above upper BB → Overbought 🔴")
    else:
        score += 4;  reasons.append("Price in upper BB zone (caution)")

    # ── EMA Crossover (20 pts) ───────────────────────────────────────
    if ema9 > ema21:
        score += 20; reasons.append(f"EMA9 {ema9:.2f} > EMA21 {ema21:.2f} → Bullish ✅")
    else:
        score += 0;  reasons.append(f"EMA9 {ema9:.2f} < EMA21 {ema21:.2f} → Bearish ❌")

    # ── Volume Confirmation (10 pts) ─────────────────────────────────
    if vol_ratio > 1.5:
        score += 10; reasons.append(f"Volume {vol_ratio:.1f}x avg → Signal confirmed 🔥")
    elif vol_ratio > 1.0:
        score += 6;  reasons.append(f"Volume {vol_ratio:.1f}x avg → Moderate confirmation")
    elif vol_ratio < 0.7:
        score += 2;  reasons.append(f"Volume {vol_ratio:.1f}x avg → Weak signal ⚠️")
    else:
        score += 4;  reasons.append(f"Volume {vol_ratio:.1f}x avg → Normal")

    score = int(np.clip(score, 0, 100))

    # ── Final Signal Label ────────────────────────────────────────────
    if   score >= 80: signal = "STRONG BUY 🚀"
    elif score >= 65: signal = "BUY 📈"
    elif score >= 45: signal = "HOLD ➡️"
    elif score >= 30: signal = "SELL 📉"
    else:             signal = "STRONG SELL 🔴"

    return {
        "signal":         signal,
        "score":          score,
        "indicators": {
            "RSI":            round(rsi, 2),
            "MACD":           round(macd_line, 4),
            "MACD_signal":    round(sig_line, 4),
            "MACD_histogram": round(hist, 4),
            "BB_upper":       round(upper_bb, 2),
            "BB_lower":       round(lower_bb, 2),
            "BB_position_pct":round(bb_pos * 100, 1),
            "EMA9":           round(ema9, 2),
            "EMA21":          round(ema21, 2),
            "volume_ratio":   round(vol_ratio, 2),
            "ATR":            round(atr, 2),
        },
        "reasons": reasons,
    }
