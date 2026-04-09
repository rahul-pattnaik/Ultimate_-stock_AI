# ai/ranking_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Master Stock Scoring Engine
# Combines: Trend · Momentum · Breakout · Volatility · Volume · RSI · MACD
# Output  : 0–100 score + letter grade + full breakdown
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close[-(period + 1):])
    gains  = np.where(deltas > 0,  deltas, 0.0).mean()
    losses = np.where(deltas < 0, -deltas, 0.0).mean()
    if losses == 0:
        return 100.0
    return float(100 - 100 / (1 + gains / losses))


def _compute_macd(close: np.ndarray):
    def ema(arr, span):
        k, out = 2 / (span + 1), [arr[0]]
        for v in arr[1:]:
            out.append(v * k + out[-1] * (1 - k))
        return np.array(out)
    macd   = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    return float(macd[-1]), float(signal[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high  = df["High"].values[-(period + 1):]
    low   = df["Low"].values[-(period + 1):]
    close = df["Close"].values[-(period + 1):]
    tr    = np.maximum(high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:]  - close[:-1])))
    return float(tr.mean())


# ── Sub-Scores (each 0–100) ───────────────────────────────────────────────────

def _trend_score(price, ma20, ma50, ma200):
    score, notes = 0.0, []
    if price > ma20:  score += 34; notes.append("Price > MA20 ✅")
    if ma20  > ma50:  score += 33; notes.append("MA20 > MA50 ✅")
    if ma50  > ma200: score += 33; notes.append("MA50 > MA200 (Golden Zone) ✅")
    return score, notes


def _momentum_score(close):
    if len(close) < 22:
        return 50.0, []
    r1  = (close[-1] - close[-2])  / close[-2]  * 100
    r5  = (close[-1] - close[-6])  / close[-6]  * 100
    r20 = (close[-1] - close[-21]) / close[-21] * 100
    composite = 0.5 * r1 + 0.3 * r5 + 0.2 * r20
    score = float(np.clip(50 + composite * 4, 0, 100))
    notes = []
    if r1  > 0: notes.append(f"1d  +{r1:.2f}% 📈")
    if r5  > 0: notes.append(f"5d  +{r5:.2f}% 📈")
    if r20 > 0: notes.append(f"20d +{r20:.2f}% 📈")
    return score, notes


def _breakout_score(df, price):
    notes = []
    high20 = float(df["High"].rolling(20).max().iloc[-2])
    high52 = float(df["High"].rolling(min(252, len(df))).max().iloc[-1])
    score  = 0.0
    if price > high20:        score += 60; notes.append("20-day breakout 🚀")
    if price > 0.95 * high52: score += 40; notes.append("Near 52-week high 💪")
    return score, notes


def _rsi_score(rsi):
    if   rsi < 30: return 90.0, [f"RSI {rsi:.1f} → Oversold 🟢"]
    elif rsi < 45: return 70.0, [f"RSI {rsi:.1f} → Mild oversold"]
    elif rsi < 60: return 55.0, [f"RSI {rsi:.1f} → Neutral"]
    elif rsi < 70: return 40.0, [f"RSI {rsi:.1f} → Approaching overbought"]
    else:          return 15.0, [f"RSI {rsi:.1f} → Overbought 🔴"]


def _macd_score(macd, signal):
    hist = macd - signal
    if   macd > signal and hist > 0: return 85.0, ["MACD bullish crossover ✅"]
    elif macd > signal:              return 60.0, ["MACD above signal (weak bull)"]
    elif macd < signal and hist < 0: return 15.0, ["MACD bearish crossover ❌"]
    else:                            return 35.0, ["MACD below signal (weak bear)"]


def _volume_score(df):
    avg_vol  = float(df["Volume"].iloc[-20:].mean())
    last_vol = float(df["Volume"].iloc[-1])
    ratio    = last_vol / avg_vol if avg_vol > 0 else 1.0
    score    = float(np.clip(50 + (ratio - 1) * 35, 0, 100))
    label    = "🔥 Surge" if ratio > 1.5 else "⚠️ Low" if ratio < 0.7 else "Normal"
    return score, [f"Volume {ratio:.1f}x avg → {label}"]


def _volatility_score(close, atr):
    price   = close[-1]
    atr_pct = (atr / price) * 100 if price > 0 else 2.0
    score   = float(np.clip(100 - atr_pct * 8, 0, 100))
    label   = "Stable 🟢" if atr_pct < 2 else "Volatile 🔴"
    return score, [f"ATR {atr_pct:.2f}% of price → {label}"]


# ── Public API ────────────────────────────────────────────────────────────────

def stock_score(df: pd.DataFrame) -> int:
    """Returns a 0–100 composite AI score. Needs ≥60 rows."""
    if df is None or len(df) < 60:
        return 0

    df    = df.copy()
    close = df["Close"].dropna().values.flatten().astype(float)
    price = close[-1]
    s     = pd.Series(close)
    ma20  = float(s.rolling(20).mean().iloc[-1])
    ma50  = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(min(200, len(close))).mean().iloc[-1])
    rsi   = _compute_rsi(close)
    macd, sig = _compute_macd(close)
    atr   = _atr(df)

    weighted = [
        (_trend_score(price, ma20, ma50, ma200)[0], 0.25),
        (_momentum_score(close)[0],                  0.20),
        (_rsi_score(rsi)[0],                         0.18),
        (_macd_score(macd, sig)[0],                  0.17),
        (_volume_score(df)[0],                       0.10),
        (_breakout_score(df, price)[0],               0.05),
        (_volatility_score(close, atr)[0],            0.05),
    ]
    return int(np.clip(round(sum(v * w for v, w in weighted)), 0, 100))


def detailed_score_report(df: pd.DataFrame) -> dict:
    """Full breakdown with sub-scores, notes, indicators and grade."""
    if df is None or len(df) < 60:
        return {"error": "Need at least 60 days of data"}

    df    = df.copy()
    close = df["Close"].dropna().values.flatten().astype(float)
    price = close[-1]
    s     = pd.Series(close)
    ma20  = float(s.rolling(20).mean().iloc[-1])
    ma50  = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(min(200, len(close))).mean().iloc[-1])
    rsi   = _compute_rsi(close)
    macd, sig = _compute_macd(close)
    atr   = _atr(df)

    parts = {
        "Trend      (25%)": _trend_score(price, ma20, ma50, ma200),
        "Momentum   (20%)": _momentum_score(close),
        "RSI        (18%)": _rsi_score(rsi),
        "MACD       (17%)": _macd_score(macd, sig),
        "Volume     (10%)": _volume_score(df),
        "Breakout    (5%)": _breakout_score(df, price),
        "Volatility  (5%)": _volatility_score(close, atr),
    }

    total = stock_score(df)
    grade = ("A+ 🏆" if total >= 85 else "A  🟢" if total >= 70 else
             "B  🟡" if total >= 55 else "C  🟠" if total >= 40 else "D  🔴")

    return {
        "total_score": total,
        "grade":       grade,
        "breakdown":   {k: {"score": round(sc, 1), "notes": n}
                        for k, (sc, n) in parts.items()},
        "indicators":  {
            "price": round(price, 2), "MA20": round(ma20, 2),
            "MA50":  round(ma50, 2),  "MA200": round(ma200, 2),
            "RSI":   round(rsi, 2),   "MACD":  round(macd, 4),
            "ATR":   round(atr, 2),
        }
    }
