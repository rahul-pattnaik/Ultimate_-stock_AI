# ai/ranking_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Master Stock Scoring Engine — Enhanced with Professional Trading Features
# Combines: Trend · Momentum · Breakout · Volatility · Volume · RSI · MACD
# Plus: RSI Divergence · Candlestick · Volume Analysis · Confluence
# Output  : 0–100 score + letter grade + full breakdown
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


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


def _obv(df: pd.DataFrame) -> float:
    """On-Balance Volume."""
    obv = 0
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv += df["Volume"].iloc[i]
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv -= df["Volume"].iloc[i]
    return float(obv)


def _find_swing_extremes(close: pd.Series, lookback: int = 14) -> Tuple[float, float, float, float]:
    """Find recent swing high/low and 52-week high/low."""
    recent_high = float(close.iloc[-lookback:].max())
    recent_low  = float(close.iloc[-lookback:].min())
    all_high    = float(close.iloc[-252:].max()) if len(close) >= 252 else recent_high
    all_low     = float(close.iloc[-252:].min()) if len(close) >= 252 else recent_low
    return recent_high, recent_low, all_high, all_low


def _detect_rsi_divergence(df: pd.DataFrame) -> Tuple[str, float]:
    """Detect RSI divergence (simplified)."""
    close = df["Close"]
    rsi_series = pd.Series(index=close.index)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_series = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    # Find swing points
    lookback = 10
    price_swing_high = close.iloc[-lookback:].idxmax()
    price_swing_low  = close.iloc[-lookback:].idxmin()
    
    rsi_at_price_high = rsi_series.loc[price_swing_high]
    rsi_at_price_low  = rsi_series.loc[price_swing_low]
    
    # Regular divergence
    if close.loc[price_swing_high] > close.iloc[-lookback-1] and rsi_at_price_high < 50:
        return "BEARISH_DIVERGENCE", 0.7
    if close.loc[price_swing_low] < close.iloc[-lookback-1] and rsi_at_price_low > 50:
        return "BULLISH_DIVERGENCE", 0.8
    
    return "NONE", 0


def _detect_candlestick_signal(df: pd.DataFrame) -> Tuple[str, float]:
    """Detect key candlestick patterns."""
    if len(df) < 3:
        return "UNKNOWN", 0
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    body     = abs(curr["Close"] - curr["Open"])
    range_   = curr["High"] - curr["Low"]
    body_pct = body / range_ if range_ > 0 else 0
    
    # Doji
    if body_pct < 0.1:
        return "DOJI", 0.5
    
    # Hammer
    lower_wick = curr["Open"] - curr["Low"] if curr["Open"] > curr["Close"] else curr["Close"] - curr["Low"]
    if lower_wick > 2 * body and body_pct < 0.35:
        return "HAMMER_BULLISH", 0.8
    
    # Shooting Star
    upper_wick = curr["High"] - curr["Open"] if curr["Open"] > curr["Close"] else curr["High"] - curr["Close"]
    if upper_wick > 2 * body and body_pct < 0.35:
        return "SHOOTING_STAR_BEARISH", 0.8
    
    # Bullish Engulfing
    if curr["Close"] > curr["Open"] and prev["Close"] < prev["Open"]:
        if curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"]:
            return "BULLISH_ENGULFING", 0.9
    
    # Bearish Engulfing
    if curr["Close"] < curr["Open"] and prev["Close"] > prev["Open"]:
        if curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]:
            return "BEARISH_ENGULFING", 0.9
    
    return "NONE", 0


def _volume_price_correlation(df: pd.DataFrame) -> Tuple[float, str]:
    """Check if volume confirms price movement."""
    close  = df["Close"]
    volume = df["Volume"]
    
    # Volume vs 20-day average
    vol_avg  = volume.iloc[-20:].mean()
    vol_curr = volume.iloc[-1]
    vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1
    
    # Price momentum
    price_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
    
    if vol_ratio > 1.5 and price_change > 0:
        return 85.0, "STRONG_CONFIRMATION"
    elif vol_ratio > 1.5 and price_change < 0:
        return 15.0, "DISTRIBUTION"
    elif vol_ratio < 0.7 and abs(price_change) < 2:
        return 50.0, "LOW_ACTIVITY"
    elif price_change > 0:
        return 65.0, "WEAK_CONFIRMATION"
    else:
        return 35.0, "UNCERTAINTY"


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


def _divergence_score(df):
    """Score RSI divergence (new)."""
    divergence_type, strength = _detect_rsi_divergence(df)
    if divergence_type == "BULLISH_DIVERGENCE":
        return 85.0, ["Bullish Divergence detected ✅"]
    elif divergence_type == "BEARISH_DIVERGENCE":
        return 15.0, ["Bearish Divergence detected 🔴"]
    return 50.0, ["No divergence"]


def _candlestick_score(df):
    """Score candlestick patterns (new)."""
    pattern, strength = _detect_candlestick_signal(df)
    if "BULLISH" in pattern:
        return 70 + strength * 20, [f"{pattern} pattern detected 🟢"]
    elif "BEARISH" in pattern:
        return 30 - strength * 20, [f"{pattern} pattern detected 🔴"]
    return 50.0, ["No significant pattern"]


def _volume_confirmation_score(df):
    """Score volume-price confirmation (new)."""
    score, label = _volume_price_correlation(df)
    return score, [f"Volume-Price: {label}"]


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
        (_trend_score(price, ma20, ma50, ma200)[0], 0.20),
        (_momentum_score(close)[0],                  0.18),
        (_rsi_score(rsi)[0],                         0.15),
        (_macd_score(macd, sig)[0],                   0.15),
        (_volume_score(df)[0],                        0.08),
        (_breakout_score(df, price)[0],               0.08),
        (_volatility_score(close, atr)[0],            0.05),
        (_divergence_score(df)[0],                    0.05),
        (_candlestick_score(df)[0],                   0.03),
        (_volume_confirmation_score(df)[0],          0.03),
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
        "Trend          (20%)": _trend_score(price, ma20, ma50, ma200),
        "Momentum       (18%)": _momentum_score(close),
        "RSI            (15%)": _rsi_score(rsi),
        "MACD           (15%)": _macd_score(macd, sig),
        "Volume         (8%)":  _volume_score(df),
        "Breakout        (8%)": _breakout_score(df, price),
        "Volatility      (5%)": _volatility_score(close, atr),
        "Divergence      (5%)": _divergence_score(df),
        "Candlestick     (3%)": _candlestick_score(df),
        "Vol Confirmation(3%)": _volume_confirmation_score(df),
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
        },
        "risk_reward": _calculate_risk_reward(df, price, atr),
    }


def _calculate_risk_reward(df: pd.DataFrame, price: float, atr: float) -> Dict:
    """Calculate risk/reward ratio based on support and resistance."""
    recent_high = float(df["High"].iloc[-20:].max())
    recent_low  = float(df["Low"].iloc[-20:].min())
    
    upside   = recent_high - price
    downside = price - recent_low
    
    if downside > 0:
        rr_ratio = upside / downside
    else:
        rr_ratio = 0
    
    if rr_ratio >= 3:
        recommendation = "EXCELLENT R/R ✅✅"
    elif rr_ratio >= 2:
        recommendation = "GOOD R/R ✅"
    elif rr_ratio >= 1:
        recommendation = "ACCEPTABLE R/R"
    else:
        recommendation = "POOR R/R ❌"
    
    return {
        "support": round(recent_low, 2),
        "resistance": round(recent_high, 2),
        "upside": round(upside, 2),
        "downside": round(downside, 2),
        "risk_reward_ratio": round(rr_ratio, 2),
        "recommendation": recommendation,
    }


def professional_score_report(df: pd.DataFrame) -> str:
    """Generate comprehensive professional trading report."""
    result = detailed_score_report(df)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = []
    lines.append("=" * 70)
    lines.append("PROFESSIONAL TRADING SCORE REPORT")
    lines.append("=" * 70)
    
    lines.append(f"\nTOTAL SCORE: {result['total_score']}/100 — Grade: {result['grade']}")
    
    lines.append("\n" + "-" * 70)
    lines.append("COMPONENT BREAKDOWN:")
    lines.append("-" * 70)
    
    for component, data in result["breakdown"].items():
        bar_len = int(data["score"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"{component:<25} {data['score']:5.1f}/100 [{bar}]")
        for note in data["notes"][:2]:
            lines.append(f"  → {note}")
    
    if "risk_reward" in result:
        rr = result["risk_reward"]
        lines.append("\n" + "-" * 70)
        lines.append("RISK/REWARD ANALYSIS:")
        lines.append("-" * 70)
        lines.append(f"  Support:    {rr['support']}")
        lines.append(f"  Resistance: {rr['resistance']}")
        lines.append(f"  Upside:     +{rr['upside']} ({rr['upside']/rr['downside']*100:.1f}%)" if rr['downside'] > 0 else f"  Upside: {rr['upside']}")
        lines.append(f"  Downside:   -{rr['downside']}")
        lines.append(f"  R/R Ratio:  {rr['risk_reward_ratio']}:1")
        lines.append(f"  Verdict:    {rr['recommendation']}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)
