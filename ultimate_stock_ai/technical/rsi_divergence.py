# technical/rsi_divergence.py
# ─────────────────────────────────────────────────────────────────────────────
# RSI Divergence Detection — Professional Trading Feature
# Detects: Regular Bullish/Bearish Divergence, Hidden Bullish/Bearish Divergence
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using standard Wilder smoothing."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _find_swing_points(series: pd.Series, lookback: int = 20) -> Tuple[List[int], List[int]]:
    """
    Find local swing highs and lows using simple N-bar extreme detection.
    Returns arrays of indices for highs and lows.
    """
    highs, lows = [], []
    for i in range(lookback, len(series) - lookback):
        if series.iloc[i] == series.iloc[i-lookback:i+lookback+1].max():
            highs.append(i)
        if series.iloc[i] == series.iloc[i-lookback:i+lookback+1].min():
            lows.append(i)
    return highs, lows


def _detect_divergence(price_highs: List[int], price_lows: List[int],
                       rsi_highs: List[int], rsi_lows: List[int],
                       price: pd.Series, rsi: pd.Series,
                       lookback: int = 8) -> List[Dict]:
    """
    Detect all types of divergence between price and RSI.
    Returns list of divergence signals with type, direction, strength.
    """
    signals = []
    
    # Regular Bearish Divergence: Price higher high, RSI lower high
    for ph in price_highs[-lookback:]:
        for rh in rsi_highs:
            if 0 < ph - rh <= lookback * 2 and ph > rh:
                if price.iloc[ph] > price.iloc[rh] and rsi.iloc[ph] < rsi.iloc[rh]:
                    price_change = ((price.iloc[ph] - price.iloc[rh]) / price.iloc[rh]) * 100
                    rsi_change   = rsi.iloc[ph] - rsi.iloc[rh]
                    signals.append({
                        "type": "REGULAR_BEARISH",
                        "direction": "SELL",
                        "price_idx": ph,
                        "rsi_idx": rh,
                        "price_change": price_change,
                        "rsi_change": rsi_change,
                        "strength": min(abs(rsi_change) / 10, 1.0),
                        "description": f"Price made higher high (+{price_change:.1f}%) but RSI lower ({rsi_change:.1f})"
                    })
    
    # Regular Bullish Divergence: Price lower low, RSI higher low
    for pl in price_lows[-lookback:]:
        for rl in rsi_lows:
            if 0 < pl - rl <= lookback * 2 and pl > rl:
                if price.iloc[pl] < price.iloc[rl] and rsi.iloc[pl] > rsi.iloc[rl]:
                    price_change = ((price.iloc[rl] - price.iloc[pl]) / price.iloc[pl]) * 100
                    rsi_change   = rsi.iloc[pl] - rsi.iloc[rl]
                    signals.append({
                        "type": "REGULAR_BULLISH",
                        "direction": "BUY",
                        "price_idx": pl,
                        "rsi_idx": rl,
                        "price_change": price_change,
                        "rsi_change": rsi_change,
                        "strength": min(abs(rsi_change) / 10, 1.0),
                        "description": f"Price made lower low but RSI higher (+{rsi_change:.1f})"
                    })
    
    # Hidden Bearish Divergence: Price lower high, RSI higher high
    for ph in price_highs[-lookback:]:
        for rh in rsi_highs:
            if 0 < rh - ph <= lookback * 2 and rh > ph:
                if price.iloc[rh] < price.iloc[ph] and rsi.iloc[rh] > rsi.iloc[ph]:
                    signals.append({
                        "type": "HIDDEN_BEARISH",
                        "direction": "SELL",
                        "price_idx": rh,
                        "rsi_idx": ph,
                        "strength": 0.7,
                        "description": "Hidden bearish: price lower high, RSI higher high"
                    })
    
    # Hidden Bullish Divergence: Price higher low, RSI lower low
    for pl in price_lows[-lookback:]:
        for rl in rsi_lows:
            if 0 < rl - pl <= lookback * 2 and rl > pl:
                if price.iloc[rl] > price.iloc[pl] and rsi.iloc[rl] < rsi.iloc[pl]:
                    signals.append({
                        "type": "HIDDEN_BULLISH",
                        "direction": "BUY",
                        "price_idx": rl,
                        "rsi_idx": pl,
                        "strength": 0.7,
                        "description": "Hidden bullish: price higher low, RSI lower low"
                    })
    
    return signals


def detect_rsi_divergence(df: pd.DataFrame, period: int = 14) -> Dict:
    """
    Main function to detect all RSI divergence types.
    
    Returns:
        dict with:
        - divergences: list of all detected divergences
        - latest_divergence: most recent divergence or None
        - current_rsi: current RSI value
        - signal: overall signal (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
        - signal_score: 0-100 score
    """
    if df is None or len(df) < 50:
        return {"error": "Need at least 50 bars of data"}
    
    close = df["Close"]
    rsi   = _compute_rsi(close, period)
    
    price_highs, price_lows = _find_swing_points(close, lookback=15)
    rsi_highs,   rsi_lows   = _find_swing_points(rsi,   lookback=15)
    
    if not price_highs or not price_lows or not rsi_highs or not rsi_lows:
        return {"error": "Not enough swing points detected"}
    
    divergences = _detect_divergence(
        price_highs, price_lows, rsi_highs, rsi_lows,
        close, rsi, lookback=6
    )
    
    # Current RSI status
    current_rsi = float(rsi.iloc[-1])
    current_price = float(close.iloc[-1])
    
    # Determine signal
    signal = "NEUTRAL"
    signal_score = 50
    
    if divergences:
        latest = divergences[-1]
        
        if latest["type"] in ["REGULAR_BULLISH", "HIDDEN_BULLISH"]:
            if latest["type"] == "REGULAR_BULLISH":
                signal = "STRONG_BUY" if latest["strength"] > 0.8 else "BUY"
                signal_score = int(60 + latest["strength"] * 40)
            else:
                signal = "BUY"
                signal_score = 60
        elif latest["type"] in ["REGULAR_BEARISH", "HIDDEN_BEARISH"]:
            if latest["type"] == "REGULAR_BEARISH":
                signal = "STRONG_SELL" if latest["strength"] > 0.8 else "SELL"
                signal_score = int(40 - latest["strength"] * 40)
            else:
                signal = "SELL"
                signal_score = 40
    
    # Overbought/Oversold context
    if current_rsi > 70:
        context = "OVERBOUGHT"
    elif current_rsi < 30:
        context = "OVERSOLD"
    else:
        context = "NEUTRAL"
    
    return {
        "divergences": divergences,
        "latest_divergence": divergences[-1] if divergences else None,
        "current_rsi": round(current_rsi, 2),
        "current_price": round(current_price, 2),
        "context": context,
        "signal": signal,
        "signal_score": signal_score,
        "price_highs_count": len(price_highs),
        "price_lows_count": len(price_lows),
        "rsi_highs_count": len(rsi_highs),
        "rsi_lows_count": len(rsi_lows),
    }


def rsi_divergence_report(df: pd.DataFrame, period: int = 14) -> Dict:
    """Human-readable RSI divergence report."""
    result = detect_rsi_divergence(df, period)
    
    if "error" in result:
        return result
    
    lines = []
    latest = result["latest_divergence"]
    
    if latest:
        lines.append(f"Type: {latest['type']}")
        lines.append(f"Direction: {latest['direction']}")
        lines.append(f"Strength: {latest['strength']:.0%}")
        lines.append(f"Description: {latest['description']}")
    else:
        lines.append("No divergence detected")
    
    lines.append(f"Current RSI: {result['current_rsi']}")
    lines.append(f"Context: {result['context']}")
    lines.append(f"Signal: {result['signal']}")
    
    return {
        "report": "\n".join(lines),
        "signal": result["signal"],
        "signal_score": result["signal_score"],
        "current_rsi": result["current_rsi"],
        "context": result["context"],
        "latest_divergence": latest,
    }
