# technical/volume_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# Volume Analysis Suite — Professional Trading Feature
# Includes: On-Balance Volume (OBV), Volume Spikes, VWMA, Volume Profile
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV)
    - Adds volume on up days, subtracts on down days
    - Shows cumulative money flow
    """
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df["Volume"].iloc[0]
    
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df["Volume"].iloc[i]
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df["Volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def compute_obv_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """OBV smoothed with EMA for signal line."""
    obv = compute_obv(df)
    return obv.ewm(span=period, adjust=False).mean()


def compute_vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Volume Weighted Moving Average
    - More responsive than simple MA when volume confirms price
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()


def compute_money_flow(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Chaikin Money Flow (CMF)
    - Measures money flow volume over N periods
    - Values > 0: buying pressure
    - Values < 0: selling pressure
    """
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    mf_multiplier = mf_multiplier.fillna(0)
    mf_volume     = mf_multiplier * df["Volume"]
    cmf           = mf_volume.rolling(period).sum() / df["Volume"].rolling(period).sum()
    
    # Also compute positive/negative money flow
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pos_flow = (typical * df["Volume"] * (typical > typical.shift(1))).rolling(period).sum()
    neg_flow = (typical * df["Volume"] * (typical < typical.shift(1))).rolling(period).sum()
    money_ratio = pos_flow / (neg_flow + 1e-9)
    
    return cmf, money_ratio, pos_flow / (pos_flow + neg_flow + 1e-9)


def detect_volume_spike(df: pd.DataFrame, lookback: int = 20, threshold: float = 2.0) -> List[Dict]:
    """
    Detect volume spikes - unusually high volume compared to average.
    Returns list of spike events with details.
    """
    df = df.copy()
    df["vol_avg"]   = df["Volume"].rolling(lookback).mean()
    df["vol_std"]   = df["Volume"].rolling(lookback).std()
    df["vol_zscore"] = (df["Volume"] - df["vol_avg"]) / (df["vol_std"] + 1e-9)
    df["vol_ratio"] = df["Volume"] / (df["vol_avg"] + 1e-9)
    
    spikes = []
    for i in range(lookback, len(df)):
        if df["vol_ratio"].iloc[i] >= threshold:
            price_change = ((df["Close"].iloc[i] - df["Close"].iloc[i-1]) / 
                          df["Close"].iloc[i-1]) * 100
            
            spikes.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "volume": int(df["Volume"].iloc[i]),
                "avg_volume": int(df["vol_avg"].iloc[i]),
                "ratio": round(df["vol_ratio"].iloc[i], 2),
                "zscore": round(df["vol_zscore"].iloc[i], 2),
                "price_change": round(price_change, 2),
                "price_action": "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT",
                "strength": "EXTREME" if df["vol_ratio"].iloc[i] > 4 else "STRONG" if df["vol_ratio"].iloc[i] > 3 else "MODERATE",
            })
    
    return spikes


def volume_analysis(df: pd.DataFrame) -> Dict:
    """
    Comprehensive volume analysis combining OBV, CMF, and spike detection.
    """
    if df is None or len(df) < 30:
        return {"error": "Need at least 30 bars of data"}
    
    df = df.copy()
    
    # Calculate indicators
    obv        = compute_obv(df)
    obv_ema    = compute_obv_ema(df)
    vwma       = compute_vwma(df)
    cmf, mfr, mfi_smooth = compute_money_flow(df)
    
    # Current values
    current_price  = float(df["Close"].iloc[-1])
    current_volume = int(df["Volume"].iloc[-1])
    current_obv    = float(obv.iloc[-1])
    current_cmf    = float(cmf.iloc[-1])
    
    # Volume statistics
    vol_5d_avg  = float(df["Volume"].iloc[-5:].mean())
    vol_20d_avg = float(df["Volume"].iloc[-20:].mean())
    vol_ratio   = current_volume / vol_20d_avg if vol_20d_avg > 0 else 1.0
    
    # OBV trend
    obv_10d_ago = float(obv.iloc[-11]) if len(obv) > 10 else float(obv.iloc[0])
    obv_change  = ((current_obv - obv_10d_ago) / abs(obv_10d_ago) * 100) if obv_10d_ago != 0 else 0
    
    # Detect recent spikes
    recent_spikes = detect_volume_spike(df.tail(50), lookback=20, threshold=2.0)
    latest_spike  = recent_spikes[-1] if recent_spikes else None
    
    # Price-volume divergence
    price_10d_return = ((current_price - float(df["Close"].iloc[-11])) / 
                       float(df["Close"].iloc[-11]) * 100) if len(df) > 10 else 0
    
    obv_10d_return = obv_change
    
    # Volume-price alignment
    if price_10d_return > 0 and obv_10d_return > 0:
        alignment = "CONFIRMED_UP"
        alignment_score = 80
    elif price_10d_return < 0 and obv_10d_return < 0:
        alignment = "CONFIRMED_DOWN"
        alignment_score = 80
    elif price_10d_return > 0 and obv_10d_return < 0:
        alignment = "DIVERGENCE_DOWN"
        alignment_score = 30
    elif price_10d_return < 0 and obv_10d_return > 0:
        alignment = "DIVERGENCE_UP"
        alignment_score = 70  # Often bullish divergence
    else:
        alignment = "NEUTRAL"
        alignment_score = 50
    
    # CMF signal
    if current_cmf > 0.1:
        cmf_signal = "STRONG_BUYING_PRESSURE"
    elif current_cmf > 0:
        cmf_signal = "MODERATE_BUYING_PRESSURE"
    elif current_cmf < -0.1:
        cmf_signal = "STRONG_SELLING_PRESSURE"
    elif current_cmf < 0:
        cmf_signal = "MODERATE_SELLING_PRESSURE"
    else:
        cmf_signal = "NEUTRAL"
    
    # Overall signal
    signal_score = 50 + alignment_score - 50
    if abs(vol_ratio - 1) > 0.5:
        signal_score += 10 if vol_ratio > 1 else -10
    if latest_spike:
        signal_score += 5
    
    signal_score = max(0, min(100, signal_score))
    
    if signal_score >= 70:
        signal = "STRONG_VOLUME_CONFIRMATION"
    elif signal_score >= 55:
        signal = "BULLISH_VOLUME"
    elif signal_score >= 45:
        signal = "NEUTRAL"
    elif signal_score >= 30:
        signal = "BEARISH_VOLUME"
    else:
        signal = "WEAK_VOLUME"
    
    return {
        "obv": round(current_obv, 2),
        "obv_change_10d": round(obv_change, 2),
        "obv_trend": "RISING" if obv_change > 0 else "FALLING",
        "cmf": round(current_cmf, 4),
        "cmf_signal": cmf_signal,
        "volume_ratio_20d": round(vol_ratio, 2),
        "volume_5d_avg": int(vol_5d_avg),
        "volume_20d_avg": int(vol_20d_avg),
        "volume_vs_avg": "ABOVE" if vol_ratio > 1 else "BELOW",
        "alignment": alignment,
        "alignment_score": alignment_score,
        "price_change_10d": round(price_10d_return, 2),
        "signal": signal,
        "signal_score": signal_score,
        "recent_spikes": recent_spikes[-5:] if recent_spikes else [],
        "latest_spike": latest_spike,
        "vwma_20": round(float(vwma.iloc[-1]), 2) if not pd.isna(vwma.iloc[-1]) else None,
    }


def volume_report(df: pd.DataFrame) -> str:
    """Human-readable volume analysis report."""
    result = volume_analysis(df)
    
    if "error" in result:
        return result["error"]
    
    lines = []
    lines.append("=" * 50)
    lines.append("VOLUME ANALYSIS")
    lines.append("=" * 50)
    
    lines.append(f"\nOBV (On-Balance Volume):")
    lines.append(f"  Current: {result['obv']:,.0f}")
    lines.append(f"  10-day Change: {result['obv_change_10d']:+.1f}%")
    lines.append(f"  Trend: {result['obv_trend']}")
    
    lines.append(f"\nMoney Flow (CMF):")
    lines.append(f"  CMF: {result['cmf']}")
    lines.append(f"  Signal: {result['cmf_signal']}")
    
    lines.append(f"\nVolume Comparison:")
    lines.append(f"  Current vs 20d Avg: {result['volume_ratio_20d']}x")
    lines.append(f"  Position: {result['volume_vs_avg']} average")
    
    lines.append(f"\nPrice-Volume Alignment:")
    lines.append(f"  10d Price Change: {result['price_change_10d']:+.2f}%")
    lines.append(f"  Status: {result['alignment']}")
    
    if result["latest_spike"]:
        ls = result["latest_spike"]
        lines.append(f"\nLatest Volume Spike ({ls['date']}):")
        lines.append(f"  Volume: {ls['ratio']}x average ({ls['strength']})")
        lines.append(f"  Price Action: {ls['price_action']} ({ls['price_change']:+.2f}%)")
    
    lines.append(f"\nSignal: {result['signal']} (Score: {result['signal_score']})")
    
    return "\n".join(lines)
