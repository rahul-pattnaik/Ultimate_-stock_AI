# technical/multiple_timeframe.py
# ─────────────────────────────────────────────────────────────────────────────
# Multiple Timeframe Analysis (MTFA) — Professional Trading Feature
# Combines: Intraday (5m/15m) + Swing (Daily/Weekly) + Long-term (Weekly/Monthly)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_fetcher import get_stock_data


def _compute_ma_trend(close: pd.Series, period: int) -> str:
    """Determine trend from MA position."""
    ma = close.rolling(period).mean()
    if close.iloc[-1] > ma.iloc[-1]:
        return "ABOVE"
    elif close.iloc[-1] < ma.iloc[-1]:
        return "BELOW"
    return "AT"


def _compute_momentum(close: pd.Series, period: int = 14) -> float:
    """Compute momentum as percentage change."""
    if len(close) < period + 1:
        return 0.0
    return float((close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1] * 100)


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Compute RSI value."""
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    return float((100 - (100 / (1 + rs))).iloc[-1])


def _compute_macd(close: pd.Series) -> Tuple[float, float, float]:
    """Compute MACD line, signal, and histogram."""
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def analyze_timeframe(df: pd.DataFrame, timeframe: str) -> Dict:
    """
    Analyze a single timeframe and return key metrics.
    """
    if df is None or len(df) < 10:
        return {"error": f"Insufficient data for {timeframe}"}
    
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    
    current_price = float(close.iloc[-1])
    
    # Moving averages
    ma20  = float(close.rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
    ma50  = float(close.rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else None
    
    # Trend analysis
    ema9  = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    
    # Determine short-term trend
    if close.iloc[-1] > ema9 > ema21:
        short_trend = "STRONG_UP"
    elif close.iloc[-1] > ema9:
        short_trend = "UP"
    elif close.iloc[-1] < ema9 < ema21:
        short_trend = "STRONG_DOWN"
    elif close.iloc[-1] < ema9:
        short_trend = "DOWN"
    else:
        short_trend = "NEUTRAL"
    
    # Golden/Death Cross for daily+ timeframes
    if ma20 and ma50:
        ma20_prev = float(close.rolling(20).mean().iloc[-2])
        ma50_prev = float(close.rolling(50).mean().iloc[-2])
        if ma20 > ma50 and ma20_prev <= ma50_prev:
            cross = "GOLDEN_CROSS"
        elif ma20 < ma50 and ma20_prev >= ma50_prev:
            cross = "DEATH_CROSS"
        else:
            cross = "NONE"
    else:
        cross = "N/A"
    
    # RSI
    rsi = _compute_rsi(close)
    if rsi > 70:
        rsi_status = "OVERBOUGHT"
    elif rsi < 30:
        rsi_status = "OVERSOLD"
    else:
        rsi_status = "NEUTRAL"
    
    # MACD
    macd, signal, hist = _compute_macd(close)
    if hist > 0:
        macd_status = "BULLISH"
    elif hist < 0:
        macd_status = "BEARISH"
    else:
        macd_status = "NEUTRAL"
    
    # Support/Resistance
    recent_high = float(high.iloc[-20:].max()) if len(df) >= 20 else float(high.max())
    recent_low  = float(low.iloc[-20:].min()) if len(df) >= 20 else float(low.min())
    
    # Momentum
    momentum_5  = _compute_momentum(close, 5)
    momentum_10 = _compute_momentum(close, 10)
    
    # Volatility (ATR-based)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    atr_pct = atr / current_price * 100 if current_price > 0 else 0
    
    # Trend score (0-100)
    trend_score = 50
    if current_price > ema9:   trend_score += 10
    if ema9 > ema21:           trend_score += 10
    if ma20 and current_price > ma20: trend_score += 5
    if ma50 and current_price > ma50: trend_score += 5
    if ma200 and current_price > ma200: trend_score += 5
    if hist > 0:               trend_score += 10
    if rsi < 70 and rsi > 30:  trend_score += 5
    
    return {
        "timeframe": timeframe,
        "current_price": round(current_price, 2),
        "trend": short_trend,
        "trend_score": trend_score,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma50": round(ma50, 2) if ma50 else None,
        "ma200": round(ma200, 2) if ma200 else None,
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "ma_position": {
            "vs_ma20": _compute_ma_trend(close, 20) if ma20 else "N/A",
            "vs_ma50": _compute_ma_trend(close, 50) if ma50 else "N/A",
            "vs_ma200": _compute_ma_trend(close, 200) if ma200 else "N/A",
        },
        "cross": cross,
        "rsi": round(rsi, 2),
        "rsi_status": rsi_status,
        "macd": round(macd, 4),
        "macd_signal": round(signal, 4),
        "macd_histogram": round(hist, 4),
        "macd_status": macd_status,
        "support": round(recent_low, 2),
        "resistance": round(recent_high, 2),
        "momentum_5": round(momentum_5, 2),
        "momentum_10": round(momentum_10, 2),
        "atr_pct": round(atr_pct, 2),
        "bars_analyzed": len(df),
    }


def multi_timeframe_analysis(symbol: str) -> Dict:
    """
    Comprehensive multi-timeframe analysis for a symbol.
    Automatically fetches data for different timeframes.
    """
    # Define timeframe configurations
    timeframes = {
        "INTRADAY_5M":  {"period": "5d",  "interval": "5m"},
        "INTRADAY_15M": {"period": "5d",  "interval": "15m"},
        "SWING_DAILY":  {"period": "3mo", "interval": "1d"},
        "SWING_WEEKLY": {"period": "6mo", "interval": "1wk"},
        "LONG_WEEKLY":  {"period": "1y",  "interval": "1wk"},
        "LONG_MONTHLY": {"period": "5y",  "interval": "1mo"},
    }
    
    results = {}
    errors  = []
    
    for tf_name, tf_config in timeframes.items():
        try:
            df = get_stock_data(
                symbol, 
                period=tf_config["period"], 
                interval=tf_config["interval"],
                use_cache=True
            )
            if df is not None and len(df) >= 10:
                results[tf_name] = analyze_timeframe(df, tf_name)
            else:
                errors.append(f"{tf_name}: Insufficient data")
        except Exception as e:
            errors.append(f"{tf_name}: {str(e)}")
    
    # Combine signals
    if not results:
        return {"error": "Could not fetch data for any timeframe", "errors": errors}
    
    # Extract trends
    trends = {k: v.get("trend", "NEUTRAL") for k, v in results.items() if "trend" in v}
    trend_scores = {k: v.get("trend_score", 50) for k, v in results.items()}
    
    # Consensus trend
    up_count   = sum(1 for t in trends.values() if "UP" in t)
    down_count = sum(1 for t in trends.values() if "DOWN" in t)
    
    if up_count > down_count:
        consensus_trend = "BULLISH"
        consensus_score = 50 + (up_count - down_count) * 10
    elif down_count > up_count:
        consensus_trend = "BEARISH"
        consensus_score = 50 - (down_count - up_count) * 10
    else:
        consensus_trend = "NEUTRAL"
        consensus_score = 50
    
    # Confluence check
    confluence_count = 0
    if "SWING_DAILY" in results and "SWING_WEEKLY" in results:
        if results["SWING_DAILY"].get("trend") == results["SWING_WEEKLY"].get("trend"):
            confluence_count += 1
        if results["SWING_DAILY"].get("ma_position", {}).get("vs_ma50") == \
           results["SWING_WEEKLY"].get("ma_position", {}).get("vs_ma50"):
            confluence_count += 1
    
    # Final signal
    final_score = int(np.clip(consensus_score, 0, 100))
    if final_score >= 75:
        signal = "STRONG_BUY"
    elif final_score >= 60:
        signal = "BUY"
    elif final_score >= 45:
        signal = "HOLD"
    elif final_score >= 30:
        signal = "SELL"
    else:
        signal = "STRONG_SELL"
    
    return {
        "symbol": symbol,
        "timeframes": results,
        "consensus_trend": consensus_trend,
        "consensus_score": consensus_score,
        "trend_alignment": f"{up_count} UP / {down_count} DOWN",
        "confluence_count": confluence_count,
        "signal": signal,
        "signal_score": final_score,
        "errors": errors,
    }


def mtf_report(symbol: str) -> str:
    """Generate human-readable MTF analysis report."""
    result = multi_timeframe_analysis(symbol)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"MULTIPLE TIMEFRAME ANALYSIS — {symbol}")
    lines.append("=" * 70)
    
    for tf_name in ["INTRADAY_5M", "INTRADAY_15M", "SWING_DAILY", "SWING_WEEKLY", "LONG_WEEKLY"]:
        if tf_name in result["timeframes"]:
            tf = result["timeframes"][tf_name]
            lines.append(f"\n{tf_name} ({tf.get('bars_analyzed', 0)} bars):")
            lines.append(f"  Price: {tf.get('current_price')} | Trend: {tf.get('trend')}")
            lines.append(f"  RSI: {tf.get('rsi')} ({tf.get('rsi_status')}) | MACD: {tf.get('macd_status')}")
            lines.append(f"  Support: {tf.get('support')} | Resistance: {tf.get('resistance')}")
    
    lines.append(f"\n{'=' * 70}")
    lines.append(f"CONSENSUS: {result['consensus_trend']} (Score: {result['consensus_score']})")
    lines.append(f"SIGNAL: {result['signal']} | Confluence: {result['confluence_count']}/2 timeframes")
    lines.append("=" * 70)
    
    return "\n".join(lines)
