from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from technical.breakout import breakout_analysis
from technical.support_resistance import get_support_resistance
from technical.trend_detection import trend_analysis
from technical.volatility import volatility_report


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gains / (losses + 1e-9)
    return float((100 - (100 / (1 + rs))).iloc[-1])


def _timeframe_signal(df: pd.DataFrame, window: int) -> str:
    if len(df) < max(window, 10):
        return "Neutral"
    recent = df.tail(window)
    close = recent["Close"]
    sma_fast = close.rolling(min(5, max(2, window // 3))).mean().iloc[-1]
    sma_slow = close.rolling(min(window, len(close))).mean().iloc[-1]
    momentum = (close.iloc[-1] / close.iloc[0] - 1) * 100
    if close.iloc[-1] > sma_fast > sma_slow and momentum > 2:
        return "Bullish"
    if close.iloc[-1] < sma_fast < sma_slow and momentum < -2:
        return "Bearish"
    return "Neutral"


def advanced_technical_report(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or len(df) < 60:
        return {"error": "Need at least 60 rows for advanced technical analysis"}

    close = df["Close"].astype(float)
    latest = float(close.iloc[-1])
    pct_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0.0
    pct_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else pct_5
    pct_60 = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 60 else pct_20
    rsi = _rsi(close)
    momentum_score = float(np.clip(50 + pct_5 * 2.0 + pct_20 * 1.2 + pct_60 * 0.6 - abs(rsi - 55) * 0.35, 0, 100))

    trend = trend_analysis(df)
    breakout = breakout_analysis(df)
    sr = get_support_resistance(df)
    if isinstance(sr, tuple):
        support_level, resistance_level = sr
    elif isinstance(sr, dict):
        support_level = sr.get("support")
        resistance_level = sr.get("resistance")
    else:
        support_level, resistance_level = None, None
    vol = volatility_report(df)
    regime_value = str(vol.get("regime", vol.get("volatility_regime", "Normal"))) if isinstance(vol, dict) else "Normal"
    breakout_text = str(breakout).upper()
    breakout_ready = "BREAKOUT" in breakout_text or "BUY" in breakout_text

    timeframes = {
        "short": _timeframe_signal(df, 10),
        "medium": _timeframe_signal(df, 30),
        "long": _timeframe_signal(df, 90),
    }
    bullish_votes = sum(1 for value in timeframes.values() if value == "Bullish")
    bearish_votes = sum(1 for value in timeframes.values() if value == "Bearish")

    confluence = 50.0
    confluence += 12.0 if "UPTREND" in str(trend.get("trend", "")).upper() else -12.0 if "DOWNTREND" in str(trend.get("trend", "")).upper() else 0.0
    confluence += 10.0 if breakout_ready else 0.0
    confluence += (bullish_votes - bearish_votes) * 8.0
    confluence += 8.0 if 45 <= rsi <= 65 else -5.0
    confluence = float(np.clip(confluence, 0, 100))

    return {
        "trend": trend.get("trend", "Unknown") if isinstance(trend, dict) else str(trend),
        "trend_strength": trend.get("strength", "Unknown") if isinstance(trend, dict) else "Unknown",
        "momentum_score": round(momentum_score, 2),
        "support": round(_as_float(support_level), 2),
        "resistance": round(_as_float(resistance_level), 2),
        "breakout_ready": breakout_ready,
        "volatility_regime": regime_value,
        "multi_timeframe": timeframes,
        "confluence_score": round(confluence, 2),
        "notes": [
            f"Momentum stack: 5d {pct_5:+.2f}%, 20d {pct_20:+.2f}%, 60d {pct_60:+.2f}%",
            f"RSI regime: {rsi:.2f}",
            f"Price vs support/resistance: {latest:.2f} vs {_as_float(support_level):.2f}/{_as_float(resistance_level):.2f}",
            f"Multi-timeframe alignment: {bullish_votes} bullish, {bearish_votes} bearish",
        ],
    }
