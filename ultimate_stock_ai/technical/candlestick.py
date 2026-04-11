# technical/candlestick.py
# ─────────────────────────────────────────────────────────────────────────────
# Candlestick Pattern Recognition — Professional Trading Feature
# Detects: Doji, Hammer, Engulfing, Morning/Evening Star, Harami, etc.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def _add_candle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard candlestick calculation columns."""
    df = df.copy()
    df["body"]     = df["Close"] - df["Open"]
    df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["body_size"]  = abs(df["body"])
    df["range"]      = df["High"] - df["Low"]
    return df


def _is_doji(df: pd.DataFrame, i: int, threshold: float = 0.1) -> bool:
    """Doji: Open ≈ Close, small body relative to range."""
    if i < 0:
        i = len(df) + i
    row = df.iloc[i]
    body   = abs(row["body"])
    range_ = row["range"]
    if range_ == 0:
        return False
    return body / range_ < threshold


def _is_hammer(df: pd.DataFrame, i: int) -> bool:
    """
    Hammer: Small body at top, long lower wick (≥2x body), minimal upper wick.
    Bullish reversal signal.
    """
    if i < 0:
        i = len(df) + i
    row = df.iloc[i]
    if row["range"] == 0:
        return False
    body_size     = abs(row["body"])
    lower_wick    = row["lower_wick"]
    upper_wick    = row["upper_wick"]
    
    # Lower wick at least 2x body
    # Upper wick less than 20% of range
    # Body near top of candle
    return (lower_wick >= 2 * body_size and 
            upper_wick < 0.2 * row["range"] and
            body_size > 0 and  # Bullish hammer (close > open)
            row["Close"] > row["Open"])


def _is_shooting_star(df: pd.DataFrame, i: int) -> bool:
    """Shooting Star: Small body at bottom, long upper wick. Bearish signal."""
    if i < 0:
        i = len(df) + i
    row = df.iloc[i]
    if row["range"] == 0:
        return False
    body_size  = abs(row["body"])
    upper_wick = row["upper_wick"]
    lower_wick = row["lower_wick"]
    
    return (upper_wick >= 2 * body_size and
            lower_wick < 0.2 * row["range"] and
            body_size > 0 and  # Bearish (close < open)
            row["Close"] < row["Open"])


def _is_bullish_engulfing(df: pd.DataFrame, i: int) -> Tuple[bool, str]:
    """
    Bullish Engulfing: Current bullish candle engulfs previous bearish candle.
    Strong reversal signal.
    """
    if i < 1:
        return False, ""
    curr = df.iloc[i]
    prev = df.iloc[i - 1]
    
    # Previous: bearish (close < open)
    # Current: bullish (close > open)
    # Current body engulfs previous body
    if (prev["body"] < 0 and curr["body"] > 0 and
        curr["Open"] < prev["Close"] and
        curr["Close"] > prev["Open"]):
        return True, f"Engulfed {abs(prev['body']):.2f}pt bearish candle"
    return False, ""


def _is_bearish_engulfing(df: pd.DataFrame, i: int) -> Tuple[bool, str]:
    """
    Bearish Engulfing: Current bearish candle engulfs previous bullish candle.
    Strong reversal signal.
    """
    if i < 1:
        return False, ""
    curr = df.iloc[i]
    prev = df.iloc[i - 1]
    
    # Previous: bullish (close > open)
    # Current: bearish (close < open)
    if (prev["body"] > 0 and curr["body"] < 0 and
        curr["Open"] > prev["Close"] and
        curr["Close"] < prev["Open"]):
        return True, f"Engulfed {abs(prev['body']):.2f}pt bullish candle"
    return False, ""


def _is_morning_star(df: pd.DataFrame, i: int) -> bool:
    """
    Morning Star: 3-candle pattern
    1. Large bearish candle
    2. Small body (doji/spinning top)
    3. Large bullish candle closing above midpoint of first candle
    """
    if i < 2:
        return False
    c1 = df.iloc[i - 2]
    c2 = df.iloc[i - 1]
    c3 = df.iloc[i]
    
    # Candle 1: Large bearish
    c1_bearish = c1["body"] < 0 and abs(c1["body"]) > 0.6 * c1["range"]
    # Candle 2: Small body (doji-like)
    c2_small = abs(c2["body"]) < 0.3 * c2["range"] if c2["range"] > 0 else False
    # Candle 3: Large bullish closing above midpoint of c1
    c3_bullish = c3["body"] > 0 and abs(c3["body"]) > 0.6 * c3["range"]
    midpoint_c1 = (c1["Open"] + c1["Close"]) / 2
    above_midpoint = c3["Close"] > midpoint_c1
    
    return c1_bearish and c2_small and c3_bullish and above_midpoint


def _is_evening_star(df: pd.DataFrame, i: int) -> bool:
    """
    Evening Star: 3-candle pattern (opposite of Morning Star)
    1. Large bullish candle
    2. Small body
    3. Large bearish candle closing below midpoint of first candle
    """
    if i < 2:
        return False
    c1 = df.iloc[i - 2]
    c2 = df.iloc[i - 1]
    c3 = df.iloc[i]
    
    c1_bullish = c1["body"] > 0 and abs(c1["body"]) > 0.6 * c1["range"]
    c2_small = abs(c2["body"]) < 0.3 * c2["range"] if c2["range"] > 0 else False
    c3_bearish = c3["body"] < 0 and abs(c3["body"]) > 0.6 * c3["range"]
    midpoint_c1 = (c1["Open"] + c1["Close"]) / 2
    below_midpoint = c3["Close"] < midpoint_c1
    
    return c1_bullish and c2_small and c3_bearish and below_midpoint


def _is_hammer_inverted(df: pd.DataFrame, i: int) -> bool:
    """Inverted Hammer: Like hammer but wick is above body. Bullish if confirmed."""
    if i < 0:
        i = len(df) + i
    row = df.iloc[i]
    if row["range"] == 0:
        return False
    body_size  = abs(row["body"])
    upper_wick = row["upper_wick"]
    lower_wick = row["lower_wick"]
    return (upper_wick >= 2 * body_size and
            lower_wick < 0.2 * row["range"])


def _is_hanging_man(df: pd.DataFrame, i: int) -> bool:
    """
    Hanging Man: Same as hammer but appears after an uptrend.
    Bearish reversal signal.
    """
    if i < 5:
        return False
    row = df.iloc[i]
    # Check if previous 5 candles were generally up
    prev_avg = df["Close"].iloc[i-5:i].mean()
    return (_is_hammer(df, i) and row["Close"] < prev_avg)


def _is_dark_cloud_cover(df: pd.DataFrame, i: int) -> bool:
    """
    Dark Cloud Cover: Bearish reversal pattern.
    1. Uptrend with bullish candle
    2. Gap up, then closes below midpoint of previous candle
    """
    if i < 1:
        return False
    curr = df.iloc[i]
    prev = df.iloc[i - 1]
    
    prev_bullish = prev["body"] > 0
    curr_bearish = curr["body"] < 0
    curr_opens_higher = curr["Open"] > prev["Close"]
    curr_closes_below_mid = curr["Close"] < (prev["Open"] + prev["Close"]) / 2
    
    return (prev_bullish and curr_bearish and 
            curr_opens_higher and curr_closes_below_mid)


def _is_piercing_line(df: pd.DataFrame, i: int) -> bool:
    """
    Piercing Line: Bullish reversal pattern (opposite of Dark Cloud Cover).
    """
    if i < 1:
        return False
    curr = df.iloc[i]
    prev = df.iloc[i - 1]
    
    prev_bearish = prev["body"] < 0
    curr_bullish = curr["body"] > 0
    curr_opens_lower = curr["Open"] < prev["Close"]
    curr_closes_above_mid = curr["Close"] > (prev["Open"] + prev["Close"]) / 2
    
    return (prev_bearish and curr_bullish and
            curr_opens_lower and curr_closes_above_mid)


def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    """
    Detect all candlestick patterns in the dataframe.
    Returns comprehensive analysis with patterns detected.
    """
    if df is None or len(df) < 3:
        return {"error": "Need at least 3 candles"}
    
    df = _add_candle_columns(df)
    patterns = []
    
    # Check last 20 candles for patterns
    for i in range(max(0, len(df) - 20), len(df)):
        date = df.index[i].strftime("%Y-%m-%d")
        row  = df.iloc[i]
        
        # Single-candle patterns
        if _is_doji(df, i, threshold=0.15):
            patterns.append({
                "date": date,
                "pattern": "DOJI",
                "type": "NEUTRAL",
                "candle_idx": i,
                "description": "Indecision - buyers and sellers in balance"
            })
        
        if _is_hammer(df, i):
            patterns.append({
                "date": date,
                "pattern": "HAMMER",
                "type": "BULLISH",
                "candle_idx": i,
                "description": "Bullish reversal - rejection of lower prices"
            })
        
        if _is_shooting_star(df, i):
            patterns.append({
                "date": date,
                "pattern": "SHOOTING_STAR",
                "type": "BEARISH",
                "candle_idx": i,
                "description": "Bearish reversal - rejection of higher prices"
            })
        
        if _is_hanging_man(df, i):
            patterns.append({
                "date": date,
                "pattern": "HANGING_MAN",
                "type": "BEARISH",
                "candle_idx": i,
                "description": "Bearish reversal in uptrend - warning signal"
            })
        
        if _is_hammer_inverted(df, i):
            patterns.append({
                "date": date,
                "pattern": "INVERTED_HAMMER",
                "type": "BULLISH",
                "candle_idx": i,
                "description": "Bullish if confirmed next candle"
            })
        
        # Two-candle patterns
        engulf_bull, eng_desc = _is_bullish_engulfing(df, i)
        if engulf_bull:
            patterns.append({
                "date": date,
                "pattern": "BULLISH_ENGULFING",
                "type": "BULLISH",
                "candle_idx": i,
                "description": eng_desc
            })
        
        engulf_bear, eng_desc = _is_bearish_engulfing(df, i)
        if engulf_bear:
            patterns.append({
                "date": date,
                "pattern": "BEARISH_ENGULFING",
                "type": "BEARISH",
                "candle_idx": i,
                "description": eng_desc
            })
        
        if _is_dark_cloud_cover(df, i):
            patterns.append({
                "date": date,
                "pattern": "DARK_CLOUD_COVER",
                "type": "BEARISH",
                "candle_idx": i,
                "description": "Bearish reversal - clouds over previous gains"
            })
        
        if _is_piercing_line(df, i):
            patterns.append({
                "date": date,
                "pattern": "PIERCING_LINE",
                "type": "BULLISH",
                "candle_idx": i,
                "description": "Bullish reversal - pierces through resistance"
            })
        
        # Three-candle patterns
        if _is_morning_star(df, i):
            patterns.append({
                "date": date,
                "pattern": "MORNING_STAR",
                "type": "BULLISH",
                "candle_idx": i,
                "description": "Strong bullish reversal - morning star shines"
            })
        
        if _is_evening_star(df, i):
            patterns.append({
                "date": date,
                "pattern": "EVENING_STAR",
                "type": "BEARISH",
                "candle_idx": i,
                "description": "Strong bearish reversal - evening darkness"
            })
    
    # Summary statistics
    bullish_patterns = [p for p in patterns if p["type"] == "BULLISH"]
    bearish_patterns = [p for p in patterns if p["type"] == "BEARISH"]
    neutral_patterns  = [p for p in patterns if p["type"] == "NEUTRAL"]
    
    # Latest pattern analysis
    latest_pattern = patterns[-1] if patterns else None
    
    # Determine overall signal
    if bullish_patterns and not bearish_patterns:
        signal = "BULLISH"
        signal_score = 60 + min(len(bullish_patterns) * 5, 25)
    elif bearish_patterns and not bullish_patterns:
        signal = "BEARISH"
        signal_score = 40 - min(len(bearish_patterns) * 5, 25)
    elif bullish_patterns and bearish_patterns:
        if len(bullish_patterns) > len(bearish_patterns):
            signal = "SLIGHTLY_BULLISH"
            signal_score = 55
        elif len(bearish_patterns) > len(bullish_patterns):
            signal = "SLIGHTLY_BEARISH"
            signal_score = 45
        else:
            signal = "MIXED"
            signal_score = 50
    else:
        signal = "NO_PATTERN"
        signal_score = 50
    
    return {
        "patterns": patterns,
        "latest_pattern": latest_pattern,
        "bullish_count": len(bullish_patterns),
        "bearish_count": len(bearish_patterns),
        "neutral_count": len(neutral_patterns),
        "signal": signal,
        "signal_score": signal_score,
        "bullish_patterns": bullish_patterns,
        "bearish_patterns": bearish_patterns,
    }


def candlestick_report(df: pd.DataFrame) -> str:
    """Generate human-readable candlestick analysis report."""
    result = detect_candlestick_patterns(df)
    
    if "error" in result:
        return result["error"]
    
    lines = []
    lines.append("=" * 50)
    lines.append("CANDLESTICK PATTERN ANALYSIS")
    lines.append("=" * 50)
    
    if result["bullish_count"] > 0:
        lines.append(f"\n📈 BULLISH PATTERNS ({result['bullish_count']}):")
        for p in result["bullish_patterns"]:
            lines.append(f"  • {p['pattern']} on {p['date']}: {p['description']}")
    
    if result["bearish_count"] > 0:
        lines.append(f"\n📉 BEARISH PATTERNS ({result['bearish_count']}):")
        for p in result["bearish_patterns"]:
            lines.append(f"  • {p['pattern']} on {p['date']}: {p['description']}")
    
    if result["bullish_count"] == 0 and result["bearish_count"] == 0:
        lines.append("\nNo significant patterns detected in recent candles.")
    
    lines.append(f"\nSignal: {result['signal']} (Score: {result['signal_score']})")
    
    return "\n".join(lines)
