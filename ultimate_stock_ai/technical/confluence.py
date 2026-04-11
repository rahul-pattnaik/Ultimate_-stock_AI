# technical/confluence.py
# ─────────────────────────────────────────────────────────────────────────────
# Confluence Scoring System — Professional Trading Feature
# Combines multiple signals into a unified probability score
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfluenceEngine:
    """
    Professional confluence scoring system.
    Combines multiple indicators and assigns weighted probability scores.
    """
    
    def __init__(self):
        self.weights = {
            "trend": 0.20,
            "momentum": 0.18,
            "rsi": 0.15,
            "macd": 0.12,
            "volume": 0.10,
            "support_resistance": 0.10,
            "moving_averages": 0.08,
            "candlestick": 0.05,
            "divergence": 0.02,
        }
    
    def score_trend(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score trend strength (0-100)."""
        close = df["Close"]
        score = 50
        notes = []
        
        # Price vs MAs
        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        
        if len(df) >= 200:
            if close.iloc[-1] > ma200.iloc[-1]:
                score += 20; notes.append("Price above MA200 ✅")
            else:
                score -= 20; notes.append("Price below MA200 ❌")
        
        if len(df) >= 50:
            if close.iloc[-1] > ma50.iloc[-1]:
                score += 15; notes.append("Price above MA50 ✅")
            else:
                score -= 15; notes.append("Price below MA50 ❌")
        
        if close.iloc[-1] > ma20.iloc[-1]:
            score += 10; notes.append("Price above MA20 ✅")
        else:
            score -= 10; notes.append("Price below MA20 ❌")
        
        # Golden/Death Cross
        if len(df) >= 51:
            gc = (ma50.iloc[-1] > ma200.iloc[-1]) and (ma50.iloc[-2] <= ma200.iloc[-2])
            dc = (ma50.iloc[-1] < ma200.iloc[-1]) and (ma50.iloc[-2] >= ma200.iloc[-2])
            if gc: score += 15; notes.append("Golden Cross 🟢")
            if dc: score -= 15; notes.append("Death Cross 🔴")
        
        # ADX
        returns = close.pct_change().dropna()
        adx = min(100, float(returns.rolling(14).std().iloc[-1] * 100 * 10))
        if adx > 25: score += 10; notes.append(f"Strong trend (ADX~{adx:.0f})")
        
        return max(0, min(100, score)), notes
    
    def score_momentum(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score momentum indicators (0-100)."""
        close = df["Close"]
        score = 50
        notes = []
        
        # Price momentum
        mom_5  = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(df) >= 6 else 0
        mom_10 = (close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] * 100 if len(df) >= 11 else 0
        
        if mom_5 > 2:   score += 10; notes.append(f"+5d momentum ({mom_5:.1f}%)")
        elif mom_5 < -2: score -= 10; notes.append(f"-5d momentum ({mom_5:.1f}%)")
        
        if mom_10 > 5:  score += 10; notes.append(f"+10d momentum ({mom_10:.1f}%)")
        elif mom_10 < -5: score -= 10; notes.append(f"-10d momentum ({mom_10:.1f}%)")
        
        # Stochastic
        low_min  = close.rolling(14).min().iloc[-1]
        high_max = close.rolling(14).max().iloc[-1]
        stoch_k  = (close.iloc[-1] - low_min) / (high_max - low_min + 1e-9) * 100
        
        if stoch_k < 20:   score += 10; notes.append("Stochastic Oversold 🟢")
        elif stoch_k > 80: score -= 10; notes.append("Stochastic Overbought 🔴")
        
        return max(0, min(100, score)), notes
    
    def score_rsi(self, df: pd.DataFrame, period: int = 14) -> Tuple[float, List[str]]:
        """Score RSI (0-100)."""
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - (100 / (1 + rs))).iloc[-1])
        
        score = 50
        notes = []
        
        if rsi < 30:
            score = 85; notes.append(f"RSI {rsi:.1f} — Oversold 🟢 (Bullish)")
        elif rsi < 40:
            score = 70; notes.append(f"RSI {rsi:.1f} — Mild Oversold")
        elif rsi < 60:
            score = 55; notes.append(f"RSI {rsi:.1f} — Neutral")
        elif rsi < 70:
            score = 40; notes.append(f"RSI {rsi:.1f} — Approaching Overbought")
        else:
            score = 20; notes.append(f"RSI {rsi:.1f} — Overbought 🔴 (Bearish)")
        
        return score, notes
    
    def score_macd(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score MACD (0-100)."""
        close  = df["Close"]
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal
        
        score = 50
        notes = []
        
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            score = 80; notes.append("MACD Bullish Crossover 🟢")
        elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            score = 20; notes.append("MACD Bearish Crossover 🔴")
        elif hist.iloc[-1] > 0:
            score = 65; notes.append("MACD Histogram Positive ✅")
        else:
            score = 35; notes.append("MACD Histogram Negative ❌")
        
        if macd.iloc[-1] > signal.iloc[-1]:
            score += 10; notes.append("MACD above Signal")
        else:
            score -= 10; notes.append("MACD below Signal")
        
        return max(0, min(100, score)), notes
    
    def score_volume(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score volume (0-100)."""
        vol    = df["Volume"]
        score  = 50
        notes   = []
        
        vol_20_avg = vol.iloc[-20:].mean()
        vol_ratio  = vol.iloc[-1] / vol_20_avg if vol_20_avg > 0 else 1
        
        if vol_ratio > 2:
            score = 80; notes.append(f"Volume surge {vol_ratio:.1f}x 🔥")
        elif vol_ratio > 1.5:
            score = 70; notes.append(f"Volume above avg {vol_ratio:.1f}x")
        elif vol_ratio > 1:
            score = 60; notes.append(f"Volume slightly above avg")
        elif vol_ratio < 0.5:
            score = 40; notes.append(f"Volume very low {vol_ratio:.1f}x")
        else:
            score = 50; notes.append("Normal volume")
        
        # Price-volume alignment
        price_up = close.iloc[-1] > close.iloc[-2] if "Close" in df.columns else False
        
        return max(0, min(100, score)), notes
    
    def score_support_resistance(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score support/resistance proximity (0-100)."""
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        score = 50
        notes  = []
        
        recent_high = high.iloc[-20:].max()
        recent_low  = low.iloc[-20:].min()
        current     = close.iloc[-1]
        
        dist_to_resistance = (recent_high - current) / current * 100
        dist_to_support    = (current - recent_low) / current * 100
        
        if dist_to_resistance < 2:
            score = 30; notes.append(f"Near resistance ({dist_to_resistance:.1f}% away)")
        elif dist_to_resistance < 5:
            score = 45; notes.append(f"Moderate to resistance ({dist_to_resistance:.1f}%)")
        else:
            score = 65; notes.append(f"Room to run ({dist_to_resistance:.1f}% to resistance)")
        
        if dist_to_support < 2:
            score += 15; notes.append(f"Near support ({dist_to_support:.1f}%)")
        
        return max(0, min(100, score)), notes
    
    def score_moving_averages(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score moving average alignment (0-100)."""
        close = df["Close"]
        score = 50
        notes  = []
        
        ma_list = []
        for period in [9, 21, 50, 100, 200]:
            if len(df) >= period:
                ma = close.rolling(period).mean().iloc[-1]
                ma_list.append((period, ma))
        
        for period, ma in ma_list:
            if close.iloc[-1] > ma:
                score += 5; notes.append(f"Price > MA{period}")
            else:
                score -= 5; notes.append(f"Price < MA{period}")
        
        return max(0, min(100, score)), notes
    
    def score_candlestick(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Score candlestick patterns (0-100)."""
        score = 50
        notes  = []
        
        if len(df) < 3:
            return score, notes
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        body     = abs(curr["Close"] - curr["Open"])
        range_   = curr["High"] - curr["Low"]
        body_pct = body / range_ if range_ > 0 else 0
        
        # Doji
        if body_pct < 0.1:
            score = 50; notes.append("Doji - Indecision")
        
        # Hammer
        lower_wick = curr["Open"] - curr["Low"] if curr["Open"] > curr["Close"] else curr["Close"] - curr["Low"]
        if lower_wick > 2 * body and body_pct < 0.3:
            score = 80; notes.append("Hammer - Bullish reversal 🟢")
        
        # Shooting Star
        upper_wick = curr["High"] - curr["Open"] if curr["Open"] > curr["Close"] else curr["High"] - curr["Close"]
        if upper_wick > 2 * body and body_pct < 0.3:
            score = 20; notes.append("Shooting Star - Bearish reversal 🔴")
        
        # Engulfing
        if curr["Close"] > curr["Open"] and prev["Close"] < prev["Open"]:
            if curr["Open"] < prev["Close"] and curr["Close"] > prev["Open"]:
                score = 80; notes.append("Bullish Engulfing 🟢")
        elif curr["Close"] < curr["Open"] and prev["Close"] > prev["Open"]:
            if curr["Open"] > prev["Close"] and curr["Close"] < prev["Open"]:
                score = 20; notes.append("Bearish Engulfing 🔴")
        
        return max(0, min(100, score)), notes
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive confluence analysis.
        Returns weighted score and detailed breakdown.
        """
        if df is None or len(df) < 60:
            return {"error": "Need at least 60 bars for full analysis"}
        
        close = df["Close"]
        
        # Calculate all component scores
        trend_score, trend_notes     = self.score_trend(df)
        momentum_score, mom_notes     = self.score_momentum(df)
        rsi_score, rsi_notes         = self.score_rsi(df)
        macd_score, macd_notes       = self.score_macd(df)
        volume_score, vol_notes       = self.score_volume(df)
        sr_score, sr_notes           = self.score_support_resistance(df)
        ma_score, ma_notes           = self.score_moving_averages(df)
        candle_score, candle_notes    = self.score_candlestick(df)
        
        # Calculate weighted total
        total = (
            trend_score    * self.weights["trend"] +
            momentum_score * self.weights["momentum"] +
            rsi_score      * self.weights["rsi"] +
            macd_score     * self.weights["macd"] +
            volume_score   * self.weights["volume"] +
            sr_score       * self.weights["support_resistance"] +
            ma_score       * self.weights["moving_averages"] +
            candle_score   * self.weights["candlestick"]
        )
        
        # Determine signal
        if total >= 75:
            signal = "STRONG BUY"
            signal_desc = "High confluence - multiple bullish signals align 🚀"
        elif total >= 60:
            signal = "BUY"
            signal_desc = "Bullish confluence - majority of indicators positive 📈"
        elif total >= 45:
            signal = "HOLD"
            signal_desc = "Neutral - mixed signals, wait for clarity ➡️"
        elif total >= 30:
            signal = "SELL"
            signal_desc = "Bearish confluence - majority of indicators negative 📉"
        else:
            signal = "STRONG SELL"
            signal_desc = "High bearish confluence - multiple signals align 🔴"
        
        # Count bullish/bearish indicators
        bullish_count = sum([
            trend_score > 55,
            momentum_score > 55,
            rsi_score > 60,
            rsi_score < 40,
            macd_score > 55,
            volume_score > 55,
            sr_score > 55,
            ma_score > 55,
        ])
        
        return {
            "total_score": round(total, 1),
            "signal": signal,
            "signal_description": signal_desc,
            "confidence": min(100, int(abs(total - 50) * 2 + bullish_count * 5)),
            "components": {
                "Trend":            {"score": round(trend_score, 1), "weight": 20, "notes": trend_notes},
                "Momentum":         {"score": round(momentum_score, 1), "weight": 18, "notes": mom_notes},
                "RSI":              {"score": round(rsi_score, 1), "weight": 15, "notes": rsi_notes},
                "MACD":             {"score": round(macd_score, 1), "weight": 12, "notes": macd_notes},
                "Volume":           {"score": round(volume_score, 1), "weight": 10, "notes": vol_notes},
                "Support/Resistance": {"score": round(sr_score, 1), "weight": 10, "notes": sr_notes},
                "Moving Averages":   {"score": round(ma_score, 1), "weight": 8, "notes": ma_notes},
                "Candlestick":      {"score": round(candle_score, 1), "weight": 5, "notes": candle_notes},
            },
            "bullish_indicators": bullish_count,
            "current_price": round(float(close.iloc[-1]), 2),
        }


def confluence_report(df: pd.DataFrame) -> str:
    """Generate human-readable confluence report."""
    engine  = ConfluenceEngine()
    result  = engine.analyze(df)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = []
    lines.append("=" * 70)
    lines.append("CONFLUENCE ANALYSIS — Professional Signal Detection")
    lines.append("=" * 70)
    
    lines.append(f"\nOVERALL SCORE: {result['total_score']}/100")
    lines.append(f"SIGNAL: {result['signal']}")
    lines.append(f"Description: {result['signal_description']}")
    lines.append(f"Confidence: {result['confidence']}%")
    lines.append(f"Bullish Indicators: {result['bullish_indicators']}/8")
    
    lines.append("\n" + "-" * 70)
    lines.append("COMPONENT BREAKDOWN:")
    lines.append("-" * 70)
    
    for component, data in result["components"].items():
        bar_len = int(data["score"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"\n{component:<20} {data['score']:5.1f}/100 [{bar}] ({data['weight']}%)")
        for note in data["notes"][:3]:
            lines.append(f"  • {note}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)
