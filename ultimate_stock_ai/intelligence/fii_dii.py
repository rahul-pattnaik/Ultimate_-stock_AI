# intelligence/fii_dii.py
# ─────────────────────────────────────────────────────────────────────────────
# FII/DII Activity Analysis — Professional Trading Feature
# Tracks institutional buying/selling patterns
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf


def get_fii_dii_data() -> Optional[Dict]:
    """
    Fetch FII/DII activity data from NSE.
    Note: This requires NSE API access or a data provider.
    Returns simulated data for demonstration.
    """
    try:
        nse_url = "https://www.nseindia.com/api/fiidedi"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        
        # Try to fetch real data (requires proper headers and session)
        # For now, return simulated data structure
        return {
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "fii_buy": 0,
            "fii_sell": 0,
            "fii_net": 0,
            "dii_buy": 0,
            "dii_sell": 0,
            "dii_net": 0,
            "note": "Data unavailable - requires NSE API access"
        }
    except Exception as e:
        return {
            "error": str(e),
            "note": "FII/DII data requires premium data subscription"
        }


def analyze_delivery_data(symbol: str, period: str = "1mo") -> Dict:
    """
    Analyze delivery percentage - key indicator of institutional interest.
    High delivery % + price rise = institutional accumulation.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)
        
        if df is None or len(df) < 5:
            return {"error": "Insufficient data"}
        
        # Calculate delivery metrics
        # Note: yfinance doesn't provide delivery data directly
        # Use volume as proxy for activity
        
        df["vol_ma"] = df["Volume"].rolling(20).mean()
        df["price_change"] = df["Close"].pct_change()
        
        current_vol = float(df["Volume"].iloc[-1])
        avg_vol    = float(df["vol_ma"].iloc[-1])
        vol_ratio  = current_vol / avg_vol if avg_vol > 0 else 1
        
        recent_returns = df["price_change"].iloc[-5:].mean() * 100
        
        # Delivery-like analysis using volume patterns
        if vol_ratio > 1.5 and recent_returns > 0:
            signal = "BULLISH_ACCUMULATION"
            description = "High volume + price rise = institutional buying"
        elif vol_ratio > 1.5 and recent_returns < 0:
            signal = "BEARISH_DISTRIBUTION"
            description = "High volume + price fall = institutional selling"
        elif vol_ratio < 0.7 and abs(recent_returns) < 1:
            signal = "LOW_ACTIVITY"
            description = "Low volume, no clear institutional activity"
        else:
            signal = "NEUTRAL"
            description = "Mixed signals"
        
        return {
            "symbol": symbol,
            "volume_ratio": round(vol_ratio, 2),
            "volume_status": "HIGH" if vol_ratio > 1.5 else "LOW" if vol_ratio < 0.7 else "NORMAL",
            "recent_return_5d": round(recent_returns, 2),
            "signal": signal,
            "description": description,
            "interpretation": "High delivery % with price rise indicates strong institutional accumulation"
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_block_deals(symbol: str = None) -> Dict:
    """
    Analyze block deals - large transactions that move markets.
    Returns simulated analysis for demonstration.
    """
    return {
        "note": "Block deal data requires BSE/NSE live feed subscription",
        "sample_signals": [
            "Large block buy > 1% holdings: Bullish signal",
            "Large block sell > 1% holdings: Bearish signal",
            "Multiple blocks in same direction: Strong conviction",
        ],
        "practical_tip": "Track delivery percentage increase alongside price"
    }


def smart_money_analysis(symbol: str) -> Dict:
    """
    Comprehensive smart money tracking.
    Combines volume, price action, and delivery analysis.
    """
    delivery_result = analyze_delivery_data(symbol)
    block_result    = analyze_block_deals(symbol)
    
    # Combine signals
    signals = []
    
    if "error" not in delivery_result:
        signals.append(delivery_result.get("signal", "NEUTRAL"))
    
    # Determine overall smart money bias
    bullish_count  = signals.count("BULLISH_ACCUMULATION")
    bearish_count  = signals.count("BEARISH_DISTRIBUTION")
    
    if bullish_count > bearish_count:
        bias = "INSTITUTIONAL_BUYING"
        score = 60 + bullish_count * 10
    elif bearish_count > bullish_count:
        bias = "INSTITUTIONAL_SELLING"
        score = 40 - bearish_count * 10
    else:
        bias = "NO_CLEAR_BIAS"
        score = 50
    
    return {
        "symbol": symbol,
        "smart_money_bias": bias,
        "score": max(0, min(100, score)),
        "delivery_analysis": delivery_result,
        "block_deal_analysis": block_result,
        "practical_tips": [
            "Track delivery % over rolling 5-day average",
            "Delivery > 90% with price rise = strong buy signal",
            "Delivery > 85% with price fall = distribution (sell)",
            "Compare sector delivery % to identify rotation"
        ]
    }


def fii_dii_report(symbol: str) -> str:
    """Generate human-readable FII/DII analysis report."""
    result = smart_money_analysis(symbol)
    
    if "error" in result:
        return f"Error: {result.get('error', 'Unknown error')}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("SMART MONEY ANALYSIS")
    lines.append("=" * 60)
    
    lines.append(f"\nSymbol: {result['symbol']}")
    lines.append(f"Smart Money Bias: {result['smart_money_bias']}")
    lines.append(f"Score: {result['score']}/100")
    
    if "error" not in result.get("delivery_analysis", {}):
        da = result["delivery_analysis"]
        lines.append(f"\nDelivery/Volume Analysis:")
        lines.append(f"  Volume Ratio: {da.get('volume_ratio')}x")
        lines.append(f"  Status: {da.get('volume_status')}")
        lines.append(f"  5d Return: {da.get('recent_return_5d', 0):+.2f}%")
        lines.append(f"  Signal: {da.get('signal')}")
        lines.append(f"  Interpretation: {da.get('interpretation')}")
    
    lines.append("\n📌 Practical Tips:")
    for tip in result["practical_tips"]:
        lines.append(f"  • {tip}")
    
    return "\n".join(lines)
