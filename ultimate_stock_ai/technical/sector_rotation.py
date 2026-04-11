# technical/sector_rotation.py
# ─────────────────────────────────────────────────────────────────────────────
# Sector Rotation Tracking — Professional Trading Feature
# Tracks sector momentum and identifies rotation patterns
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_fetcher import get_stock_data


NSE_SECTOR_INDICES = {
    "NIFTY50": "^NSEI",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY Realty": "^CNXREALTY",
    "NIFTY_CONSUMER": "^CNXCONSUM",
    "NIFTY_INFRA": "^CNXINFRA",
    "NIFTY_MEDIA": "^CNXMEDIA",
    "NIFTY_PSUBANK": "^CNXPSUBANK",
    "NIFTY_SERVICES": "^CNXSERVICE",
}


SECTOR_STOCKS = {
    "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "IT": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "MINDTREE.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "COLPAL.NS", " GODREJCP.NS"],
    "PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "ZYDUSLIFE.NS", "AUROPHARMA.NS", "LUPIN.NS"],
    "AUTO": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"],
    "ENERGY": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "NTPC.NS", "POWERGRID.NS"],
    "METALS": ["TATASTEEL.NS", "HINDALCO.NS", "VEDANTA.NS", "JSWSTEEL.NS", "COALINDIA.NS"],
    "FINANCE": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFC.NS", "SBILIFE.NS", "ICICIPRULI.NS"],
    "CONSUMER": ["TITAN.NS", "DMART.NS", "BERGEPAINT.NS", "ASIANPAINT.NS", "ULTRACEMCO.NS"],
}


def _calculate_momentum(close: pd.Series, periods: List[int] = [5, 20, 60]) -> Dict[str, float]:
    """Calculate momentum for multiple periods."""
    momentum = {}
    for period in periods:
        if len(close) >= period + 1:
            momentum[f"mom_{period}d"] = float(
                (close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1] * 100
            )
        else:
            momentum[f"mom_{period}d"] = 0.0
    return momentum


def _calculate_relative_strength(price: float, sector_avg_return: float) -> float:
    """Calculate relative strength vs sector."""
    return price - sector_avg_return


def analyze_sector_index(symbol: str, name: str) -> Optional[Dict]:
    """Analyze a single sector index."""
    try:
        df = get_stock_data(symbol, period="3mo", interval="1d", use_cache=True)
        if df is None or len(df) < 20:
            return None
        
        close = df["Close"]
        
        # Calculate momentum
        momentum = _calculate_momentum(close)
        
        # Current price and returns
        current_price = float(close.iloc[-1])
        returns_1m    = momentum.get("mom_20d", 0)
        returns_3m    = momentum.get("mom_60d", 0)
        
        # Volatility
        returns = close.pct_change().dropna()
        volatility = float(returns.rolling(20).std().iloc[-1] * 100 * np.sqrt(252))
        
        # Trend
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(df) >= 50 else ma20
        
        if current_price > ma20 > ma50:
            trend = "STRONG_UP"
        elif current_price > ma20:
            trend = "UP"
        elif current_price < ma20 < ma50:
            trend = "STRONG_DOWN"
        elif current_price < ma20:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"
        
        # Score
        score = 50 + returns_1m * 2 + returns_3m * 0.5
        if trend == "STRONG_UP":
            score += 15
        elif trend == "UP":
            score += 5
        elif trend == "STRONG_DOWN":
            score -= 15
        elif trend == "DOWN":
            score -= 5
        
        score = int(np.clip(score, 0, 100))
        
        return {
            "name": name,
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "momentum_5d": round(momentum.get("mom_5d", 0), 2),
            "momentum_20d": round(returns_1d, 2),
            "momentum_60d": round(returns_3m, 2),
            "volatility": round(volatility, 2),
            "trend": trend,
            "score": score,
            "above_ma20": current_price > ma20,
            "above_ma50": current_price > ma50,
        }
    except Exception as e:
        return None


def analyze_sector_stocks(stocks: List[str], sector_name: str) -> Optional[Dict]:
    """Analyze a sector by averaging its top stocks."""
    momentum_5d_list  = []
    momentum_20d_list = []
    scores = []
    
    for stock in stocks[:5]:  # Top 5 stocks
        try:
            df = get_stock_data(stock, period="3mo", interval="1d", use_cache=True)
            if df is None or len(df) < 20:
                continue
            
            mom = _calculate_momentum(df["Close"])
            momentum_5d_list.append(mom.get("mom_5d", 0))
            momentum_20d_list.append(mom.get("mom_20d", 0))
            
            # Individual stock score
            price = float(df["Close"].iloc[-1])
            ma20  = float(df["Close"].rolling(20).mean().iloc[-1])
            score = 50 + (mom.get("mom_20d", 0) * 2)
            scores.append(score)
        except:
            continue
    
    if not scores:
        return None
    
    avg_mom_5d  = np.mean(momentum_5d_list)
    avg_mom_20d = np.mean(momentum_20d_list)
    avg_score   = np.mean(scores)
    
    return {
        "sector": sector_name,
        "stocks_analyzed": len(scores),
        "avg_momentum_5d": round(avg_mom_5d, 2),
        "avg_momentum_20d": round(avg_mom_20d, 2),
        "avg_score": round(avg_score, 0),
        "momentum": "STRONG" if avg_mom_20d > 5 else "WEAK" if avg_mom_20d < -5 else "NEUTRAL",
    }


def sector_rotation_analysis() -> Dict:
    """
    Comprehensive sector rotation analysis.
    Analyzes NSE sector indices and key sector stocks.
    """
    sector_results = []
    
    # Analyze sector indices
    for name, symbol in NSE_SECTOR_INDICES.items():
        result = analyze_sector_index(symbol, name)
        if result:
            sector_results.append(result)
    
    # Rank sectors by momentum score
    if sector_results:
        sector_results = sorted(sector_results, key=lambda x: x["score"], reverse=True)
        
        # Identify leaders and laggards
        top_3  = sector_results[:3]
        bottom_3 = sector_results[-3:]
        
        # Detect rotation
        recent_leaders = [s for s in sector_results if s["momentum_20d"] > 2]
        recent_laggards = [s for s in sector_results if s["momentum_20d"] < -2]
        
        if recent_leaders:
            strongest_sector = max(recent_leaders, key=lambda x: x["momentum_20d"])["name"]
        else:
            strongest_sector = sector_results[0]["name"] if sector_results else "N/A"
        
        if recent_laggards:
            weakest_sector = min(recent_laggards, key=lambda x: x["momentum_20d"])["name"]
        else:
            weakest_sector = sector_results[-1]["name"] if sector_results else "N/A"
        
        # Overall market bias
        bullish_sectors = sum(1 for s in sector_results if s["trend"].endswith("UP"))
        bearish_sectors = sum(1 for s in sector_results if s["trend"].endswith("DOWN"))
        
        if bullish_sectors > bearish_sectors:
            market_bias = "BULLISH"
        elif bearish_sectors > bullish_sectors:
            market_bias = "BEARISH"
        else:
            market_bias = "NEUTRAL"
        
        return {
            "all_sectors": sector_results,
            "top_sectors": top_3,
            "bottom_sectors": bottom_3,
            "strongest_sector": strongest_sector,
            "weakest_sector": weakest_sector,
            "market_bias": market_bias,
            "bullish_count": bullish_sectors,
            "bearish_count": bearish_sectors,
            "rotation_signal": f"Rotate from {weakest_sector} to {strongest_sector}" if 
                               weakest_sector != strongest_sector else "No rotation signal",
        }
    
    return {"error": "Could not analyze sectors"}


def sector_report() -> str:
    """Generate human-readable sector rotation report."""
    result = sector_rotation_analysis()
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    lines = []
    lines.append("=" * 70)
    lines.append("SECTOR ROTATION ANALYSIS — NSE India")
    lines.append("=" * 70)
    
    lines.append("\n📈 TOP PERFORMING SECTORS:")
    for s in result["top_sectors"]:
        lines.append(f"  {s['name']}: Score {s['score']} | Trend: {s['trend']}")
        lines.append(f"    Momentum: 5d {s['momentum_5d']:+.2f}% | 20d {s['momentum_20d']:+.2f}%")
    
    lines.append("\n📉 BOTTOM PERFORMING SECTORS:")
    for s in result["bottom_sectors"]:
        lines.append(f"  {s['name']}: Score {s['score']} | Trend: {s['trend']}")
        lines.append(f"    Momentum: 5d {s['momentum_5d']:+.2f}% | 20d {s['momentum_20d']:+.2f}%")
    
    lines.append(f"\n{'=' * 70}")
    lines.append(f"MARKET BIAS: {result['market_bias']} ({result['bullish_count']} bullish / {result['bearish_count']} bearish)")
    lines.append(f"STRONGEST: {result['strongest_sector']} | WEAKEST: {result['weakest_sector']}")
    lines.append(f"ROTATION SIGNAL: {result['rotation_signal']}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def get_sector_for_stock(symbol: str) -> Optional[str]:
    """Identify which sector a stock belongs to."""
    symbol = symbol.upper()
    for sector, stocks in SECTOR_STOCKS.items():
        if any(symbol in s or s in symbol for s in stocks):
            return sector
    return None
