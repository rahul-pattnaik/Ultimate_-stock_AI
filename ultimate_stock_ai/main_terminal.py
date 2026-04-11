# main_terminal.py
# ─────────────────────────────────────────────────────────────────────────────
# Ultimate Stock AI — Main Terminal
# Enhanced with: RSI Divergence, Candlestick, Volume Analysis, MTF Analysis,
#                Sector Rotation, Confluence Scoring, Risk/Reward Analysis
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np

from core.data_fetcher import get_stock_data, get_latest_price
from core.utils       import (fmt_price, fmt_pct, fmt_volume,
                               score_to_signal, score_to_grade,
                               signal_color, RESET, print_header)

from technical.trend_detection   import detect_trend, trend_analysis
from technical.breakout          import breakout_signal, breakout_analysis
from technical.support_resistance import get_support_resistance, get_sr_zones
from technical.volume_profile    import volume_profile, volume_profile_full
from technical.momentum          import momentum_report
from technical.moving_averages   import ma_signal
from technical.supertrend        import supertrend_signal
from technical.volatility        import volatility_report
from technical.vwap              import vwap_bands
from technical.fibonacci         import fibonacci_retracements
from technical.ichimoku          import ichimoku_signal

# NEW IMPORTS
from technical.rsi_divergence    import detect_rsi_divergence, rsi_divergence_report
from technical.candlestick       import detect_candlestick_patterns, candlestick_report
from technical.volume_analysis   import volume_analysis, volume_report
from technical.confluence        import ConfluenceEngine, confluence_report
from intelligence.fii_dii        import smart_money_analysis, fii_dii_report

from ai.price_prediction   import predict_price
from ai.signal_model       import ai_signal
from ai.ranking_engine     import stock_score, detailed_score_report, professional_score_report


# ── Symbol Resolver ───────────────────────────────────────────────────────────

def resolve_symbol(raw: str) -> str:
    """
    Auto-detect Indian stocks and append .NS suffix if needed.
    Tries .NS first, then .BO, then returns raw for US stocks.
    """
    raw = raw.strip().upper()

    if "." in raw or raw.startswith("^"):
        return raw

    for suffix in [".NS", ".BO"]:
        ticker = raw + suffix
        try:
            test = yf.download(ticker, period="5d", progress=False, timeout=10)
            if not test.empty:
                print(f"  → Resolved to {ticker} (Indian NSE/BSE)")
                return ticker
        except Exception:
            pass
        time.sleep(0.3)

    return raw


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(char="─", width=70):
    print(char * width)


def _safe(fn, *args, **kwargs):
    """Call fn safely; return error dict on exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)}


def _print_signal_banner(score: int, signal: str, grade: str):
    """Print professional signal banner."""
    color = signal_color(signal)
    
    if signal == "STRONG BUY":
        emoji = "🚀"
        bar = "█" * 50 + "░" * 0
    elif signal == "BUY":
        emoji = "📈"
        bar = "█" * 35 + "░" * 15
    elif signal == "HOLD":
        emoji = "➡️"
        bar = "█" * 25 + "░" * 25
    elif signal == "SELL":
        emoji = "📉"
        bar = "█" * 15 + "░" * 35
    else:
        emoji = "🔴"
        bar = "░" * 50
    
    _sep("═")
    print(f"\n  {emoji} FINAL VERDICT: {color}{signal}{RESET}")
    print(f"     Score: {score}/100 | Grade: {grade}")
    print(f"     Confidence: [{bar}] {score}%")
    _sep("═")


# ── Main Analysis ─────────────────────────────────────────────────────────────

def run_analysis(symbol: str, verbose: bool = True):
    """Run full professional analysis on a single symbol."""

    print(f"\n⏳ Fetching data for {symbol}...")
    df = get_stock_data(symbol, verbose=verbose)

    if df is None or df.empty:
        print(f"❌ Could not fetch data for {symbol}. Check the symbol and try again.")
        print("   Examples: RELIANCE.NS  TCS.NS  AAPL  MSFT  NIFTY50.NS")
        return

    price      = float(df["Close"].iloc[-1])
    bars       = len(df)
    date_range = f"{df.index[0].date()} → {df.index[-1].date()}"

    print(f"✅ {symbol} | {bars} bars | {date_range}")
    print(f"   Current Price: {fmt_price(price)}\n")

    # ═══════════════════════════════════════════════════════════════════
    # 1. TRADITIONAL TECHNICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print_header(f"① Technical Analysis — {symbol}")

    trend      = _safe(detect_trend,        df)
    trend_full = _safe(trend_analysis,      df)
    breakout   = _safe(breakout_signal,     df)
    bo_full    = _safe(breakout_analysis,   df)
    support, resistance = _safe(get_support_resistance, df) or (None, None)
    sr_zones   = _safe(get_sr_zones,        df)
    poc        = _safe(volume_profile,      df)
    vol_full   = _safe(volume_profile_full, df)
    momentum   = _safe(momentum_report,     df)
    mas        = _safe(ma_signal,           df)
    st         = _safe(supertrend_signal,   df)
    volatility = _safe(volatility_report,   df)
    vwap_info  = _safe(vwap_bands,         df)
    fib        = _safe(fibonacci_retracements, df)
    ichi       = _safe(ichimoku_signal,     df)

    # Trend
    _sep()
    print(f"  📈 TREND          : {trend}")
    if isinstance(trend_full, dict) and "error" not in trend_full:
        print(f"     Strength       : {trend_full.get('strength','')}")
        print(f"     ADX            : {trend_full.get('adx','')}")
        tf = trend_full.get("timeframes", {})
        print(f"     Short/Med/Long : {tf.get('short_term','')} / "
              f"{tf.get('medium_term','')} / {tf.get('long_term','')}")

    # Breakout
    _sep()
    bo_icon = "✅" if breakout else "❌"
    print(f"  🚀 BREAKOUT       : {bo_icon} {bo_full.get('signal','') if isinstance(bo_full,dict) else ''}")
    if isinstance(bo_full, dict) and "error" not in bo_full:
        print(f"     Score          : {bo_full.get('score',0)}/100")
        for sig in bo_full.get("signals_hit", []):
            print(f"     • {sig}")

    # Support / Resistance
    _sep()
    print(f"  🟢 SUPPORT        : {fmt_price(support) if support else 'N/A'}")
    print(f"  🔴 RESISTANCE     : {fmt_price(resistance) if resistance else 'N/A'}")
    if isinstance(sr_zones, dict) and "risk_reward" in sr_zones and sr_zones["risk_reward"]:
        print(f"     Risk/Reward    : {sr_zones['risk_reward']}x")
    if isinstance(sr_zones, dict) and "pivot_points" in sr_zones:
        pp = sr_zones["pivot_points"]
        print(f"     Pivot PP       : {pp.get('PP','')}")
        print(f"     R1/R2/R3       : {pp.get('R1','')}/{pp.get('R2','')}/{pp.get('R3','')}")
        print(f"     S1/S2/S3       : {pp.get('S1','')}/{pp.get('S2','')}/{pp.get('S3','')}")

    # Volume Profile
    _sep()
    print(f"  📊 VOLUME POC     : {fmt_price(poc) if poc else 'N/A'}")
    if isinstance(vol_full, dict) and "error" not in vol_full:
        print(f"     VAH/VAL        : {vol_full.get('VAH','')} / {vol_full.get('VAL','')}")
        print(f"     Position       : {vol_full.get('price_position','')}")

    # Momentum
    _sep()
    if isinstance(momentum, dict) and "error" not in momentum:
        print(f"  ⚡ MOMENTUM       : {momentum.get('signal','')}")
        print(f"     Bulls          : {momentum.get('bull_count','')}")
        inds = momentum.get("indicators", {})
        print(f"     RSI14          : {inds.get('RSI_14','')}")
        print(f"     MACD hist      : {inds.get('MACD_hist','')}")
        print(f"     Stoch K        : {inds.get('Stoch_K','')}")
        print(f"     MFI            : {inds.get('MFI_14','')}")

    # Moving Averages
    _sep()
    if isinstance(mas, dict) and "error" not in mas:
        print(f"  📉 MA SIGNAL      : {mas.get('signal','')}")
        print(f"     Bull Count     : {mas.get('bull_count','')}")
        print(f"     Golden Cross   : {mas.get('golden_cross','')}")
        print(f"     Death Cross    : {mas.get('death_cross','')}")

    # Supertrend
    _sep()
    if isinstance(st, dict) and "error" not in st:
        print(f"  🔁 SUPERTREND     : {st.get('direction','')}")
        print(f"     Line           : {fmt_price(st.get('supertrend_line',0))}")
        print(f"     Distance       : {st.get('distance_pct','')}")
        for flip in st.get("recent_flips", [])[:2]:
            print(f"     Flip           : {flip.get('date','')} → {flip.get('direction','')}")

    # VWAP
    _sep()
    if isinstance(vwap_info, dict) and "error" not in vwap_info:
        print(f"  💧 VWAP           : {fmt_price(vwap_info.get('vwap',0))}")
        print(f"     Position       : {vwap_info.get('position','')}")
        bands = vwap_info.get("bands", {})
        if bands:
            print(f"     ±1σ Band       : {bands.get('lower_1.0sd','')} – {bands.get('upper_1.0sd','')}")

    # Fibonacci
    _sep()
    if isinstance(fib, dict) and "error" not in fib:
        print(f"  🌀 FIBONACCI      :")
        print(f"     Swing High     : {fmt_price(fib.get('swing_high',0))}")
        print(f"     Swing Low      : {fmt_price(fib.get('swing_low',0))}")
        print(f"     Golden (61.8%) : {fmt_price(fib.get('golden_ratio_level',0))}")
        print(f"     Price Zone     : {fib.get('price_zone','')}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. NEW: RSI DIVERGENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print_header(f"② RSI Divergence — {symbol}")
    
    rsi_div = _safe(detect_rsi_divergence, df)
    
    if isinstance(rsi_div, dict) and "error" not in rsi_div:
        _sep()
        print(f"  📊 RSI VALUE      : {rsi_div.get('current_rsi', 'N/A')}")
        print(f"  🎯 CONTEXT        : {rsi_div.get('context', 'N/A')}")
        
        latest_div = rsi_div.get("latest_divergence")
        if latest_div:
            print(f"  🔍 LATEST DIVERGENCE:")
            print(f"     Type           : {latest_div.get('type', 'N/A')}")
            print(f"     Direction      : {latest_div.get('direction', 'N/A')}")
            print(f"     Strength       : {latest_div.get('strength', 0)*100:.0f}%")
            print(f"     Description    : {latest_div.get('description', 'N/A')}")
        else:
            print(f"  🔍 DIVERGENCE     : No divergence detected")
        
        print(f"  🎯 SIGNAL         : {rsi_div.get('signal', 'NEUTRAL')}")
        print(f"     Score          : {rsi_div.get('signal_score', 50)}/100")
    else:
        print(f"  ⚠️ RSI Divergence analysis unavailable")

    # ═══════════════════════════════════════════════════════════════════
    # 3. NEW: CANDLESTICK PATTERN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print_header(f"③ Candlestick Patterns — {symbol}")
    
    candle = _safe(detect_candlestick_patterns, df)
    
    if isinstance(candle, dict) and "error" not in candle:
        _sep()
        if candle.get("bullish_count", 0) > 0:
            print(f"  📈 BULLISH PATTERNS ({candle.get('bullish_count', 0)}):")
            for p in candle.get("bullish_patterns", [])[-3:]:
                print(f"     • {p['pattern']} on {p['date']}: {p['description']}")
        
        if candle.get("bearish_count", 0) > 0:
            print(f"  📉 BEARISH PATTERNS ({candle.get('bearish_count', 0)}):")
            for p in candle.get("bearish_patterns", [])[-3:]:
                print(f"     • {p['pattern']} on {p['date']}: {p['description']}")
        
        if candle.get("bullish_count", 0) == 0 and candle.get("bearish_count", 0) == 0:
            print(f"  ⚪ No significant patterns detected in recent candles.")
        
        latest = candle.get("latest_pattern")
        if latest:
            print(f"\n  🎯 LATEST SIGNAL  : {latest['type']} — {latest['pattern']}")
        
        print(f"  📊 SIGNAL SCORE   : {candle.get('signal_score', 50)}/100")
    else:
        print(f"  ⚠️ Candlestick analysis unavailable")

    # ═══════════════════════════════════════════════════════════════════
    # 4. NEW: VOLUME ANALYSIS (OBV, CMF, Spikes)
    # ═══════════════════════════════════════════════════════════════════
    print_header(f"④ Volume Analysis — {symbol}")
    
    vol_ana = _safe(volume_analysis, df)
    
    if isinstance(vol_ana, dict) and "error" not in vol_ana:
        _sep()
        print(f"  📊 OBV (On-Balance Volume):")
        print(f"     Current          : {vol_ana.get('obv', 'N/A'):,.0f}")
        print(f"     10-day Change    : {vol_ana.get('obv_change_10d', 0):+.1f}%")
        print(f"     Trend            : {vol_ana.get('obv_trend', 'N/A')}")
        
        print(f"\n  💰 MONEY FLOW (CMF):")
        print(f"     CMF Value        : {vol_ana.get('cmf', 'N/A')}")
        print(f"     Signal          : {vol_ana.get('cmf_signal', 'N/A')}")
        
        print(f"\n  📈 VOLUME METRICS:")
        print(f"     Current vs 20d   : {vol_ana.get('volume_ratio_20d', 1):.2f}x")
        print(f"     Position         : {vol_ana.get('volume_vs_avg', 'N/A')} average")
        
        print(f"\n  🔗 PRICE-VOLUME ALIGNMENT:")
        print(f"     Status           : {vol_ana.get('alignment', 'N/A')}")
        print(f"     10d Price Change : {vol_ana.get('price_change_10d', 0):+.2f}%")
        
        spike = vol_ana.get("latest_spike")
        if spike:
            print(f"\n  🔥 LATEST VOLUME SPIKE:")
            print(f"     Date            : {spike.get('date', 'N/A')}")
            print(f"     Ratio           : {spike.get('ratio', 1):.1f}x average")
            print(f"     Strength        : {spike.get('strength', 'N/A')}")
            print(f"     Price Action    : {spike.get('price_action', 'N/A')} ({spike.get('price_change', 0):+.2f}%)")
        
        print(f"\n  🎯 VOLUME SIGNAL  : {vol_ana.get('signal', 'N/A')} (Score: {vol_ana.get('signal_score', 50)})")
    else:
        print(f"  ⚠️ Volume analysis unavailable")

    # ═══════════════════════════════════════════════════════════════════
    # 5. NEW: CONFLUENCE SCORING
    # ═══════════════════════════════════════════════════════════════════
    print_header(f"⑤ Confluence Analysis — {symbol}")
    
    confluence = _safe(ConfluenceEngine().analyze, df)
    
    if isinstance(confluence, dict) and "error" not in confluence:
        _sep()
        print(f"  🎯 CONFLUENCE SCORE: {confluence.get('total_score', 50)}/100")
        print(f"  📊 SIGNAL          : {confluence.get('signal', 'N/A')}")
        print(f"  💡 DESCRIPTION     : {confluence.get('signal_description', 'N/A')}")
        print(f"  📈 CONFIDENCE     : {confluence.get('confidence', 0)}%")
        print(f"  ✅ BULLISH COUNT  : {confluence.get('bullish_indicators', 0)}/8 indicators")
        
        print(f"\n  📋 TOP COMPONENTS:")
        components = confluence.get("components", {})
        sorted_comps = sorted(components.items(), key=lambda x: x[1]["score"], reverse=True)[:4]
        for name, data in sorted_comps:
            bar_len = int(data["score"] / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"     {name:<20} {data['score']:5.1f}/100 [{bar}]")
    else:
        print(f"  ⚠️ Confluence analysis unavailable")

    # ═══════════════════════════════════════════════════════════════════
    # 6. AI ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print_header(f"⑥ AI Analysis — {symbol}")

    prediction = _safe(predict_price, df, 5)
    ai         = _safe(ai_signal,     df)
    score      = _safe(stock_score,   df)
    score      = score if isinstance(score, int) else 0
    full_report = _safe(detailed_score_report, df)
    prof_report = _safe(professional_score_report, df)

    # Price Prediction
    _sep()
    if isinstance(prediction, dict) and "error" not in prediction:
        change = prediction.get("change_percent", 0)
        color  = "\033[92m" if change > 0 else "\033[91m"
        print(f"  🔮 5-DAY PREDICTION:")
        print(f"     Current        : {fmt_price(prediction.get('current_price',0))}")
        print(f"     Predicted      : {fmt_price(prediction.get('predicted_price_5d',0))}")
        print(f"     Change         : {color}{fmt_pct(change)}{RESET} {prediction.get('direction','')}")
        print(f"     Confidence     : {prediction.get('confidence','')}")
        print(f"     Linear/Poly/WMA: {prediction.get('model_forecasts',{}).get('linear','')} / "
              f"{prediction.get('model_forecasts',{}).get('polynomial','')} / "
              f"{prediction.get('model_forecasts',{}).get('wma','')}")

    # AI Signal
    _sep()
    if isinstance(ai, dict) and "error" not in ai:
        sig_label = ai.get("signal", "HOLD")
        color = signal_color(sig_label)
        print(f"  🤖 AI SIGNAL      : {color}{sig_label}{RESET}")
        print(f"     Signal Score   : {ai.get('score',0)}/100")
        inds = ai.get("indicators", {})
        print(f"     RSI            : {inds.get('RSI','')}")
        print(f"     MACD Hist      : {inds.get('MACD_histogram','')}")
        print(f"     BB Position    : {inds.get('BB_position_pct','')}%")
        print(f"     Vol Ratio      : {inds.get('volume_ratio','')}x")
        print()
        print("     Reasoning:")
        for reason in ai.get("reasons", []):
            print(f"       • {reason}")

    # Score Breakdown
    _sep()
    if isinstance(full_report, dict) and "error" not in full_report:
        total = full_report.get("total_score", 0)
        grade = full_report.get("grade", "N/A")
        color = signal_color(score_to_signal(total))
        print(f"  📊 AI SCORE       : {color}{total}/100 — Grade {grade}{RESET}")
        
        # Show risk/reward
        rr = full_report.get("risk_reward", {})
        if rr:
            print(f"\n  📐 RISK/REWARD:")
            print(f"     Support        : {fmt_price(rr.get('support', 0))}")
            print(f"     Resistance     : {fmt_price(rr.get('resistance', 0))}")
            print(f"     R/R Ratio      : {rr.get('risk_reward_ratio', 0)}:1")
            print(f"     Recommendation : {rr.get('recommendation', 'N/A')}")
        
        breakdown = full_report.get("breakdown", {})
        print(f"\n  📋 COMPONENT BREAKDOWN:")
        for component, data in breakdown.items():
            sc = data.get("score", 0)
            notes = ", ".join(data.get("notes", [])[:1])
            print(f"     {component:<25}: {sc:5.1f}  {notes}")
    else:
        color = signal_color(score_to_signal(score))
        print(f"  📊 AI SCORE       : {color}{score}/100{RESET}")

    # ── Smart Money Analysis ──────────────────────────────────────────
    print_header(f"⑦ Smart Money — {symbol}")
    sm = _safe(smart_money_analysis, symbol)
    if isinstance(sm, dict) and "error" not in sm:
        _sep()
        print(f"  🏦 Smart Money Bias: {sm.get('smart_money_bias', 'N/A')}")
        print(f"     Score           : {sm.get('score', 50)}/100")
        da = sm.get("delivery_analysis", {})
        if "error" not in da:
            print(f"\n  📦 Delivery/Volume Analysis:")
            print(f"     Volume Ratio   : {da.get('volume_ratio', 1):.2f}x")
            print(f"     Status         : {da.get('volume_status', 'N/A')}")
            print(f"     Signal         : {da.get('signal', 'N/A')}")
    else:
        print(f"  ⚠️ Smart money data unavailable (requires premium data feed)")

    # ── Final Verdict ───────────────────────────────────────────────
    _sep("═")
    
    # Determine overall signal from all sources
    signals = []
    if isinstance(full_report, dict) and "error" not in full_report:
        signals.append(full_report.get("total_score", 50))
    if isinstance(confluence, dict) and "error" not in confluence:
        signals.append(confluence.get("total_score", 50))
    if isinstance(rsi_div, dict) and "error" not in rsi_div:
        signals.append(rsi_div.get("signal_score", 50))
    if isinstance(vol_ana, dict) and "error" not in vol_ana:
        signals.append(vol_ana.get("signal_score", 50))
    
    avg_signal = sum(signals) / len(signals) if signals else 50
    verdict    = score_to_signal(int(avg_signal))
    grade      = score_to_grade(int(avg_signal))
    
    print(f"\n  ★  AGGREGATE SCORE : {int(avg_signal)}/100")
    print(f"  ★  FINAL VERDICT    : {signal_color(verdict)}{verdict}{RESET} (Grade: {grade})\n")
    
    # Print confluence signals
    print(f"  📊 Signal Sources:")
    print(f"     AI Score        : {full_report.get('total_score', score) if isinstance(full_report, dict) else score}/100")
    print(f"     Confluence      : {confluence.get('total_score', 50) if isinstance(confluence, dict) else 50}/100")
    print(f"     RSI Divergence  : {rsi_div.get('signal_score', 50) if isinstance(rsi_div, dict) else 50}/100")
    print(f"     Volume          : {vol_ana.get('signal_score', 50) if isinstance(vol_ana, dict) else 50}/100")
    
    _sep("═")

    # ── Latest Quote ───────────────────────────────────────────────
    print()
    quote = _safe(get_latest_price, symbol)
    if isinstance(quote, dict) and "error" not in quote:
        ch    = quote.get("change", 0)
        pct   = quote.get("change_pct", 0)
        color = "\033[92m" if ch >= 0 else "\033[91m"
        print(f"  Live Quote  : {fmt_price(quote.get('price',0))}  "
              f"{color}{ch:+.2f} ({pct:+.2f}%){RESET}  "
              f"| {quote.get('trend','')}  "
              f"| Vol: {fmt_volume(quote.get('volume',0))}")
        print(f"  Day Range   : {fmt_price(quote.get('low',0))} – {fmt_price(quote.get('high',0))}")
        print(f"  Date        : {quote.get('date','')}")
    print()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 70)
    print("  🚀 Ultimate Stock AI — Professional Trading Terminal")
    print("  Features: RSI Divergence | Candlestick | Volume | Confluence | MTF")
    print("  Supports: NSE (RELIANCE.NS) | BSE (.BO) | US (AAPL)")
    print("═" * 70)

    raw = input("\nEnter Stock Symbol: ").strip()
    if not raw:
        print("No symbol entered. Exiting.")
        return

    symbol = resolve_symbol(raw)
    run_analysis(symbol, verbose=True)


if __name__ == "__main__":
    main()
