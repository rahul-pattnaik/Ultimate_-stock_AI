# main_terminal.py
# ─────────────────────────────────────────────────────────────────────────────
# Ultimate Stock AI — Main Terminal
# Fixed: yfinance MultiIndex columns, Indian stock suffix auto-detection,
#        graceful error handling, coloured output, full signal display
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

from ai.price_prediction   import predict_price
from ai.signal_model       import ai_signal
from ai.ranking_engine     import stock_score, detailed_score_report


# ── Symbol Resolver ───────────────────────────────────────────────────────────

def resolve_symbol(raw: str) -> str:
    """
    Auto-detect Indian stocks and append .NS suffix if needed.
    Tries .NS first, then .BO, then returns raw for US stocks.
    """
    raw = raw.strip().upper()

    # Already has exchange suffix or is an index
    if "." in raw or raw.startswith("^"):
        return raw

    # Try NSE suffix first (most liquid Indian exchange)
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

    # Assume US stock
    return raw


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(char="─", width=60):
    print(char * width)


def _safe(fn, *args, **kwargs):
    """Call fn safely; return error dict on exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)}


def _print_dict(d: dict, indent=2):
    """Pretty-print a dict, skipping None/error values."""
    pad = " " * indent
    for k, v in d.items():
        if k == "error":
            print(f"{pad}⚠️  {v}")
        elif isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent + 2)
        elif isinstance(v, list):
            print(f"{pad}{k}: {v}")
        else:
            print(f"{pad}{k:<28}: {v}")


# ── Main Analysis ─────────────────────────────────────────────────────────────

def run_analysis(symbol: str, verbose: bool = True):
    """Run full analysis on a single symbol."""

    # ── 1. Fetch Data ─────────────────────────────────────────────────
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

    # ── 2. Technical Analysis ─────────────────────────────────────────
    print_header(f"Technical Analysis — {symbol}")

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
        print(f"     Slope          : {trend_full.get('slope_pct','')}% per bar")
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

    # Volatility
    _sep()
    if isinstance(volatility, dict) and "error" not in volatility:
        print(f"  🌊 VOLATILITY     : {volatility.get('volatility_regime','')}")
        print(f"     ATR14          : {volatility.get('ATR_14','')}")
        print(f"     HV20           : {volatility.get('hist_vol_20d','')}")
        print(f"     Z-Score        : {volatility.get('z_score','')} ({volatility.get('z_label','')})")
        print(f"     Keltner Pos    : {volatility.get('keltner_position','')}")

    # VWAP
    _sep()
    if isinstance(vwap_info, dict) and "error" not in vwap_info:
        print(f"  💧 VWAP           : {fmt_price(vwap_info.get('vwap',0))}")
        print(f"     Position       : {vwap_info.get('position','')}")
        print(f"     Distance       : {vwap_info.get('distance_pct','')}")
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

    # Ichimoku
    _sep()
    if isinstance(ichi, dict) and "error" not in ichi:
        print(f"  ☁️  ICHIMOKU       : {ichi.get('signal','')}")
        print(f"     Bull Conditions: {ichi.get('bull_conditions','')}")
        print(f"     Cloud Position : {ichi.get('cloud_position','')}")
        print(f"     TK Cross       : {ichi.get('tk_cross','')}")

    # ── 3. AI Analysis ────────────────────────────────────────────────
    print_header(f"AI Analysis — {symbol}")

    prediction = _safe(predict_price, df, 5)
    ai         = _safe(ai_signal,     df)
    score      = _safe(stock_score,   df)
    score      = score if isinstance(score, int) else 0
    full_report = _safe(detailed_score_report, df)

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
    else:
        print(f"  🔮 Prediction      : {prediction}")

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
    else:
        print(f"  🤖 AI Signal       : {ai}")

    # Score Breakdown
    _sep()
    if isinstance(full_report, dict) and "error" not in full_report:
        total = full_report.get("total_score", 0)
        grade = full_report.get("grade", "N/A")
        color = signal_color(score_to_signal(total))
        print(f"  📊 AI SCORE       : {color}{total}/100 — Grade {grade}{RESET}")
        breakdown = full_report.get("breakdown", {})
        for component, data in breakdown.items():
            sc = data.get("score", 0)
            notes = ", ".join(data.get("notes", []))
            print(f"     {component:<22}: {sc:5.1f}  {notes}")
    else:
        color = signal_color(score_to_signal(score))
        print(f"  📊 AI SCORE       : {color}{score}/100{RESET}")

    # ── 4. Final Verdict ─────────────────────────────────────────────
    _sep("═")
    verdict = score_to_signal(score)
    grade   = score_to_grade(score)
    color   = signal_color(verdict)
    print(f"\n  ★  FINAL VERDICT  : {color}{verdict}  (Score: {score}/100 | Grade: {grade}){RESET}\n")
    _sep("═")

    # ── 5. Latest Quote ───────────────────────────────────────────────
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
    print("\n" + "═" * 60)
    print("  🚀 Ultimate Stock AI — Terminal Analyzer")
    print("  Supports: NSE (RELIANCE.NS) | BSE (.BO) | US (AAPL)")
    print("  Just type the bare name for Indian stocks (e.g. ITC)")
    print("═" * 60)

    raw = input("\nEnter Stock Symbol: ").strip()
    if not raw:
        print("No symbol entered. Exiting.")
        return

    symbol = resolve_symbol(raw)
    run_analysis(symbol, verbose=True)


if __name__ == "__main__":
    main()
