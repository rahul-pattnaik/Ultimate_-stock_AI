# main_terminal.py
# ─────────────────────────────────────────────────────────────────────────────
# Ultimate Stock AI — Full-Featured Terminal
# Uses EVERY available module across all packages:
#   technical  · ai · fundamentals · intelligence · options
#   portfolio  · scanners · core (backtester + nse_universe)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np

# ── Core ──────────────────────────────────────────────────────────────────────
from core.data_fetcher  import get_stock_data, get_latest_price
from core.utils         import (fmt_price, fmt_pct, fmt_volume,
                                 score_to_signal, score_to_grade,
                                 signal_color, RESET, print_header)
from core.backtester    import Backtester
from core.nse_universe  import get_nifty50

# ── Technical ─────────────────────────────────────────────────────────────────
from technical.trend_detection    import detect_trend, trend_analysis
from technical.breakout           import breakout_signal, breakout_analysis
from technical.support_resistance import get_support_resistance, get_sr_zones
from technical.volume_profile     import volume_profile, volume_profile_full
from technical.momentum           import momentum_report
from technical.moving_averages    import ma_signal
from technical.supertrend         import supertrend_signal
from technical.volatility         import volatility_report
from technical.vwap               import vwap_bands
from technical.fibonacci          import fibonacci_retracements
from technical.ichimoku           import ichimoku_signal
from technical.elliott_wave       import detect_elliott_wave          # ← NEW

# ── AI ────────────────────────────────────────────────────────────────────────
from ai.price_prediction    import predict_price
from ai.signal_model        import ai_signal
from ai.ranking_engine      import stock_score, detailed_score_report
from ai.lstm_predictor      import lstm_predict                       # ← NEW
from ai.random_forest_signal import rf_signal                         # ← NEW
from ai.reinforcement_agent import TradingAgent                       # ← NEW
from ai.sentiment_model     import analyze_news                       # ← NEW

# ── Fundamentals ─────────────────────────────────────────────────────────────
from fundamentals.financials      import FinancialDataHandler         # ← NEW
from fundamentals.ratios          import TechnicalIndicators, RatioAnalyzer  # ← NEW
from fundamentals.valuation       import ValuationEngine              # ← NEW
from fundamentals.peer_comparison import PeerComparator               # ← NEW

# ── Intelligence ─────────────────────────────────────────────────────────────
from intelligence.block_deals     import detect_block_deal            # ← NEW
from intelligence.insider_trading import insider_signal               # ← NEW
from intelligence.promoter_pledge import pledge_risk                  # ← NEW
from intelligence.fii_dii         import analyze_fii_dii              # ← NEW (when FII data available)

# ── Options ───────────────────────────────────────────────────────────────────
from options.options_strategy_builder import bull_call_spread         # ← NEW

# ── Portfolio ─────────────────────────────────────────────────────────────────
from portfolio.optimizer   import (optimize_portfolio, max_drawdown,  # ← NEW
                                    sharpe_ratio, sortino_ratio)
from portfolio.sharpe_ratio  import sharpe_ratio  as sharpe_standalone
from portfolio.sortino_ratio import sortino_ratio as sortino_standalone

# ── Scanners ──────────────────────────────────────────────────────────────────
from scanners.breakout_scanner  import BreakoutScanner                # ← NEW
from scanners.momentum_scanner  import MomentumScanner                # ← NEW
from scanners.swing_scanner     import SwingScanner                   # ← NEW
from scanners.value_scanner     import ValueScanner                   # ← NEW


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

MENU = """
╔══════════════════════════════════════════════════════════╗
║       🚀 Ultimate Stock AI — Full Terminal               ║
║  NSE: RELIANCE.NS  BSE: TCS.BO  US: AAPL  Index: ^NSEI  ║
╠══════════════════════════════════════════════════════════╣
║  [1]  Full Analysis (all modules)                        ║
║  [2]  Technical Analysis only                            ║
║  [3]  AI Signals & Predictions                           ║
║  [4]  Elliott Wave + Fibonacci                           ║
║  [5]  Scanner Suite (Breakout/Momentum/Swing/Value)      ║
║  [6]  Backtest (AI signal strategy)                      ║
║  [7]  Portfolio Optimizer (Nifty 50 basket)              ║
║  [8]  Options Strategy Builder                           ║
║  [9]  Intelligence Dashboard (Insider/FII/Block deals)   ║
║  [10] RL Agent Recommendation                            ║
║  [11] News Sentiment Analyzer                            ║
║  [0]  Exit                                               ║
╚══════════════════════════════════════════════════════════╝
"""


def _sep(char="─", width=62):
    print(char * width)


def _safe(fn, *args, **kwargs):
    """Call fn safely; return error dict on exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)}


def _h(title: str):
    """Section header."""
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print('═' * 62)


def resolve_symbol(raw: str) -> str:
    raw = raw.strip().upper()
    if "." in raw or raw.startswith("^"):
        return raw
    for suffix in [".NS", ".BO"]:
        ticker = raw + suffix
        try:
            test = yf.download(ticker, period="5d", progress=False, timeout=10)
            if not test.empty:
                print(f"  → Resolved to {ticker}")
                return ticker
        except Exception:
            pass
        time.sleep(0.3)
    return raw


def fetch(symbol: str):
    """Fetch OHLCV data and print basic info."""
    print(f"\n⏳ Fetching {symbol}...")
    df = get_stock_data(symbol, verbose=False)
    if df is None or df.empty:
        print(f"❌ Could not fetch data for {symbol}.")
        return None
    price = float(df["Close"].iloc[-1])
    print(f"✅ {symbol}  {len(df)} bars  "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    print(f"   Current Price: {fmt_price(price)}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Technical Analysis  (all 12 technical indicators)
# ══════════════════════════════════════════════════════════════════════════════

def section_technical(symbol: str, df: pd.DataFrame):
    _h(f"📈 Technical Analysis — {symbol}")

    price = float(df["Close"].iloc[-1])

    trend       = _safe(detect_trend, df)
    trend_full  = _safe(trend_analysis, df)
    breakout    = _safe(breakout_signal, df)
    bo_full     = _safe(breakout_analysis, df)
    sr          = _safe(get_support_resistance, df)
    support, resistance = sr if isinstance(sr, tuple) else (None, None)
    sr_zones    = _safe(get_sr_zones, df)
    poc         = _safe(volume_profile, df)
    vol_full    = _safe(volume_profile_full, df)
    momentum    = _safe(momentum_report, df)
    mas         = _safe(ma_signal, df)
    st          = _safe(supertrend_signal, df)
    volatility  = _safe(volatility_report, df)
    vwap_info   = _safe(vwap_bands, df)
    fib         = _safe(fibonacci_retracements, df)
    ichi        = _safe(ichimoku_signal, df)

    # Trend
    _sep()
    print(f"  📈 TREND          : {trend}")
    if isinstance(trend_full, dict) and "error" not in trend_full:
        tf = trend_full.get("timeframes", {})
        print(f"     Strength       : {trend_full.get('strength', '')}")
        print(f"     ADX            : {trend_full.get('adx', '')}")
        print(f"     Slope          : {trend_full.get('slope_pct', '')}% per bar")
        print(f"     Short/Med/Long : {tf.get('short_term','')} / "
              f"{tf.get('medium_term','')} / {tf.get('long_term','')}")

    # Breakout
    _sep()
    bo_icon = "✅" if breakout else "❌"
    print(f"  🚀 BREAKOUT       : {bo_icon} "
          f"{bo_full.get('signal','') if isinstance(bo_full,dict) else ''}")
    if isinstance(bo_full, dict) and "error" not in bo_full:
        print(f"     Score          : {bo_full.get('score', 0)}/100")
        for sig in bo_full.get("signals_hit", []):
            print(f"     • {sig}")

    # Support / Resistance
    _sep()
    print(f"  🟢 SUPPORT        : {fmt_price(support) if support else 'N/A'}")
    print(f"  🔴 RESISTANCE     : {fmt_price(resistance) if resistance else 'N/A'}")
    if isinstance(sr_zones, dict):
        if sr_zones.get("risk_reward"):
            print(f"     Risk/Reward    : {sr_zones['risk_reward']}x")
        pp = sr_zones.get("pivot_points", {})
        if pp:
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
        inds = momentum.get("indicators", {})
        print(f"  ⚡ MOMENTUM       : {momentum.get('signal','')}")
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
        print(f"     Line           : {fmt_price(st.get('supertrend_line', 0))}")
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
        print(f"  💧 VWAP           : {fmt_price(vwap_info.get('vwap', 0))}")
        print(f"     Position       : {vwap_info.get('position','')}")
        print(f"     Distance       : {vwap_info.get('distance_pct','')}")
        bands = vwap_info.get("bands", {})
        if bands:
            print(f"     ±1σ Band       : {bands.get('lower_1.0sd','')} – {bands.get('upper_1.0sd','')}")

    # Fibonacci
    _sep()
    if isinstance(fib, dict) and "error" not in fib:
        print(f"  🌀 FIBONACCI      :")
        print(f"     Swing High     : {fmt_price(fib.get('swing_high', 0))}")
        print(f"     Swing Low      : {fmt_price(fib.get('swing_low', 0))}")
        print(f"     Golden (61.8%) : {fmt_price(fib.get('golden_ratio_level', 0))}")
        print(f"     Price Zone     : {fib.get('price_zone','')}")

    # Ichimoku
    _sep()
    if isinstance(ichi, dict) and "error" not in ichi:
        print(f"  ☁️  ICHIMOKU       : {ichi.get('signal','')}")
        print(f"     Bull Conditions: {ichi.get('bull_conditions','')}")
        print(f"     Cloud Position : {ichi.get('cloud_position','')}")
        print(f"     TK Cross       : {ichi.get('tk_cross','')}")

    return price


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — AI Signals & Predictions (5 AI modules)
# ══════════════════════════════════════════════════════════════════════════════

def section_ai(symbol: str, df: pd.DataFrame):
    _h(f"🤖 AI Signals & Predictions — {symbol}")

    # ── Classic price prediction (linear/poly/WMA ensemble) ──────────
    _sep()
    print("  🔮 5-DAY ENSEMBLE PREDICTION (Linear + Poly + WMA)")
    prediction = _safe(predict_price, df, 5)
    if isinstance(prediction, dict) and "error" not in prediction:
        change = prediction.get("change_percent", 0)
        color  = "\033[92m" if change > 0 else "\033[91m"
        print(f"     Current        : {fmt_price(prediction.get('current_price', 0))}")
        print(f"     Predicted      : {fmt_price(prediction.get('predicted_price_5d', 0))}")
        print(f"     Change         : {color}{fmt_pct(change)}{RESET} {prediction.get('direction','')}")
        print(f"     Confidence     : {prediction.get('confidence','')}")
        mf = prediction.get("model_forecasts", {})
        print(f"     Lin/Poly/WMA   : {mf.get('linear','')} / {mf.get('polynomial','')} / {mf.get('wma','')}")
    else:
        print(f"     ⚠️  {prediction.get('error','') if isinstance(prediction,dict) else prediction}")

    # ── LSTM deep-learning prediction ────────────────────────────────
    _sep()
    print("  🧠 LSTM DEEP-LEARNING PREDICTION")
    if len(df) >= 80:
        lstm = _safe(lstm_predict, df, days_ahead=5)
        if isinstance(lstm, dict) and "error" not in lstm:
            change_l = lstm.get("change_pct", 0)
            color    = "\033[92m" if change_l > 0 else "\033[91m"
            print(f"     Current Price  : {fmt_price(lstm.get('current_price', 0))}")
            preds = lstm.get("predicted_prices", [])
            if preds:
                print(f"     Day forecasts  : {' → '.join(fmt_price(p) for p in preds)}")
            print(f"     Direction      : {color}{lstm.get('direction','')}{RESET}  "
                  f"({fmt_pct(change_l)})")
            tm = lstm.get("training_metrics", {})
            if tm:
                print(f"     Train MAE      : {tm.get('mae','')}")
                print(f"     Train R²       : {tm.get('r2','')}")
        else:
            print(f"     ⚠️  {lstm.get('error','') if isinstance(lstm,dict) else lstm}")
    else:
        print("     ⚠️  Not enough data for LSTM (need ≥ 80 bars).")

    # ── Random Forest signal ──────────────────────────────────────────
    _sep()
    print("  🌲 RANDOM FOREST SIGNAL")
    rf = _safe(rf_signal, df)
    if isinstance(rf, dict) and "error" not in rf:
        sig   = rf.get("signal", "")
        color = signal_color("BUY" if "BUY" in sig else "SELL")
        print(f"     Signal         : {color}{sig}{RESET}")
        print(f"     Probability    : {rf.get('probability','')}")
        print(f"     Buy/Sell prob  : {rf.get('buy_prob','')} / {rf.get('sell_prob','')}")
        trn = rf.get("training", {})
        if trn:
            print(f"     Accuracy       : {trn.get('accuracy','')}")
    else:
        print(f"     ⚠️  {rf.get('error','') if isinstance(rf,dict) else rf}")

    # ── Combined AI signal model ──────────────────────────────────────
    _sep()
    print("  🤖 COMBINED AI SIGNAL MODEL")
    ai = _safe(ai_signal, df)
    if isinstance(ai, dict) and "error" not in ai:
        sig_label = ai.get("signal", "HOLD")
        color     = signal_color(sig_label)
        inds      = ai.get("indicators", {})
        print(f"     Signal         : {color}{sig_label}{RESET}")
        print(f"     Score          : {ai.get('score', 0)}/100")
        print(f"     RSI            : {inds.get('RSI','')}")
        print(f"     MACD Hist      : {inds.get('MACD_histogram','')}")
        print(f"     BB Position    : {inds.get('BB_position_pct','')}%")
        print(f"     Vol Ratio      : {inds.get('volume_ratio','')}x")
        for reason in ai.get("reasons", []):
            print(f"     • {reason}")
    else:
        print(f"     ⚠️  {ai.get('error','') if isinstance(ai,dict) else ai}")

    # ── Ranking / score breakdown ─────────────────────────────────────
    _sep()
    print("  📊 AI RANKING SCORE")
    score       = _safe(stock_score, df)
    score       = score if isinstance(score, int) else 0
    full_report = _safe(detailed_score_report, df)
    if isinstance(full_report, dict) and "error" not in full_report:
        total = full_report.get("total_score", 0)
        grade = full_report.get("grade", "N/A")
        color = signal_color(score_to_signal(total))
        print(f"     Score/Grade    : {color}{total}/100 — Grade {grade}{RESET}")
        for component, data in full_report.get("breakdown", {}).items():
            sc    = data.get("score", 0)
            notes = ", ".join(data.get("notes", []))
            print(f"     {component:<22}: {sc:5.1f}  {notes}")
    else:
        color = signal_color(score_to_signal(score))
        print(f"     Score          : {color}{score}/100{RESET}")

    # ── Final verdict ─────────────────────────────────────────────────
    _sep("═")
    verdict = score_to_signal(score)
    grade   = score_to_grade(score)
    color   = signal_color(verdict)
    print(f"\n  ★  FINAL VERDICT  : {color}{verdict}  "
          f"(Score: {score}/100 | Grade: {grade}){RESET}\n")
    _sep("═")
    return score


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Elliott Wave + Fibonacci
# ══════════════════════════════════════════════════════════════════════════════

def section_elliott_fibonacci(symbol: str, df: pd.DataFrame):
    _h(f"🌊 Elliott Wave & Fibonacci — {symbol}")

    # Elliott Wave
    _sep()
    print("  〰️  ELLIOTT WAVE ANALYSIS")
    ew = _safe(detect_elliott_wave, df)
    if isinstance(ew, dict) and "error" not in ew:
        print(f"     Wave Count     : {ew.get('wave_count','')}")
        print(f"     Current Wave   : {ew.get('current_wave','')}")
        print(f"     Pattern Valid  : {ew.get('pattern_valid','')}")
        print(f"     Bias           : {ew.get('bias','')}")
        targets = ew.get("targets", {})
        if targets:
            print(f"     Wave 3 Target  : {targets.get('wave3_target','')}")
            print(f"     Wave 5 Target  : {targets.get('wave5_target','')}")
            print(f"     Correction Tgt : {targets.get('correction_target','')}")
        for note in ew.get("notes", []):
            print(f"     • {note}")
    else:
        print(f"     ⚠️  {ew.get('error','') if isinstance(ew,dict) else ew}")

    # Fibonacci
    _sep()
    print("  🌀 FIBONACCI RETRACEMENTS")
    fib = _safe(fibonacci_retracements, df)
    if isinstance(fib, dict) and "error" not in fib:
        print(f"     Swing High     : {fmt_price(fib.get('swing_high', 0))}")
        print(f"     Swing Low      : {fmt_price(fib.get('swing_low', 0))}")
        print(f"     Current Price  : {fmt_price(float(df['Close'].iloc[-1]))}")
        print(f"     Price Zone     : {fib.get('price_zone','')}")
        print(f"     Golden (61.8%) : {fmt_price(fib.get('golden_ratio_level', 0))}")
        levels = fib.get("levels", {})
        if levels:
            for lvl, val in levels.items():
                print(f"     {lvl:<12}         : {fmt_price(val)}")
    else:
        print(f"     ⚠️  {fib.get('error','') if isinstance(fib,dict) else fib}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Scanner Suite
# ══════════════════════════════════════════════════════════════════════════════

def section_scanners(symbol: str, df: pd.DataFrame):
    _h(f"🔍 Scanner Suite — {symbol}")

    # Breakout Scanner
    _sep()
    print("  🚀 BREAKOUT SCANNER")
    try:
        bs     = BreakoutScanner()
        result = bs.scan_breakout(df, symbol)
        if result:
            print(f"     Type           : {result.breakout_type}")
            print(f"     Signal         : {result.signal}")
            print(f"     Strength       : {getattr(result,'strength','')} / {getattr(result,'score','')} score")
            print(f"     Target         : {fmt_price(getattr(result,'target',0)) if getattr(result,'target',None) else 'N/A'}")
            print(f"     Stop Loss      : {fmt_price(getattr(result,'stop_loss',0)) if getattr(result,'stop_loss',None) else 'N/A'}")
        else:
            print("     No breakout setup detected at this time.")
    except Exception as e:
        print(f"     ⚠️  {e}")

    # Momentum Scanner
    _sep()
    print("  ⚡ MOMENTUM SCANNER")
    try:
        ms     = MomentumScanner()
        result = ms.analyze_momentum(df, symbol)
        if result:
            stars  = "★" * getattr(result, "rating", 0)
            print(f"     Rating         : {stars} ({getattr(result,'rating','')} / 5)")
            print(f"     Signal         : {result.signal}")
            print(f"     RSI            : {getattr(result,'rsi','')}")
            print(f"     MACD           : {getattr(result,'macd','')}")
            print(f"     Volume Surge   : {getattr(result,'volume_surge','')}")
        else:
            print("     No momentum signal detected.")
    except Exception as e:
        print(f"     ⚠️  {e}")

    # Swing Scanner
    _sep()
    print("  🔄 SWING TRADING SCANNER")
    try:
        ss     = SwingScanner()
        result = ss.scan_swing_setup(df, symbol)
        if result:
            print(f"     Signal         : {result.signal}")
            print(f"     Strength       : {getattr(result,'strength',0)*100:.1f}%")
            print(f"     Entry          : {fmt_price(getattr(result,'entry',0))}")
            print(f"     Target         : {fmt_price(getattr(result,'target',0))}")
            print(f"     Stop Loss      : {fmt_price(getattr(result,'stop_loss',0))}")
            rr = getattr(result, "risk_reward", None)
            print(f"     Risk/Reward    : {f'{rr:.2f}x' if rr else 'N/A'}")
        else:
            print("     No swing setup detected.")
    except Exception as e:
        print(f"     ⚠️  {e}")

    # Value Scanner
    _sep()
    print("  💰 VALUE SCANNER (price-derived metrics)")
    try:
        vs     = ValueScanner()
        price  = float(df["Close"].iloc[-1])
        # ValueScanner works with fundamental data; use what we can from price action
        result = _safe(vs.scan_value_stock, symbol, {"price": price, "df": df})
        if isinstance(result, dict) and "error" not in result:
            print(f"     Grade          : {result.get('grade','')}")
            print(f"     Signal         : {result.get('signal','')}")
            print(f"     Score          : {result.get('score','')}")
            print(f"     Intrinsic Val  : {result.get('intrinsic_value','')}")
        elif hasattr(result, "grade"):
            print(f"     Grade          : {result.grade}")
            print(f"     Signal         : {result.signal}")
        else:
            print(f"     ⚠️  {result.get('error','N/A') if isinstance(result,dict) else result}")
    except Exception as e:
        print(f"     ⚠️  {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Backtester
# ══════════════════════════════════════════════════════════════════════════════

def section_backtest(symbol: str, df: pd.DataFrame):
    _h(f"📅 Backtest — AI Signal Strategy — {symbol}")
    print("  Running backtest on the AI signal model strategy...")

    def signal_fn(df_slice: pd.DataFrame) -> str:
        """Use ai_signal as the strategy signal."""
        result = _safe(ai_signal, df_slice)
        if isinstance(result, dict) and "signal" in result:
            return result["signal"]
        return "HOLD"

    try:
        bt     = Backtester(df, signal_fn)
        result = bt.run()
        _sep()
        print(f"  {'Initial Capital':<26}: ₹{result.get('initial_capital', 0):,.0f}")
        print(f"  {'Final Capital':<26}: ₹{result.get('final_capital', 0):,.0f}")
        print(f"  {'Net PnL':<26}: ₹{result.get('net_pnl', 0):,.0f}")
        print(f"  {'Total Return':<26}: {result.get('total_return_pct', 0):.2f}%")
        print(f"  {'CAGR':<26}: {result.get('cagr', 0):.2f}%")
        print(f"  {'Sharpe Ratio':<26}: {result.get('sharpe_ratio', 0):.2f}")
        print(f"  {'Sortino Ratio':<26}: {result.get('sortino_ratio', 0):.2f}")
        print(f"  {'Max Drawdown':<26}: {result.get('max_drawdown_pct', 0):.2f}%")
        print(f"  {'Win Rate':<26}: {result.get('win_rate', 0):.1f}%")
        print(f"  {'Total Trades':<26}: {result.get('total_trades', 0)}")
        print(f"  {'Profit Factor':<26}: {result.get('profit_factor', 0):.2f}")
        # Portfolio-level Sharpe/Sortino from raw equity curve
        equity = result.get("equity_curve", [])
        if len(equity) > 2:
            daily_ret = np.diff(equity) / np.array(equity[:-1])
            sh = sharpe_standalone(daily_ret)
            so = sortino_standalone(daily_ret)
            print(f"  {'Sharpe (portfolio)':<26}: {sh:.4f}")
            print(f"  {'Sortino (portfolio)':<26}: {so:.4f}")
            dd = max_drawdown(np.array(equity))
            print(f"  {'Max Drawdown $':<26}: {dd:.4f}")
        if hasattr(bt, "print_report"):
            bt.print_report(result)
    except Exception as e:
        print(f"  ⚠️  Backtest error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — Portfolio Optimizer
# ══════════════════════════════════════════════════════════════════════════════

def section_portfolio():
    _h("📂 Portfolio Optimizer — Nifty 50 Basket")

    try:
        universe = get_nifty50()
        # Take first 10 for speed
        symbols  = universe[:10] if universe else [
            "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
            "HINDUNILVR.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS","BAJFINANCE.NS"
        ]
        print(f"  Fetching price data for {len(symbols)} stocks...")
        price_dfs = {}
        for sym in symbols:
            df = _safe(get_stock_data, sym, verbose=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                price_dfs[sym] = df
            time.sleep(0.15)

        if len(price_dfs) < 2:
            print("  ⚠️  Not enough data to optimize portfolio.")
            return

        result = optimize_portfolio(price_dfs, method="max_sharpe")
        _sep()
        print(f"  Method         : Max Sharpe Ratio")
        print(f"  Expected Return: {result.get('expected_return', 0)*100:.2f}% p.a.")
        print(f"  Expected Vol   : {result.get('volatility', 0)*100:.2f}% p.a.")
        print(f"  Sharpe Ratio   : {result.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown   : {result.get('max_drawdown', 0)*100:.2f}%")
        print()
        print("  ── Optimal Weights ──────────────────────────────────")
        weights = result.get("weights", {})
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar  = "█" * int(w * 40)
            print(f"     {sym:<18}: {w*100:6.2f}%  {bar}")
    except Exception as e:
        print(f"  ⚠️  Portfolio optimizer error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — Options Strategy Builder
# ══════════════════════════════════════════════════════════════════════════════

def section_options(symbol: str, df: pd.DataFrame):
    _h(f"📋 Options Strategy Builder — {symbol}")

    price = float(df["Close"].iloc[-1])
    _sep()
    print(f"  Current Price  : {fmt_price(price)}")

    # Bull Call Spread
    print("\n  📈 BULL CALL SPREAD (Moderately Bullish)")
    bcs = _safe(bull_call_spread, price)
    if isinstance(bcs, dict) and "error" not in bcs:
        print(f"     Buy Call ATM   : {fmt_price(bcs.get('buy_call', 0))}")
        print(f"     Sell Call OTM  : {fmt_price(bcs.get('sell_call', 0))}")
        spread_width = bcs.get('sell_call', 0) - bcs.get('buy_call', 0)
        print(f"     Spread Width   : {fmt_price(spread_width)}")
        print(f"     Max Profit     : Spread width minus net premium paid")
        print(f"     Max Loss       : Net premium paid")

    # Additional strategies computed manually from price
    _sep()
    print("\n  📉 BEAR PUT SPREAD (Moderately Bearish)")
    buy_put  = price * 0.98
    sell_put = price * 0.95
    print(f"     Buy Put ATM    : {fmt_price(buy_put)}")
    print(f"     Sell Put OTM   : {fmt_price(sell_put)}")
    print(f"     Spread Width   : {fmt_price(buy_put - sell_put)}")

    print("\n  ⚖️  IRON CONDOR (Range Bound / Low Volatility)")
    print(f"     Sell Call OTM  : {fmt_price(price * 1.05)}")
    print(f"     Buy Call OTM+  : {fmt_price(price * 1.08)}")
    print(f"     Sell Put OTM   : {fmt_price(price * 0.95)}")
    print(f"     Buy Put OTM+   : {fmt_price(price * 0.92)}")
    print(f"     Max Profit     : Net premium collected")
    print(f"     Breakevens     : {fmt_price(price*0.95)} — {fmt_price(price*1.05)}")

    print("\n  🔔 STRADDLE (High Volatility Play)")
    atm = price
    print(f"     Buy Call ATM   : {fmt_price(atm)}")
    print(f"     Buy Put ATM    : {fmt_price(atm)}")
    print(f"     Profit if move : > ±{fmt_price(atm * 0.05)} (±5% estimate)")


# ══════════════════════════════════════════════════════════════════════════════
# Section 8 — Intelligence Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def section_intelligence(symbol: str, df: pd.DataFrame):
    _h(f"🕵️  Intelligence Dashboard — {symbol}")

    price      = float(df["Close"].iloc[-1])
    avg_vol    = float(df["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in df else 0
    last_vol   = float(df["Volume"].iloc[-1]) if "Volume" in df else 0

    # Block Deals
    _sep()
    print("  🏦 BLOCK DEAL DETECTOR")
    bd_result = _safe(detect_block_deal, last_vol, avg_vol)
    print(f"     Last Volume    : {fmt_volume(last_vol)}")
    print(f"     Avg Volume(20) : {fmt_volume(avg_vol)}")
    print(f"     Signal         : {bd_result if not isinstance(bd_result,dict) else bd_result}")
    vol_ratio  = last_vol / avg_vol if avg_vol > 0 else 0
    print(f"     Volume Ratio   : {vol_ratio:.2f}x {'🔥 UNUSUAL' if vol_ratio > 2 else ''}")

    # Insider Trading
    _sep()
    print("  👤 INSIDER TRADING SIGNAL")
    print("  (Note: Real insider data requires NSE/BSE feeds.)")
    print("  Using demo estimates for illustration:")
    insider_buy  = 1500000   # demo: ₹1.5Cr buy
    insider_sell = 800000    # demo: ₹0.8Cr sell
    ins_sig = _safe(insider_signal, insider_buy, insider_sell)
    print(f"     Insider Buy    : ₹{insider_buy:,.0f}")
    print(f"     Insider Sell   : ₹{insider_sell:,.0f}")
    print(f"     Net Signal     : {ins_sig if not isinstance(ins_sig,dict) else ins_sig}")

    # Promoter Pledge
    _sep()
    print("  📌 PROMOTER PLEDGE RISK")
    print("  (Note: Requires screener/NSE fundamental feed for real values.)")
    demo_pledge_pct = 18.5   # demo %
    pledge_result   = _safe(pledge_risk, demo_pledge_pct)
    print(f"     Pledge %       : {demo_pledge_pct}% (demo value)")
    print(f"     Risk Level     : {pledge_result if not isinstance(pledge_result,dict) else pledge_result}")

    # FII/DII (requires actual FII DataFrame; show structure)
    _sep()
    print("  📊 FII/DII FLOW (demo)")
    print("  (Requires FII/DII data with FII_Net & DII_Net columns.)")
    print("  Connect to NSE FII/DII feed to get live analysis.")
    print("  Module: intelligence/fii_dii.py → analyze_fii_dii(df)")


# ══════════════════════════════════════════════════════════════════════════════
# Section 9 — RL Agent Recommendation
# ══════════════════════════════════════════════════════════════════════════════

def section_rl_agent(symbol: str, df: pd.DataFrame):
    _h(f"🎮 RL Agent (Q-Learning) — {symbol}")

    print("  Training Q-Learning agent on historical data...")
    try:
        agent  = TradingAgent()
        stats  = agent.train(df, episodes=30)
        _sep()
        print(f"  Training Episodes  : 30")
        print(f"  Total Trades       : {stats.get('total_trades', 0)}")
        print(f"  Win Rate           : {stats.get('win_rate', 0):.1f}%")
        print(f"  Total PnL          : {stats.get('total_pnl', 0):.4f}")
        print(f"  Final Epsilon      : {stats.get('final_epsilon', 0):.4f}")

        # Get current recommendation
        close  = df["Close"].values
        delta  = np.diff(close)
        g      = np.where(delta > 0, delta, 0)
        l      = np.where(delta < 0, -delta, 0)
        rs     = (np.mean(g[-14:]) + 1e-9) / (np.mean(l[-14:]) + 1e-9)
        rsi    = 100 - 100 / (1 + rs)

        ema12  = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26  = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd_h = (ema12 - ema26)[-1]
        ma20   = np.mean(close[-20:])
        price  = close[-1]

        action_names = {0: "HOLD ➡️", 1: "BUY 📈", 2: "SELL 📉"}
        action = agent.predict(rsi, macd_h, price, ma20)
        color  = signal_color("BUY" if action == 1 else "SELL" if action == 2 else "HOLD")
        _sep()
        print(f"\n  ★  RL RECOMMENDATION : {color}{action_names.get(action,'HOLD')}{RESET}")
        print(f"     RSI used       : {rsi:.2f}")
        print(f"     MACD hist      : {macd_h:.4f}")
        print(f"     Price vs MA20  : {'Above' if price > ma20 else 'Below'}")
    except Exception as e:
        print(f"  ⚠️  RL Agent error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 10 — News Sentiment Analyzer
# ══════════════════════════════════════════════════════════════════════════════

def section_sentiment(symbol: str):
    _h(f"📰 News Sentiment Analyzer — {symbol}")

    print("  Enter news headlines (one per line, blank line to finish):")
    print("  [Or press Enter directly to use demo headlines]\n")

    headlines = []
    while True:
        try:
            line = input("  > ").strip()
            if not line:
                break
            headlines.append(line)
        except (EOFError, KeyboardInterrupt):
            break

    if not headlines:
        # Demo headlines
        headlines = [
            f"{symbol} reports record quarterly earnings, beats estimates",
            f"Concerns over {symbol} debt levels emerge in analyst notes",
            f"{symbol} announces major partnership deal with global firm",
        ]
        print(f"  Using {len(headlines)} demo headlines.\n")

    _sep()
    overall_scores = []
    for i, headline in enumerate(headlines, 1):
        result = _safe(analyze_news, headline)
        if isinstance(result, dict) and "error" not in result:
            label  = result.get("label", "")
            score  = result.get("score", 0)
            conf   = result.get("confidence", 0)
            model  = result.get("model", "")
            color  = "\033[92m" if label == "positive" else \
                     "\033[91m" if label == "negative" else "\033[93m"
            print(f"  [{i}] {headline[:55]}")
            print(f"       Sentiment : {color}{label.upper()}{RESET}  "
                  f"Score: {score:+.3f}  Conf: {conf:.1%}  ({model})")
            overall_scores.append(score)
        else:
            err = result.get("error","") if isinstance(result,dict) else result
            print(f"  [{i}] ⚠️  {err}")

    if overall_scores:
        avg   = np.mean(overall_scores)
        color = "\033[92m" if avg > 0.1 else "\033[91m" if avg < -0.1 else "\033[93m"
        _sep()
        print(f"\n  📊 AGGREGATE SENTIMENT: {color}{avg:+.3f}{RESET}  "
              f"({'Bullish 📈' if avg > 0.1 else 'Bearish 📉' if avg < -0.1 else 'Neutral ➡️'})")


# ══════════════════════════════════════════════════════════════════════════════
# Full Analysis — All Modules
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(symbol: str, df: pd.DataFrame):
    """Run every module in sequence."""
    price = section_technical(symbol, df)
    score = section_ai(symbol, df)
    section_elliott_fibonacci(symbol, df)
    section_scanners(symbol, df)
    section_backtest(symbol, df)
    section_options(symbol, df)
    section_intelligence(symbol, df)
    section_rl_agent(symbol, df)


# ══════════════════════════════════════════════════════════════════════════════
# Live Quote
# ══════════════════════════════════════════════════════════════════════════════

def show_quote(symbol: str):
    quote = _safe(get_latest_price, symbol)
    if isinstance(quote, dict) and "error" not in quote:
        ch    = quote.get("change", 0)
        pct   = quote.get("change_pct", 0)
        color = "\033[92m" if ch >= 0 else "\033[91m"
        print(f"\n  Live Quote  : {fmt_price(quote.get('price',0))}  "
              f"{color}{ch:+.2f} ({pct:+.2f}%){RESET}  "
              f"| {quote.get('trend','')}  "
              f"| Vol: {fmt_volume(quote.get('volume',0))}")
        print(f"  Day Range   : {fmt_price(quote.get('low',0))} – "
              f"{fmt_price(quote.get('high',0))}")
        print(f"  Date        : {quote.get('date','')}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(MENU)

    choice = input("Select option [0-11]: ").strip()

    if choice == "0":
        print("Goodbye! 👋")
        return

    # Portfolio optimizer does not need a single symbol
    if choice == "7":
        section_portfolio()
        return

    raw    = input("\nEnter Stock Symbol (e.g. RELIANCE / TCS.NS / AAPL): ").strip()
    if not raw:
        print("No symbol entered.")
        return
    symbol = resolve_symbol(raw)

    # Sentiment analyzer needs symbol but not df
    if choice == "11":
        section_sentiment(symbol)
        return

    df = fetch(symbol)
    if df is None:
        return

    show_quote(symbol)

    dispatch = {
        "1":  lambda: run_full_analysis(symbol, df),
        "2":  lambda: section_technical(symbol, df),
        "3":  lambda: section_ai(symbol, df),
        "4":  lambda: section_elliott_fibonacci(symbol, df),
        "5":  lambda: section_scanners(symbol, df),
        "6":  lambda: section_backtest(symbol, df),
        "8":  lambda: section_options(symbol, df),
        "9":  lambda: section_intelligence(symbol, df),
        "10": lambda: section_rl_agent(symbol, df),
    }

    fn = dispatch.get(choice)
    if fn:
        fn()
    else:
        print(f"Unknown option: {choice}")
        run_full_analysis(symbol, df)


if __name__ == "__main__":
    main()
