# core/utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# Covers: formatting · logging · terminal output · data helpers · timing
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import logging
import functools
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import CURRENCY_SYMBOL, DISPLAY_SEPARATOR, SCORE_STRONG_BUY, SCORE_BUY, SCORE_HOLD


# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logger(name: str = "stock_ai",
                 level: int = logging.INFO,
                 log_file: str = None) -> logging.Logger:
    """
    Configure and return a logger with console + optional file handler.

    Args:
        name     : logger name
        level    : logging level (default INFO)
        log_file : optional file path to write logs

    Returns configured Logger.
    """
    logger    = logging.getLogger(name)
    logger.setLevel(level)
    fmt       = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if not logger.handlers:
        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ── Timing Decorator ──────────────────────────────────────────────────────────

def timer(func):
    """Decorator: print how long a function took to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱  {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ── Number Formatting ─────────────────────────────────────────────────────────

def fmt_price(price: float, currency: str = CURRENCY_SYMBOL) -> str:
    """Format a price with currency symbol and 2 decimal places."""
    return f"{currency}{price:,.2f}"


def fmt_pct(value: float, show_sign: bool = True) -> str:
    """Format a percentage value."""
    sign = "+" if show_sign and value >= 0 else ""
    return f"{sign}{value:.2f}%"


def fmt_volume(vol: int) -> str:
    """Format volume with K/M/Cr suffix."""
    if vol >= 1_00_00_000:  return f"{vol / 1_00_00_000:.2f} Cr"
    if vol >= 1_000_000:    return f"{vol / 1_000_000:.2f}M"
    if vol >= 1_000:        return f"{vol / 1_000:.1f}K"
    return str(vol)


def fmt_large_number(n: float) -> str:
    """Format large numbers (market cap etc.) with readable suffix."""
    if   n >= 1e12: return f"₹{n/1e12:.2f}T"
    elif n >= 1e9:  return f"₹{n/1e9:.2f}B"
    elif n >= 1e7:  return f"₹{n/1e7:.2f}Cr"
    elif n >= 1e5:  return f"₹{n/1e5:.2f}L"
    return f"₹{n:,.0f}"


# ── Signal & Score Helpers ────────────────────────────────────────────────────

def score_to_signal(score: int) -> str:
    """Convert numeric score (0-100) to signal label."""
    if   score >= SCORE_STRONG_BUY: return "STRONG BUY 🚀"
    elif score >= SCORE_BUY:        return "BUY 📈"
    elif score >= SCORE_HOLD:       return "HOLD ➡️"
    elif score >= SCORE_SELL:       return "SELL 📉"
    else:                           return "STRONG SELL 🔴"


def score_to_grade(score: int) -> str:
    """Convert score to letter grade."""
    if   score >= 90: return "A+"
    elif score >= 80: return "A"
    elif score >= 70: return "B+"
    elif score >= 60: return "B"
    elif score >= 50: return "C"
    elif score >= 40: return "D"
    else:             return "F"


def signal_color(signal: str) -> str:
    """Return ANSI color code for a signal string."""
    s = signal.upper()
    if "STRONG BUY" in s:  return "\033[92m"   # bright green
    if "BUY"        in s:  return "\033[32m"   # green
    if "HOLD"       in s:  return "\033[33m"   # yellow
    if "SELL"       in s:  return "\033[91m"   # bright red
    return "\033[0m"


RESET = "\033[0m"


# ── Terminal Output ───────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print(f"\n{DISPLAY_SEPARATOR}")
    print(f"  {title.upper()}")
    print(DISPLAY_SEPARATOR)


def print_stock_report(symbol: str, analysis: dict) -> None:
    """
    Pretty-print a full stock analysis dict to the terminal.
    Accepts the output format from main_terminal.py.
    """
    print_header(f"Stock Analysis: {symbol}")

    # Price & prediction
    pred = analysis.get("prediction", {})
    if pred and "error" not in pred:
        print(f"  Current Price   : {fmt_price(pred.get('current_price', 0))}")
        print(f"  5d Prediction   : {fmt_price(pred.get('predicted_price_5d', 0))} "
              f"({fmt_pct(pred.get('change_percent', 0))}) {pred.get('direction', '')}")
        print(f"  Confidence      : {pred.get('confidence', 'N/A')}")

    print()

    # Signal
    sig = analysis.get("ai_signal", {})
    if isinstance(sig, dict) and "signal" in sig:
        color = signal_color(sig["signal"])
        print(f"  AI Signal       : {color}{sig['signal']}{RESET}")
        print(f"  Signal Score    : {sig.get('score', 'N/A')}/100")

    # Score
    score = analysis.get("score", 0)
    if score:
        color = signal_color(score_to_signal(score))
        print(f"  AI Score        : {color}{score}/100 ({score_to_grade(score)}){RESET}")
        print(f"  Verdict         : {color}{score_to_signal(score)}{RESET}")

    print()

    # Technical
    trend     = analysis.get("trend", "N/A")
    breakout  = analysis.get("breakout", False)
    support   = analysis.get("support", "N/A")
    resistance= analysis.get("resistance", "N/A")
    volume    = analysis.get("volume_poc", "N/A")

    print(f"  Trend           : {trend}")
    print(f"  Breakout        : {'✅ YES' if breakout else '❌ NO'}")
    print(f"  Support         : {fmt_price(support) if isinstance(support, (int, float)) else support}")
    print(f"  Resistance      : {fmt_price(resistance) if isinstance(resistance, (int, float)) else resistance}")
    print(f"  Volume POC      : {fmt_price(volume) if isinstance(volume, (int, float)) else volume}")

    print(DISPLAY_SEPARATOR)


def print_screener_results(results: list, top_n: int = 10) -> None:
    """
    Print ranked stock screener results in a table format.

    Args:
        results : list of dicts with 'symbol' and 'score' keys
        top_n   : how many to display
    """
    print_header(f"Top {top_n} Stocks by AI Score")
    results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_n]
    print(f"  {'#':<3} {'Symbol':<15} {'Score':>6} {'Grade':>5} {'Signal':<20}")
    print(f"  {'-'*3} {'-'*15} {'-'*6} {'-'*5} {'-'*20}")
    for i, r in enumerate(results, 1):
        score  = r.get("score", 0)
        signal = score_to_signal(score)
        grade  = score_to_grade(score)
        color  = signal_color(signal)
        print(f"  {i:<3} {r.get('symbol', ''):<15} {score:>6} {grade:>5} "
              f"{color}{signal:<20}{RESET}")
    print()


# ── Data Helpers ──────────────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily and log return columns to DataFrame."""
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"]   = np.log(df["Close"] / df["Close"].shift(1))
    return df


def rolling_stats(series: pd.Series, window: int = 20) -> pd.DataFrame:
    """Compute rolling mean, std, min, max for a Series."""
    return pd.DataFrame({
        "mean": series.rolling(window).mean(),
        "std":  series.rolling(window).std(),
        "min":  series.rolling(window).min(),
        "max":  series.rolling(window).max(),
    })


def normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def detect_gaps(df: pd.DataFrame, threshold_pct: float = 2.0) -> pd.DataFrame:
    """
    Detect price gaps between sessions.

    Args:
        df            : OHLCV DataFrame
        threshold_pct : minimum % gap to flag

    Returns DataFrame of gap events.
    """
    gaps = []
    for i in range(1, len(df)):
        prev_close = float(df["Close"].iloc[i - 1])
        curr_open  = float(df["Open"].iloc[i])
        gap_pct    = (curr_open - prev_close) / prev_close * 100
        if abs(gap_pct) >= threshold_pct:
            gaps.append({
                "date":      str(df.index[i].date()),
                "prev_close": round(prev_close, 2),
                "open":       round(curr_open, 2),
                "gap_pct":    round(gap_pct, 2),
                "type":       "Gap Up 📈" if gap_pct > 0 else "Gap Down 📉",
            })
    return pd.DataFrame(gaps)


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """Division that returns `default` instead of ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else default


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into a single-level dict."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def timestamp() -> str:
    """Return current datetime as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
