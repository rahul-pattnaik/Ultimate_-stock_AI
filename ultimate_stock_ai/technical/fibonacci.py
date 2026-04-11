# technical/fibonacci.py
# ─────────────────────────────────────────────────────────────────────────────
# Fibonacci Analysis
# Levels: Retracements · Extensions · Fan lines · Confluence zones
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000]
EXTENSION_LEVELS   = [1.272, 1.414, 1.618, 2.000, 2.618]


def fibonacci_retracements(df: pd.DataFrame,
                           lookback: int = 60) -> dict:
    """
    Compute Fibonacci retracement levels from the highest high to the
    lowest low within the lookback window.

    Args:
        df       : yfinance OHLCV DataFrame
        lookback : bars to scan for swing high/low

    Returns:
        swing_high, swing_low, all retracement levels, current price position
    """
    if df is None or len(df) < 10:
        return {"error": "Need at least 10 rows"}

    window = df.iloc[-min(lookback, len(df)):]
    high   = float(window["High"].max())
    low    = float(window["Low"].min())
    price  = float(df["Close"].iloc[-1])
    rng    = high - low

    # Retracements (from high down)
    ret_levels = {
        f"{int(r * 100)}% ({round(high - r * rng, 2)})": round(high - r * rng, 2)
        for r in RETRACEMENT_LEVELS
    }

    # Extensions (from low, projecting beyond high)
    ext_levels = {
        f"{r}x ({round(low + r * rng, 2)})": round(low + r * rng, 2)
        for r in EXTENSION_LEVELS
    }

    # Find nearest support / resistance fib levels
    all_prices = sorted(ret_levels.values())
    nearest_support    = max((p for p in all_prices if p < price), default=None)
    nearest_resistance = min((p for p in all_prices if p > price), default=None)

    # Classify price location
    ret_vals = list(ret_levels.values())
    for i in range(len(ret_vals) - 1):
        if ret_vals[i+1] <= price <= ret_vals[i]:
            zone = f"Between {int(RETRACEMENT_LEVELS[i]*100)}%–{int(RETRACEMENT_LEVELS[i+1]*100)}% retracement"
            break
    else:
        zone = "Outside retracement range"

    return {
        "swing_high":          round(high, 2),
        "swing_low":           round(low, 2),
        "current_price":       round(price, 2),
        "retracement_levels":  ret_levels,
        "extension_levels":    ext_levels,
        "nearest_fib_support": nearest_support,
        "nearest_fib_resistance": nearest_resistance,
        "price_zone":          zone,
        "golden_ratio_level":  round(high - 0.618 * rng, 2),   # most important
        "key_support":         round(high - 0.618 * rng, 2),
        "key_resistance":      round(high - 0.382 * rng, 2),
    }


def fibonacci_extensions(high: float, low: float,
                         retracement: float = 0.618) -> dict:
    """
    Project Fibonacci extension targets given a known H/L and retracement.

    Args:
        high        : swing high price
        low         : swing low price
        retracement : expected pullback level (default 61.8%)

    Returns dict of extension price targets.
    """
    rng     = high - low
    pull_back = high - retracement * rng
    return {
        f"ext_{int(e * 1000) / 10}%": round(pull_back + e * rng, 2)
        for e in EXTENSION_LEVELS
    }


def fib_confluence_zones(df: pd.DataFrame,
                          lookbacks: list = [20, 60, 120]) -> dict:
    """
    Find price zones where Fibonacci levels from multiple timeframes cluster.
    Clustered levels = stronger S/R zones.

    Args:
        df        : OHLCV DataFrame
        lookbacks : list of lookback windows to compute fib levels from

    Returns confluence zones sorted by strength.
    """
    if df is None or len(df) < 20:
        return {"error": "Need at least 20 rows"}

    all_levels = []
    for lb in lookbacks:
        result = fibonacci_retracements(df, lookback=lb)
        if "error" not in result:
            all_levels.extend(result["retracement_levels"].values())

    all_levels.sort()
    tolerance = float(df["Close"].iloc[-1]) * 0.005   # 0.5% cluster window

    clusters   = []
    current    = [all_levels[0]]
    for p in all_levels[1:]:
        if p - current[-1] < tolerance:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    confluence = sorted(
        [{"price": round(sum(c)/len(c), 2), "strength": len(c)}
         for c in clusters if len(c) > 1],
        key=lambda x: x["strength"], reverse=True
    )

    return {
        "confluence_zones": confluence[:8],
        "strongest_zone":   confluence[0] if confluence else None,
    }
