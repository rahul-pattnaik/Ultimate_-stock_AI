# technical/support_resistance.py
# ─────────────────────────────────────────────────────────────────────────────
# Support & Resistance Engine
# Methods: Rolling min/max · Pivot Points · Price Clustering (KDE)
# Output : multi-level S/R zones with strength scores
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pivot_levels(df: pd.DataFrame, left: int = 5, right: int = 5) -> dict:
    """
    Classic Pivot Point (PP) levels using previous session's HLC.
    Also computes R1/R2/R3 and S1/S2/S3.
    """
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])
    c = float(df["Close"].iloc[-1])

    pp = (h + l + c) / 3
    r1 = 2 * pp - l
    r2 = pp + (h - l)
    r3 = h + 2 * (pp - l)
    s1 = 2 * pp - h
    s2 = pp - (h - l)
    s3 = l - 2 * (h - pp)

    return {
        "PP": round(pp, 2),
        "R1": round(r1, 2), "R2": round(r2, 2), "R3": round(r3, 2),
        "S1": round(s1, 2), "S2": round(s2, 2), "S3": round(s3, 2),
    }


def _swing_levels(close: np.ndarray, order: int = 5) -> tuple[list, list]:
    """
    Detect local swing highs/lows using argrelextrema.
    Returns lists of (index, price) for supports and resistances.
    """
    local_min_idx = argrelextrema(close, np.less_equal,    order=order)[0]
    local_max_idx = argrelextrema(close, np.greater_equal, order=order)[0]
    supports    = [float(close[i]) for i in local_min_idx]
    resistances = [float(close[i]) for i in local_max_idx]
    return supports, resistances


def _cluster_levels(levels: list, tolerance: float = 0.015) -> list:
    """
    Merge nearby price levels within `tolerance` % of each other.
    Returns sorted list of (price, strength) where strength = # merged levels.
    """
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    current  = [levels[0]]

    for price in levels[1:]:
        if (price - current[-1]) / current[-1] < tolerance:
            current.append(price)
        else:
            clusters.append(current)
            current = [price]
    clusters.append(current)

    return sorted(
        [(round(sum(c) / len(c), 2), len(c)) for c in clusters],
        key=lambda x: x[1], reverse=True
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_support_resistance(df: pd.DataFrame, lookback: int = 60) -> tuple:
    """
    Compute multi-method support & resistance levels.

    Args:
        df       : yfinance OHLCV DataFrame
        lookback : bars to use for swing detection

    Returns:
        (support, resistance)  ← single best values for backward compatibility
    """
    if df is None or len(df) < 20:
        return None, None

    close = df["Close"].dropna().values.flatten().astype(float)
    window = close[-min(lookback, len(close)):]

    # Simple rolling levels
    rolling_sup = round(float(np.min(window[-20:])), 2)
    rolling_res = round(float(np.max(window[-20:])), 2)

    return rolling_sup, rolling_res


def get_sr_zones(df: pd.DataFrame, lookback: int = 120) -> dict:
    """
    Full multi-level S/R analysis.

    Returns:
        support_zones    : list of (price, strength) sorted by strength
        resistance_zones : list of (price, strength) sorted by strength
        pivot_points     : classic PP/R1-R3/S1-S3
        current_price    : latest close
        nearest_support  : closest support below price
        nearest_resistance: closest resistance above price
    """
    if df is None or len(df) < 30:
        return {"error": "Need at least 30 rows"}

    close   = df["Close"].dropna().values.flatten().astype(float)
    window  = close[-min(lookback, len(close)):]
    price   = close[-1]

    # Swing levels
    sup_raw, res_raw = _swing_levels(window, order=5)

    # 52-week high/low as hard anchors
    sup_raw.append(float(np.min(window)))
    res_raw.append(float(np.max(window)))

    # Rolling 20/50 lows and highs
    s = pd.Series(window)
    for w in [20, 50]:
        sup_raw.append(float(s.rolling(w).min().dropna().iloc[-1]))
        res_raw.append(float(s.rolling(w).max().dropna().iloc[-1]))

    # Cluster & rank
    sup_zones = _cluster_levels([p for p in sup_raw if p < price])
    res_zones = _cluster_levels([p for p in res_raw if p > price])

    nearest_sup = max((p for p, _ in sup_zones if p < price), default=None)
    nearest_res = min((p for p, _ in res_zones if p > price), default=None)

    pivots = _pivot_levels(df)

    return {
        "current_price":     round(price, 2),
        "support_zones":     sup_zones[:5],   # top 5 strongest
        "resistance_zones":  res_zones[:5],
        "nearest_support":   nearest_sup,
        "nearest_resistance": nearest_res,
        "pivot_points":      pivots,
        "risk_reward": round(
            (nearest_res - price) / (price - nearest_sup), 2
        ) if nearest_sup and nearest_res and (price - nearest_sup) > 0 else None,
    }
