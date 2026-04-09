# technical/vwap.py
# ─────────────────────────────────────────────────────────────────────────────
# VWAP — Volume Weighted Average Price
# Includes: VWAP · SD bands (±1σ, ±2σ) · Price position · Anchored VWAP
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3


def vwap(df: pd.DataFrame) -> float:
    """
    Rolling VWAP over all available data.
    Returns the latest VWAP value.
    """
    tp   = _typical_price(df)
    vol  = df["Volume"].astype(float)
    cumtp  = (tp * vol).cumsum()
    cumvol = vol.cumsum()
    vwap_series = cumtp / cumvol
    return round(float(vwap_series.iloc[-1]), 2)


def vwap_bands(df: pd.DataFrame, sd_mult: list = [1.0, 2.0]) -> dict:
    """
    VWAP with upper/lower standard deviation bands.

    Args:
        df      : yfinance OHLCV DataFrame
        sd_mult : list of σ multipliers for bands (default [1, 2])

    Returns:
        vwap, bands, price position, distance from VWAP
    """
    if df is None or len(df) < 5:
        return {"error": "Insufficient data"}

    tp   = _typical_price(df)
    vol  = df["Volume"].astype(float).replace(0, np.nan).fillna(1)

    cumtp   = (tp * vol).cumsum()
    cumvol  = vol.cumsum()
    vwap_s  = cumtp / cumvol

    # Variance: sum(vol * (tp - vwap)^2) / sum(vol)
    variance = ((tp - vwap_s) ** 2 * vol).cumsum() / cumvol
    sd       = np.sqrt(variance)

    vwap_val = round(float(vwap_s.iloc[-1]), 2)
    sd_val   = round(float(sd.iloc[-1]), 4)
    price    = round(float(df["Close"].iloc[-1]), 2)

    bands = {}
    for m in sd_mult:
        bands[f"upper_{m}sd"] = round(vwap_val + m * sd_val, 2)
        bands[f"lower_{m}sd"] = round(vwap_val - m * sd_val, 2)

    # Position
    dist_pct = round((price - vwap_val) / vwap_val * 100, 2)
    if   price > bands.get("upper_2.0sd", vwap_val + 2*sd_val):
        position = "Extremely Above VWAP (overbought) 🔴"
    elif price > bands.get("upper_1.0sd", vwap_val + sd_val):
        position = "Above VWAP Upper Band ⚠️"
    elif price > vwap_val:
        position = "Above VWAP — Bullish ✅"
    elif price > bands.get("lower_1.0sd", vwap_val - sd_val):
        position = "Below VWAP — Bearish 📉"
    else:
        position = "Below VWAP Lower Band — Oversold 🟢"

    return {
        "vwap":         vwap_val,
        "price":        price,
        "sd":           sd_val,
        "bands":        bands,
        "position":     position,
        "distance_pct": f"{dist_pct:+.2f}%",
    }


def anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> pd.Series:
    """
    Anchored VWAP — starts accumulation from a specific bar index.
    Useful for anchoring to earnings, swing lows, IPO dates etc.

    Args:
        df         : full OHLCV DataFrame
        anchor_idx : row index to start from (0 = first bar)

    Returns pd.Series of AVWAP values from anchor to present.
    """
    sub  = df.iloc[anchor_idx:].copy()
    tp   = _typical_price(sub)
    vol  = sub["Volume"].astype(float).replace(0, 1)
    avwap = (tp * vol).cumsum() / vol.cumsum()
    return avwap.round(2)
