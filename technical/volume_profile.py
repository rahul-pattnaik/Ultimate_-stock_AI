# technical/volume_profile.py
# ─────────────────────────────────────────────────────────────────────────────
# Volume Profile Engine
# Computes: POC · VAH · VAL · Value Area · Volume by price histogram
# Output   : POC float (backward compat) + full profile dict
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


def volume_profile(df: pd.DataFrame, bins: int = 30) -> float:
    """
    Point of Control (POC) — backward compatible with main_terminal.py.
    Returns the price level with the highest traded volume.
    """
    if df is None or len(df) < 5:
        return 0.0

    price  = df["Close"].dropna().values.flatten().astype(float)
    volume = df["Volume"].values.flatten().astype(float)

    hist, edges = np.histogram(price, bins=bins, weights=volume)
    poc_idx = int(np.argmax(hist))
    poc     = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    return round(poc, 2)


def volume_profile_full(df: pd.DataFrame,
                        bins: int = 30,
                        value_area_pct: float = 0.70) -> dict:
    """
    Full Volume Profile with:
        POC  — Point of Control (highest volume price)
        VAH  — Value Area High (top of 70% volume zone)
        VAL  — Value Area Low  (bottom of 70% volume zone)
        HVN  — High Volume Nodes (price magnets)
        LVN  — Low Volume Nodes  (price gaps / fast-move zones)

    Args:
        df             : yfinance OHLCV DataFrame
        bins           : price bucket resolution
        value_area_pct : fraction of total volume to define Value Area

    Returns full profile dict.
    """
    if df is None or len(df) < 10:
        return {"error": "Need at least 10 rows"}

    price  = df["Close"].dropna().values.flatten().astype(float)
    volume = df["Volume"].values.flatten().astype(float)
    current_price = float(price[-1])

    hist, edges = np.histogram(price, bins=bins, weights=volume)
    bin_centers  = (edges[:-1] + edges[1:]) / 2
    total_vol    = hist.sum()

    # ── POC ───────────────────────────────────────────────────────────
    poc_idx = int(np.argmax(hist))
    poc     = round(float(bin_centers[poc_idx]), 2)

    # ── Value Area (VA = 70% of volume centred around POC) ────────────
    va_target  = total_vol * value_area_pct
    va_vol     = hist[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx

    while va_vol < va_target:
        can_go_lo = lo_idx > 0
        can_go_hi = hi_idx < len(hist) - 1

        add_lo = hist[lo_idx - 1] if can_go_lo else 0
        add_hi = hist[hi_idx + 1] if can_go_hi else 0

        if not can_go_lo and not can_go_hi:
            break
        if add_hi >= add_lo:
            hi_idx += 1; va_vol += add_hi
        else:
            lo_idx -= 1; va_vol += add_lo

    vah = round(float(bin_centers[hi_idx]), 2)
    val = round(float(bin_centers[lo_idx]), 2)

    # ── HVN / LVN ─────────────────────────────────────────────────────
    mean_vol  = hist.mean()
    std_vol   = hist.std()
    hvn_mask  = hist > mean_vol + 0.5 * std_vol
    lvn_mask  = hist < mean_vol - 0.5 * std_vol

    hvn = sorted([round(float(bin_centers[i]), 2)
                  for i in np.where(hvn_mask)[0]], reverse=True)
    lvn = sorted([round(float(bin_centers[i]), 2)
                  for i in np.where(lvn_mask)[0]])

    # ── Price position relative to value area ─────────────────────────
    if   current_price > vah: position = "Above Value Area (extended / sell zone)"
    elif current_price < val: position = "Below Value Area (discount / buy zone)"
    else:                     position = "Inside Value Area (fair value)"

    # ── Volume distribution (top 5 bins for chart) ────────────────────
    top5_idx  = np.argsort(hist)[-5:][::-1]
    histogram = [{"price": round(float(bin_centers[i]), 2),
                  "volume": int(hist[i]),
                  "pct":    round(float(hist[i] / total_vol * 100), 1)}
                 for i in top5_idx]

    return {
        "current_price":   round(current_price, 2),
        "POC":             poc,
        "VAH":             vah,
        "VAL":             val,
        "price_position":  position,
        "HVN":             hvn[:5],   # top 5 high-volume nodes
        "LVN":             lvn[:5],   # top 5 low-volume nodes
        "top_volume_bins": histogram,
        "total_volume":    int(total_vol),
        "value_area_pct":  f"{value_area_pct * 100:.0f}%",
    }
