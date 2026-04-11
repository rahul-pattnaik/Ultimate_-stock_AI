# technical/elliott_wave.py
# ─────────────────────────────────────────────────────────────────────────────
# Elliott Wave Detector
# Approach : ZigZag pivot detection → wave labeling → Fibonacci validation
# Output   : detected wave count, current wave, targets, wave structure
# Note     : Rule-based approximation (not a trained ML model)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── ZigZag ────────────────────────────────────────────────────────────────────

def _zigzag(close: np.ndarray, threshold: float = 0.03) -> list:
    """
    Detect significant pivot highs/lows using a minimum % move threshold.

    Args:
        close     : 1-D price array
        threshold : minimum move (as fraction) to register a new pivot

    Returns list of (index, price, direction) where direction: 1=high, -1=low
    """
    pivots    = [(0, close[0], 1 if close[1] > close[0] else -1)]
    direction = pivots[0][2]
    last_price = close[0]

    for i in range(1, len(close)):
        price = close[i]
        if direction == 1:                              # looking for higher high
            if price > last_price:
                pivots[-1] = (i, price, 1)             # extend current high
                last_price  = price
            elif (last_price - price) / last_price > threshold:
                pivots.append((i, price, -1))           # new low pivot
                direction  = -1
                last_price = price
        else:                                           # looking for lower low
            if price < last_price:
                pivots[-1] = (i, price, -1)
                last_price  = price
            elif (price - last_price) / last_price > threshold:
                pivots.append((i, price, 1))
                direction  = 1
                last_price = price

    return pivots


# ── Wave Rules ────────────────────────────────────────────────────────────────

def _validate_impulse(waves: list) -> tuple[bool, list]:
    """
    Check basic Elliott Wave impulse rules for waves [0..5]:
        Wave 2 cannot retrace below Wave 1 start
        Wave 3 cannot be the shortest impulse wave
        Wave 4 cannot enter Wave 1 territory
    """
    if len(waves) < 6:
        return False, ["Not enough pivots"]

    prices = [w[1] for w in waves[:6]]
    errors = []

    # Rule 1: Wave 2 does not go below wave 0
    if prices[2] < prices[0]:
        errors.append("Wave 2 below Wave 1 origin (invalid)")

    # Rule 2: Wave 3 not shortest (compare to waves 1 and 5)
    w1 = abs(prices[1] - prices[0])
    w3 = abs(prices[3] - prices[2])
    w5 = abs(prices[5] - prices[4])
    if w3 < w1 and w3 < w5:
        errors.append("Wave 3 is shortest — invalid impulse")

    # Rule 3: Wave 4 does not enter Wave 1 territory
    if prices[4] < prices[1]:
        errors.append("Wave 4 overlaps Wave 1 (invalid)")

    return len(errors) == 0, errors


def _fib_targets(low: float, high: float) -> dict:
    """Fibonacci extension targets from a wave base."""
    rng = high - low
    return {
        "1.272x": round(high + 0.272 * rng, 2),
        "1.618x": round(high + 0.618 * rng, 2),
        "2.000x": round(high + 1.000 * rng, 2),
        "2.618x": round(high + 1.618 * rng, 2),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def detect_elliott_wave(df: pd.DataFrame,
                        threshold: float = 0.03) -> dict:
    """
    Detect Elliott Wave structure in price data.

    Args:
        df        : yfinance OHLCV DataFrame (≥30 bars)
        threshold : ZigZag sensitivity (0.03 = 3% minimum move)

    Returns:
        wave_count     : number of pivots detected
        current_wave   : estimated wave position (1-5 or A-C)
        valid_impulse  : whether detected 5-wave structure is valid
        wave_pivots    : list of (bar_index, price, direction)
        targets        : Fibonacci extension price targets
        structure      : "Impulse" | "Corrective" | "Unclear"
    """
    if df is None or len(df) < 30:
        return {"error": "Need at least 30 rows"}

    close   = df["Close"].dropna().values.flatten().astype(float)
    pivots  = _zigzag(close, threshold)

    wave_count = len(pivots)
    directions = [p[2] for p in pivots]
    prices     = [p[1] for p in pivots]

    # Estimate current wave
    current_wave = wave_count % 5 or 5
    wave_labels  = {1: "Wave 1", 2: "Wave 2 (retracement)",
                    3: "Wave 3 (strongest)", 4: "Wave 4 (consolidation)",
                    5: "Wave 5 (final push)"}
    wave_label = wave_labels.get(current_wave, f"Wave {current_wave}")

    # Validate impulse if we have 6+ pivots
    valid, errors = _validate_impulse(pivots) if wave_count >= 6 else (False, ["Not enough waves"])

    # Structure type
    if wave_count >= 5 and valid:
        structure = "Impulse (5-wave) ✅"
    elif wave_count >= 3:
        structure = "Corrective (A-B-C) — possible"
    else:
        structure = "Unclear (more data needed)"

    # Fibonacci targets from last full wave
    targets = {}
    if len(prices) >= 2:
        low  = min(prices[-2], prices[-1])
        high = max(prices[-2], prices[-1])
        targets = _fib_targets(low, high)

    # Summary of last 5 pivots for display
    recent_pivots = [
        {"bar": p[0], "price": round(p[1], 2),
         "type": "High 🔼" if p[2] == 1 else "Low 🔽"}
        for p in pivots[-5:]
    ]

    # Degree bias
    last_dir = directions[-1] if directions else 0
    bias = "Bullish" if last_dir == 1 else "Bearish"

    return {
        "structure":      structure,
        "wave_count":     wave_count,
        "current_wave":   wave_label,
        "bias":           bias,
        "valid_impulse":  valid,
        "validation_notes": errors if errors else ["All impulse rules satisfied ✅"],
        "fibonacci_targets": targets,
        "recent_pivots":  recent_pivots,
        "current_price":  round(float(close[-1]), 2),
        "sensitivity":    f"{int(threshold * 100)}% ZigZag",
    }
