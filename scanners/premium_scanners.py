from __future__ import annotations

from typing import Any

import pandas as pd

from technical.advanced_analysis import advanced_technical_report


def run_premium_scanners(datasets: dict[str, pd.DataFrame], fundamentals_map: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    near_breakout: list[dict[str, Any]] = []
    rsi_oversold: list[dict[str, Any]] = []
    golden_cross: list[dict[str, Any]] = []
    high_volume: list[dict[str, Any]] = []
    value_candidates: list[dict[str, Any]] = []

    for symbol, df in datasets.items():
        if df is None or df.empty or len(df) < 60:
            continue
        close = df["Close"]
        volume = df["Volume"]
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])
        avg20_vol = float(volume.tail(20).mean())
        vol_ratio = float(volume.iloc[-1] / avg20_vol) if avg20_vol else 1.0
        tech = advanced_technical_report(df)
        resistance = float(tech.get("resistance", 0.0))
        support = float(tech.get("support", 0.0))
        rsi_note = next((note for note in tech.get("notes", []) if note.startswith("RSI regime")), "")
        rsi_value = float(rsi_note.split(":")[-1].strip()) if ":" in rsi_note else 50.0
        fundamentals = fundamentals_map.get(symbol, {})

        if resistance and 0 < (resistance - close.iloc[-1]) / close.iloc[-1] <= 0.03:
            near_breakout.append({"symbol": symbol, "price": round(float(close.iloc[-1]), 2), "resistance": round(resistance, 2)})
        if rsi_value <= 35:
            rsi_oversold.append({"symbol": symbol, "rsi": round(rsi_value, 2), "support": round(support, 2)})
        if ma50 > ma200 and float(close.iloc[-2]) > ma50:
            golden_cross.append({"symbol": symbol, "ma50": round(ma50, 2), "ma200": round(ma200, 2)})
        if vol_ratio >= 1.8:
            high_volume.append({"symbol": symbol, "volume_ratio": round(vol_ratio, 2), "price": round(float(close.iloc[-1]), 2)})
        if float(fundamentals.get("fundamental_score", 0.0)) >= 65 and float(fundamentals.get("fair_value_upside_pct", 0.0)) >= 10:
            value_candidates.append({"symbol": symbol, "fundamental_score": fundamentals.get("fundamental_score"), "upside_pct": fundamentals.get("fair_value_upside_pct")})

    return {
        "near_breakout": near_breakout,
        "rsi_oversold": rsi_oversold,
        "golden_cross": golden_cross,
        "high_volume_movers": high_volume,
        "strong_fundamentals_under_value": value_candidates,
    }
