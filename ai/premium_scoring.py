from __future__ import annotations

from typing import Any

import numpy as np


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _confidence_label(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


def generate_premium_scores(
    technical_report: dict[str, Any],
    fundamental_report: dict[str, Any],
    ai_signal_result: Any = None,
    prediction: Any = None,
) -> dict[str, Any]:
    technical_score = _as_float(technical_report.get("confluence_score"), 50.0)
    momentum_score = _as_float(technical_report.get("momentum_score"), 50.0)
    fundamental_score = _as_float(fundamental_report.get("fundamental_score"), 50.0)
    upside = _as_float(fundamental_report.get("fair_value_upside_pct"), 0.0)

    ai_score = 50.0
    if isinstance(ai_signal_result, dict):
        ai_score = _as_float(ai_signal_result.get("score"), ai_score)
    if isinstance(prediction, dict):
        ai_score = (ai_score + np.clip(50 + _as_float(prediction.get("change_percent"), 0.0) * 2.0, 0, 100)) / 2.0

    buy_score = float(np.clip(technical_score * 0.35 + momentum_score * 0.2 + fundamental_score * 0.25 + ai_score * 0.2, 0, 100))
    swing_score = float(np.clip(momentum_score * 0.45 + technical_score * 0.4 + ai_score * 0.15, 0, 100))
    long_term_score = float(np.clip(fundamental_score * 0.55 + technical_score * 0.2 + ai_score * 0.1 + np.clip(50 + upside, 0, 100) * 0.15, 0, 100))

    risk_penalty = 0.0
    volatility_regime = str(technical_report.get("volatility_regime", "")).lower()
    if "high" in volatility_regime:
        risk_penalty += 18.0
    if str(fundamental_report.get("debt_health", "")).lower() == "stretched":
        risk_penalty += 20.0
    if "downtrend" in str(technical_report.get("trend", "")).lower():
        risk_penalty += 12.0
    risk_score = float(np.clip(35 + risk_penalty, 0, 100))

    confidence = float(np.clip((technical_score + fundamental_score + ai_score) / 3.0, 0, 100))
    trend = str(technical_report.get("trend", "Neutral"))
    if buy_score >= 75:
        action = "BUY"
    elif buy_score >= 60:
        action = "ACCUMULATE"
    elif buy_score >= 45:
        action = "HOLD"
    else:
        action = "AVOID"

    risk_label = "Low" if risk_score < 40 else "Medium" if risk_score < 70 else "High"
    return {
        "buy_score": round(buy_score, 2),
        "swing_score": round(swing_score, 2),
        "long_term_score": round(long_term_score, 2),
        "risk_score": round(risk_score, 2),
        "risk_label": risk_label,
        "confidence": round(confidence, 2),
        "confidence_label": _confidence_label(confidence),
        "trend": trend,
        "action": action,
    }
