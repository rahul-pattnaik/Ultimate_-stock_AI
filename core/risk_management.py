from __future__ import annotations

from typing import Any


def build_risk_plan(
    current_price: float,
    support: float,
    resistance: float,
    capital: float = 100000.0,
    risk_pct: float = 0.01,
    volatility_regime: str = "Normal",
) -> dict[str, Any]:
    stop_loss = support * 0.985 if support > 0 else current_price * 0.95
    target = resistance if resistance > current_price else current_price * 1.08
    per_share_risk = max(current_price - stop_loss, current_price * 0.01)
    capital_at_risk = capital * risk_pct
    quantity = max(int(capital_at_risk / per_share_risk), 1)
    reward = max(target - current_price, current_price * 0.02)
    risk_reward = reward / per_share_risk if per_share_risk > 0 else 0.0
    max_drawdown_warning = "Elevated" if "high" in volatility_regime.lower() or risk_reward < 1.2 else "Normal"

    return {
        "stop_loss": round(stop_loss, 2),
        "target_price": round(target, 2),
        "position_size_shares": quantity,
        "capital_at_risk": round(capital_at_risk, 2),
        "risk_reward_ratio": round(risk_reward, 2),
        "max_drawdown_warning": max_drawdown_warning,
    }
