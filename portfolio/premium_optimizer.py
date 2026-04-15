from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sector_guess(symbol: str) -> str:
    token = symbol.upper()
    if "BANK" in token or "FIN" in token:
        return "Financials"
    if "PHARMA" in token or "MED" in token:
        return "Healthcare"
    if "TECH" in token or "INF" in token or token in {"TCS.NS", "INFY.NS", "WIPRO.NS"}:
        return "Technology"
    if "AUTO" in token or "MOTOR" in token:
        return "Automobile"
    if "POWER" in token or "ENER" in token or "ONGC" in token:
        return "Energy"
    return "Diversified"


def portfolio_health_report(price_dfs: dict[str, pd.DataFrame], weights: dict[str, float]) -> dict[str, Any]:
    symbols = list(price_dfs.keys())
    if not symbols:
        return {"error": "No holdings provided"}

    sector_exposure: dict[str, float] = {}
    for symbol, weight in weights.items():
        sector = _sector_guess(symbol)
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + float(weight)

    diversification = max(0.0, min(100.0, len(symbols) * 12 - max(sector_exposure.values()) * 35))

    closes = pd.DataFrame({symbol: df["Close"] for symbol, df in price_dfs.items()}).dropna()
    returns = closes.pct_change().dropna()
    weight_vector = np.array([weights.get(symbol, 0.0) for symbol in closes.columns], dtype=float)
    if weight_vector.sum() == 0:
        weight_vector = np.ones(len(closes.columns)) / max(len(closes.columns), 1)
    weight_vector = weight_vector / weight_vector.sum()
    portfolio_returns = returns.values @ weight_vector
    volatility = float(np.std(portfolio_returns) * np.sqrt(252) * 100) if len(portfolio_returns) else 0.0

    target_weight = 1.0 / len(symbols)
    rebalance = {
        symbol: round(target_weight - float(weights.get(symbol, 0.0)), 4)
        for symbol in symbols
        if abs(float(weights.get(symbol, 0.0)) - target_weight) > 0.05
    }

    risk_allocation = {
        symbol: round(float(weights.get(symbol, 0.0)) * volatility / 100.0, 4)
        for symbol in symbols
    }

    return {
        "diversification_score": round(diversification, 2),
        "sector_exposure": {key: round(value * 100, 2) for key, value in sector_exposure.items()},
        "portfolio_volatility": round(volatility, 2),
        "risk_allocation": risk_allocation,
        "rebalancing_suggestions": rebalance,
    }


def sip_simulation(monthly_investment: float, annual_return_pct: float, years: int) -> dict[str, Any]:
    monthly_rate = annual_return_pct / 12 / 100
    months = years * 12
    value = 0.0
    invested = 0.0
    for _ in range(months):
        value = (value + monthly_investment) * (1 + monthly_rate)
        invested += monthly_investment
    return {
        "invested_amount": round(invested, 2),
        "estimated_value": round(value, 2),
        "wealth_gain": round(value - invested, 2),
    }
