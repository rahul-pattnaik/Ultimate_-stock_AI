# portfolio/optimizer.py
# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Optimizer
# Methods: Mean-Variance (Markowitz) · Max Sharpe · Min Volatility · Equal Weight
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Optional


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from daily returns."""
    returns = np.asarray(returns, dtype=float)
    excess  = returns - risk_free / 252
    std     = np.std(excess)
    if std < 1e-9:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    returns   = np.asarray(returns, dtype=float)
    excess    = returns - risk_free / 252
    downside  = excess[excess < 0]
    down_std  = np.std(downside) if len(downside) > 1 else 1e-9
    return float(np.mean(excess) / down_std * np.sqrt(252))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a negative percentage."""
    equity = np.asarray(equity_curve, dtype=float)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-9)
    return float(dd.min()) * 100


def portfolio_returns(weights: np.ndarray,
                      returns_matrix: np.ndarray) -> np.ndarray:
    """Daily portfolio returns given weights and (n_days × n_assets) matrix."""
    return returns_matrix @ weights


def portfolio_volatility(weights: np.ndarray,
                         cov_matrix: np.ndarray) -> float:
    """Annualised portfolio volatility."""
    return float(np.sqrt(weights @ cov_matrix @ weights * 252)) * 100


def equal_weight_portfolio(symbols: list) -> dict:
    """Simple equal-weight allocation."""
    n = len(symbols)
    w = 1.0 / n
    return {sym: round(w, 4) for sym in symbols}


def optimize_portfolio(price_dfs: dict,
                       method: str = "max_sharpe",
                       n_simulations: int = 2000,
                       risk_free: float = 0.05) -> dict:
    """
    Monte Carlo portfolio optimiser.

    Args:
        price_dfs    : {symbol: pd.DataFrame with 'Close' column}
        method       : "max_sharpe" | "min_volatility" | "equal_weight"
        n_simulations: number of random portfolios to simulate
        risk_free    : annual risk-free rate (default 5% for India)

    Returns dict with optimal weights, expected return, volatility, Sharpe.
    """
    if not price_dfs:
        return {"error": "No price data provided"}

    symbols = list(price_dfs.keys())

    if method == "equal_weight":
        return {
            "method":  "Equal Weight",
            "weights": equal_weight_portfolio(symbols),
        }

    # Build aligned returns matrix
    closes  = pd.DataFrame({sym: df["Close"] for sym, df in price_dfs.items()})
    closes  = closes.dropna()
    returns = closes.pct_change().dropna()

    if len(returns) < 30:
        return {"error": "Need at least 30 days of overlapping data"}

    mu  = returns.mean().values * 252          # annualised expected returns
    cov = returns.cov().values  * 252          # annualised covariance
    n   = len(symbols)

    best_score = -np.inf if method == "max_sharpe" else np.inf
    best_weights = np.ones(n) / n

    for _ in range(n_simulations):
        w  = np.random.dirichlet(np.ones(n))
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w)) * 100
        sr  = (ret * 100 - risk_free) / (vol + 1e-9)

        if method == "max_sharpe":
            if sr > best_score:
                best_score   = sr
                best_weights = w
        elif method == "min_volatility":
            if vol < best_score:
                best_score   = vol
                best_weights = w

    best_ret = float(best_weights @ mu) * 100
    best_vol = float(np.sqrt(best_weights @ cov @ best_weights)) * 100
    best_sr  = (best_ret - risk_free) / (best_vol + 1e-9)

    return {
        "method":            method.replace("_", " ").title(),
        "weights":           {sym: round(float(w), 4)
                              for sym, w in zip(symbols, best_weights)},
        "expected_return":   f"{best_ret:.2f}%",
        "volatility":        f"{best_vol:.2f}%",
        "sharpe_ratio":      round(best_sr, 3),
        "simulations_run":   n_simulations,
    }
