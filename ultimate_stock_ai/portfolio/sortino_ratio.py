# portfolio/sortino_ratio.py
import numpy as np


def sortino_ratio(returns, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio — penalises only downside volatility."""
    returns  = np.asarray(returns, dtype=float)
    excess   = returns - risk_free / 252
    downside = excess[excess < 0]
    down_std = np.std(downside) if len(downside) > 1 else 1e-9
    return float(np.mean(excess) / down_std * np.sqrt(252))
