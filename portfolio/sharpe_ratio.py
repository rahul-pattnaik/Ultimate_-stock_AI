# portfolio/sharpe_ratio.py
import numpy as np


def sharpe_ratio(returns, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from daily returns array."""
    returns = np.asarray(returns, dtype=float)
    excess  = returns - risk_free / 252
    std     = np.std(excess)
    return float(np.mean(excess) / std * np.sqrt(252)) if std > 1e-9 else 0.0
