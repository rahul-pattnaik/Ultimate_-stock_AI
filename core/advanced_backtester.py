from __future__ import annotations

from typing import Callable

import pandas as pd

from .backtester import Backtester


def compare_strategies(
    df: pd.DataFrame,
    strategies: dict[str, Callable[[pd.DataFrame], str]],
    initial_capital: float,
    position_size: float,
    stop_loss: float,
    take_profit: float,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name, strategy_fn in strategies.items():
        bt = Backtester(
            df=df,
            signal_fn=strategy_fn,
            initial_capital=initial_capital,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        result = bt.run()
        metrics = result.get("metrics", {})
        results[name] = {
            "final_equity": result.get("final_equity"),
            "cagr": metrics.get("cagr_pct"),
            "win_rate": metrics.get("win_rate_pct"),
            "max_drawdown": metrics.get("max_drawdown_pct"),
            "sharpe": metrics.get("sharpe_ratio"),
            "total_return": metrics.get("total_return_pct"),
            "trades": metrics.get("total_trades"),
        }
    return results
