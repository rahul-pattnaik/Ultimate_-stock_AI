# core/backtester.py
# ─────────────────────────────────────────────────────────────────────────────
# Strategy Backtester
# Supports: signal-based entry/exit · stop-loss · take-profit · position sizing
# Metrics : CAGR · Sharpe · Max Drawdown · Win Rate · Profit Factor · Calmar
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import (BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION,
                     BACKTEST_SLIPPAGE, BACKTEST_POSITION_SIZE,
                     BACKTEST_STOP_LOSS, BACKTEST_TAKE_PROFIT)


# ── Trade Record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  str
    exit_date:   str   = ""
    entry_price: float = 0.0
    exit_price:  float = 0.0
    shares:      int   = 0
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    exit_reason: str   = ""   # "signal" | "stop_loss" | "take_profit" | "end"


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    """
    Event-driven backtester for rule-based trading strategies.

    Usage:
        bt = Backtester(df, signal_fn)
        result = bt.run()
        bt.print_report(result)

    Args:
        df              : OHLCV DataFrame (yfinance format)
        signal_fn       : callable(df_slice) → "BUY" | "SELL" | "HOLD"
        initial_capital : starting cash
        commission      : % per trade (0.001 = 0.1%)
        slippage        : % price slippage on execution
        position_size   : fraction of capital per trade (0.1 = 10%)
        stop_loss       : max loss before auto-exit (0.05 = 5%)
        take_profit     : profit target for auto-exit (0.15 = 15%)
    """

    def __init__(
        self,
        df:             pd.DataFrame,
        signal_fn:      Callable,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission:     float  = BACKTEST_COMMISSION,
        slippage:       float  = BACKTEST_SLIPPAGE,
        position_size:  float  = BACKTEST_POSITION_SIZE,
        stop_loss:      float  = BACKTEST_STOP_LOSS,
        take_profit:    float  = BACKTEST_TAKE_PROFIT,
    ):
        self.df             = df.copy().dropna()
        self.signal_fn      = signal_fn
        self.capital        = initial_capital
        self.initial_capital = initial_capital
        self.commission     = commission
        self.slippage       = slippage
        self.position_size  = position_size
        self.stop_loss      = stop_loss
        self.take_profit    = take_profit

    # ── Execution Helpers ─────────────────────────────────────────────

    def _buy_price(self, price: float) -> float:
        return price * (1 + self.slippage)

    def _sell_price(self, price: float) -> float:
        return price * (1 - self.slippage)

    def _commission_cost(self, price: float, shares: int) -> float:
        return price * shares * self.commission

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the backtest. Returns full result dict with trades and metrics.
        """
        df           = self.df
        equity_curve = [self.initial_capital]
        dates        = [str(df.index[0].date())]
        trades       = []

        cash         = self.initial_capital
        position     = 0      # shares held
        entry_price  = 0.0
        entry_date   = ""

        # Warm-up: need at least 50 bars before generating signals
        warmup = min(50, len(df) // 4)

        for i in range(warmup, len(df)):
            bar        = df.iloc[i]
            close      = float(bar["Close"])
            date_str   = str(df.index[i].date())
            df_slice   = df.iloc[:i + 1]   # signal sees only past data

            portfolio_val = cash + position * close
            equity_curve.append(portfolio_val)
            dates.append(date_str)

            # ── Stop Loss / Take Profit ────────────────────────────────
            if position > 0:
                gain_pct = (close - entry_price) / entry_price

                if gain_pct <= -self.stop_loss:
                    # Stop loss hit
                    sell_p = self._sell_price(close)
                    proceeds = position * sell_p - self._commission_cost(sell_p, position)
                    pnl = proceeds - position * entry_price
                    trades.append(Trade(
                        entry_date=entry_date, exit_date=date_str,
                        entry_price=round(entry_price, 2),
                        exit_price=round(sell_p, 2), shares=position,
                        pnl=round(pnl, 2),
                        pnl_pct=round(gain_pct * 100, 2),
                        exit_reason="stop_loss",
                    ))
                    cash    += proceeds
                    position = 0
                    continue

                if gain_pct >= self.take_profit:
                    # Take profit hit
                    sell_p = self._sell_price(close)
                    proceeds = position * sell_p - self._commission_cost(sell_p, position)
                    pnl = proceeds - position * entry_price
                    trades.append(Trade(
                        entry_date=entry_date, exit_date=date_str,
                        entry_price=round(entry_price, 2),
                        exit_price=round(sell_p, 2), shares=position,
                        pnl=round(pnl, 2),
                        pnl_pct=round(gain_pct * 100, 2),
                        exit_reason="take_profit",
                    ))
                    cash    += proceeds
                    position = 0
                    continue

            # ── Signal ────────────────────────────────────────────────
            try:
                signal = self.signal_fn(df_slice)
                # Normalise: accept string or dict with 'signal' key
                if isinstance(signal, dict):
                    signal = signal.get("signal", "HOLD")
                signal = str(signal).upper()
            except Exception:
                signal = "HOLD"

            # Normalise to BUY / SELL / HOLD
            if any(k in signal for k in ["STRONG BUY", "BUY"]):
                signal = "BUY"
            elif any(k in signal for k in ["STRONG SELL", "SELL"]):
                signal = "SELL"
            else:
                signal = "HOLD"

            # ── Execute ────────────────────────────────────────────────
            if signal == "BUY" and position == 0:
                buy_p    = self._buy_price(close)
                alloc    = cash * self.position_size
                shares   = int(alloc / buy_p)
                if shares > 0:
                    cost     = shares * buy_p + self._commission_cost(buy_p, shares)
                    cash    -= cost
                    position  = shares
                    entry_price = buy_p
                    entry_date  = date_str

            elif signal == "SELL" and position > 0:
                sell_p   = self._sell_price(close)
                proceeds = position * sell_p - self._commission_cost(sell_p, position)
                gain_pct = (sell_p - entry_price) / entry_price
                pnl      = proceeds - position * entry_price
                trades.append(Trade(
                    entry_date=entry_date, exit_date=date_str,
                    entry_price=round(entry_price, 2),
                    exit_price=round(sell_p, 2), shares=position,
                    pnl=round(pnl, 2),
                    pnl_pct=round(gain_pct * 100, 2),
                    exit_reason="signal",
                ))
                cash    += proceeds
                position = 0

        # ── Close open position at end ─────────────────────────────────
        if position > 0:
            final_price = float(df["Close"].iloc[-1])
            sell_p      = self._sell_price(final_price)
            proceeds    = position * sell_p - self._commission_cost(sell_p, position)
            gain_pct    = (sell_p - entry_price) / entry_price
            pnl         = proceeds - position * entry_price
            trades.append(Trade(
                entry_date=entry_date, exit_date=str(df.index[-1].date()),
                entry_price=round(entry_price, 2),
                exit_price=round(sell_p, 2), shares=position,
                pnl=round(pnl, 2),
                pnl_pct=round(gain_pct * 100, 2),
                exit_reason="end",
            ))
            cash += proceeds
            position = 0

        final_equity = cash
        equity_arr   = np.array(equity_curve)

        return {
            "trades":         trades,
            "equity_curve":   equity_curve,
            "dates":          dates,
            "final_equity":   round(final_equity, 2),
            "metrics":        self._compute_metrics(equity_arr, trades),
        }

    # ── Metrics ───────────────────────────────────────────────────────

    def _compute_metrics(self, equity: np.ndarray, trades: list) -> dict:
        """Compute all performance metrics from equity curve and trades."""
        start = equity[0]
        end   = equity[-1]
        n_days = len(equity)

        # CAGR
        years = n_days / 252
        cagr  = ((end / start) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        # Daily returns
        returns = np.diff(equity) / equity[:-1]

        # Sharpe (annualised, assume 0% risk-free)
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / (peak + 1e-9)
        max_dd = float(dd.min()) * 100

        # Calmar = CAGR / |Max Drawdown|
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Trade stats
        n_trades = len(trades)
        if n_trades > 0:
            pnls     = [t.pnl for t in trades]
            wins     = [p for p in pnls if p > 0]
            losses   = [p for p in pnls if p <= 0]
            win_rate = len(wins) / n_trades * 100
            avg_win  = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            profit_factor = (sum(wins) / abs(sum(losses))
                             if losses and sum(losses) != 0 else float("inf"))
            best_trade  = max(pnls)
            worst_trade = min(pnls)

            # Exit reason breakdown
            exit_reasons = {}
            for t in trades:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
            best_trade = worst_trade = 0.0
            exit_reasons = {}

        total_return = (end - start) / start * 100

        return {
            "total_return_pct":  round(total_return, 2),
            "cagr_pct":          round(cagr, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "max_drawdown_pct":  round(max_dd, 2),
            "calmar_ratio":      round(calmar, 3),
            "n_trades":          n_trades,
            "win_rate_pct":      round(win_rate, 2),
            "avg_win":           round(avg_win, 2),
            "avg_loss":          round(avg_loss, 2),
            "profit_factor":     round(profit_factor, 3),
            "best_trade":        round(best_trade, 2),
            "worst_trade":       round(worst_trade, 2),
            "exit_reasons":      exit_reasons,
            "initial_capital":   round(start, 2),
            "final_equity":      round(end, 2),
        }

    # ── Report ────────────────────────────────────────────────────────

    @staticmethod
    def print_report(result: dict) -> None:
        """Pretty-print backtest results to terminal."""
        m = result["metrics"]
        sep = "=" * 50
        print(f"\n{sep}")
        print("       BACKTEST RESULTS")
        print(sep)
        print(f"  Initial Capital : ₹{m['initial_capital']:,.2f}")
        print(f"  Final Equity    : ₹{m['final_equity']:,.2f}")
        print(f"  Total Return    : {m['total_return_pct']:+.2f}%")
        print(f"  CAGR            : {m['cagr_pct']:+.2f}%")
        print(sep)
        print(f"  Sharpe Ratio    : {m['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown    : {m['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio    : {m['calmar_ratio']:.3f}")
        print(sep)
        print(f"  Total Trades    : {m['n_trades']}")
        print(f"  Win Rate        : {m['win_rate_pct']:.2f}%")
        print(f"  Profit Factor   : {m['profit_factor']:.3f}")
        print(f"  Avg Win         : ₹{m['avg_win']:,.2f}")
        print(f"  Avg Loss        : ₹{m['avg_loss']:,.2f}")
        print(f"  Best Trade      : ₹{m['best_trade']:,.2f}")
        print(f"  Worst Trade     : ₹{m['worst_trade']:,.2f}")
        print(sep)
        if m["exit_reasons"]:
            print("  Exit Breakdown  :")
            for reason, count in m["exit_reasons"].items():
                print(f"    {reason:<15} : {count}")
        print(sep)

    @staticmethod
    def trade_log(result: dict) -> pd.DataFrame:
        """Return trades as a DataFrame for further analysis."""
        trades = result["trades"]
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "entry_date":  t.entry_date,
            "exit_date":   t.exit_date,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "shares":      t.shares,
            "pnl":         t.pnl,
            "pnl_pct":     t.pnl_pct,
            "exit_reason": t.exit_reason,
        } for t in trades])
