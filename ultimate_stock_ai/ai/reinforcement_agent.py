# ai/reinforcement_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning Trading Agent
# Actions : 0 = HOLD · 1 = BUY · 2 = SELL
# State   : discretised RSI + MACD direction + price vs MA20
# Reward  : P&L-based with drawdown penalty
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import json
import os
from typing import Optional


ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2
ACTION_NAMES = {0: "HOLD ➡️", 1: "BUY 📈", 2: "SELL 📉"}


class TradingAgent:
    """
    Tabular Q-Learning agent for stock trading.

    State space  : (rsi_zone, macd_dir, price_vs_ma, position)
    Action space : HOLD | BUY | SELL
    """

    def __init__(
        self,
        learning_rate:   float = 0.1,
        discount_factor: float = 0.95,
        epsilon:         float = 1.0,       # exploration rate
        epsilon_min:     float = 0.05,
        epsilon_decay:   float = 0.995,
    ):
        self.lr      = learning_rate
        self.gamma   = discount_factor
        self.epsilon = epsilon
        self.eps_min = epsilon_min
        self.eps_dec = epsilon_decay

        self.q_table:  dict[str, list[float]] = {}
        self.position: int   = 0             # 0 = no position, 1 = long
        self.buy_price: float = 0.0

        # Stats
        self.total_trades = 0
        self.wins         = 0
        self.total_pnl    = 0.0

    # ── State Encoding ────────────────────────────────────────────────

    @staticmethod
    def encode_state(rsi: float, macd_hist: float,
                     price: float, ma20: float,
                     position: int) -> str:
        """
        Discretise continuous indicators into a hashable state string.
        Keeps the Q-table small and generalisable.
        """
        # RSI zones: oversold(0) / neutral(1) / overbought(2)
        rsi_zone = 0 if rsi < 35 else 2 if rsi > 65 else 1

        # MACD histogram direction: bullish(1) / bearish(0)
        macd_dir = 1 if macd_hist > 0 else 0

        # Price vs MA20: above(1) / below(0)
        price_zone = 1 if price > ma20 else 0

        return f"{rsi_zone}_{macd_dir}_{price_zone}_{position}"

    def _get_q(self, state: str) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    # ── Action Selection ──────────────────────────────────────────────

    def choose_action(self, state: str, training: bool = False) -> int:
        """
        Epsilon-greedy during training; greedy during inference.
        Also enforces trading logic:
          - Can't SELL if not holding
          - Can't BUY if already long
        """
        if training and np.random.rand() < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            q = self._get_q(state)
            action = int(np.argmax(q))

        # Enforce validity
        if action == ACTION_BUY  and self.position == 1: action = ACTION_HOLD
        if action == ACTION_SELL and self.position == 0: action = ACTION_HOLD

        return action

    # ── Reward Function ───────────────────────────────────────────────

    def compute_reward(self, action: int, current_price: float,
                       next_price: float) -> float:
        """
        Reward design:
          BUY          : small negative (commission simulation)
          SELL (profit): positive P&L
          SELL (loss)  : negative P&L (amplified)
          HOLD (long)  : unrealised P&L nudge
          HOLD (flat)  : tiny negative (opportunity cost)
        """
        commission = 0.001  # 0.1% per trade

        if action == ACTION_BUY:
            self.position  = 1
            self.buy_price = current_price
            return -commission * current_price

        elif action == ACTION_SELL and self.position == 1:
            pnl = (current_price - self.buy_price) / self.buy_price
            self.position  = 0
            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.wins += 1
            self.buy_price = 0.0
            penalty = 1.5 if pnl < 0 else 1.0   # penalise losses more
            return pnl * 100 * penalty

        elif action == ACTION_HOLD and self.position == 1:
            # Unrealised nudge — encourage holding winners
            unrealised = (current_price - self.buy_price) / self.buy_price
            return unrealised * 10

        else:
            return -0.01   # small cost for idle cash

    # ── Q-Learning Update ─────────────────────────────────────────────

    def update(self, state: str, action: int, reward: float,
               next_state: str) -> None:
        """Bellman equation Q-update."""
        q_current = self._get_q(state)
        q_next    = self._get_q(next_state)

        target = reward + self.gamma * max(q_next)
        q_current[action] += self.lr * (target - q_current[action])

        # Decay exploration
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_dec

    # ── Training Loop ─────────────────────────────────────────────────

    def train(self, df, episodes: int = 50) -> dict:
        """
        Train the agent on historical OHLCV data.

        Args:
            df       : yfinance DataFrame
            episodes : number of full-history passes

        Returns training summary dict.
        """
        import pandas as pd

        df    = df.copy().dropna()
        close = df["Close"].values.flatten().astype(float)
        ma20  = pd.Series(close).rolling(20).mean().values

        # Pre-compute RSI + MACD for speed
        def rsi_series(c, p=14):
            d = np.diff(c)
            r = []
            for i in range(p, len(c)):
                g = np.where(d[i - p:i] > 0,  d[i - p:i], 0).mean()
                l = np.where(d[i - p:i] < 0, -d[i - p:i], 0).mean()
                r.append(50.0 if l == 0 else 100 - 100 / (1 + g / l))
            return [50.0] * p + r

        def macd_hist(c):
            def ema(a, s):
                k, o = 2/(s+1), [a[0]]
                for v in a[1:]: o.append(v*k + o[-1]*(1-k))
                return np.array(o)
            m = ema(c, 12) - ema(c, 26)
            return m - ema(m, 9)

        rsi_vals  = rsi_series(close)
        hist_vals = macd_hist(close)

        start = 26  # skip warm-up period
        episode_rewards = []

        for ep in range(episodes):
            self.position  = 0
            self.buy_price = 0.0
            ep_reward      = 0.0

            for i in range(start, len(close) - 1):
                state = self.encode_state(
                    rsi_vals[i], hist_vals[i],
                    close[i], ma20[i] if not np.isnan(ma20[i]) else close[i],
                    self.position
                )
                action = self.choose_action(state, training=True)
                reward = self.compute_reward(action, close[i], close[i + 1])

                next_state = self.encode_state(
                    rsi_vals[i + 1], hist_vals[i + 1],
                    close[i + 1],
                    ma20[i + 1] if not np.isnan(ma20[i + 1]) else close[i + 1],
                    self.position
                )
                self.update(state, action, reward, next_state)
                ep_reward += reward

            episode_rewards.append(round(ep_reward, 2))

        win_rate = (self.wins / self.total_trades * 100
                    if self.total_trades > 0 else 0.0)

        return {
            "episodes":    episodes,
            "q_states":    len(self.q_table),
            "total_trades": self.total_trades,
            "win_rate":    f"{win_rate:.1f}%",
            "total_pnl":   f"{self.total_pnl * 100:.2f}%",
            "final_epsilon": round(self.epsilon, 4),
        }

    # ── Inference ─────────────────────────────────────────────────────

    def predict(self, rsi: float, macd_hist: float,
                price: float, ma20: float) -> dict:
        """
        Get the agent's recommended action for current market conditions.
        """
        state  = self.encode_state(rsi, macd_hist, price, ma20, self.position)
        q      = self._get_q(state)
        action = int(np.argmax(q))

        # Mask invalid actions
        if action == ACTION_BUY  and self.position == 1: action = ACTION_HOLD
        if action == ACTION_SELL and self.position == 0: action = ACTION_HOLD

        return {
            "action":     ACTION_NAMES[action],
            "action_id":  action,
            "state":      state,
            "q_values":   {ACTION_NAMES[i]: round(q[i], 4) for i in range(3)},
            "position":   "LONG" if self.position == 1 else "FLAT",
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str = "ai/rl_agent.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "q_table":  self.q_table,
                "epsilon":  self.epsilon,
                "position": self.position,
            }, f)

    def load(self, path: str = "ai/rl_agent.json") -> bool:
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.q_table  = data["q_table"]
        self.epsilon  = data["epsilon"]
        self.position = data["position"]
        return True
