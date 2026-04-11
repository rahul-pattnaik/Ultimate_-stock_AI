"""
Advanced Swing Trading Scanner
Features: Multi-timeframe analysis, risk/reward calculation, pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SwingSignalStrength(Enum):
    """Signal strength classification"""
    VERY_WEAK = 0.0
    WEAK = 0.25
    NEUTRAL = 0.5
    STRONG = 0.75
    VERY_STRONG = 1.0


@dataclass
class SwingTradeSetup:
    """Comprehensive swing trade setup"""
    symbol: str
    signal: str
    strength: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: float  # 0-100
    reasons: List[str]
    timeframe_agreement: float  # % of timeframes in agreement
    volume_confirmation: bool
    technical_score: float  # 0-100 based on indicators


class SwingScanner:
    """Advanced swing trading analysis"""

    def __init__(self, min_confidence: float = 60.0):
        self.min_confidence = min_confidence
        self.atr_period = 14
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, 
                     close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return pd.Series(tr).rolling(window=period).mean().values

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(window=period).mean().values
        avg_loss = pd.Series(losses).rolling(window=period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([np.full(1, np.nan), rsi])

    def calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        ema_fast = pd.Series(prices).ewm(span=self.macd_fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=self.macd_slow).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=self.macd_signal).mean().values
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def identify_swing_highs(self, high: np.ndarray, lookback: int = 5) -> np.ndarray:
        """Identify swing highs (local maxima)"""
        swing_highs = np.zeros(len(high), dtype=bool)
        for i in range(lookback, len(high) - lookback):
            if high[i] == np.max(high[max(0, i-lookback):i+lookback+1]):
                swing_highs[i] = True
        return swing_highs

    def identify_swing_lows(self, low: np.ndarray, lookback: int = 5) -> np.ndarray:
        """Identify swing lows (local minima)"""
        swing_lows = np.zeros(len(low), dtype=bool)
        for i in range(lookback, len(low) - lookback):
            if low[i] == np.min(low[max(0, i-lookback):i+lookback+1]):
                swing_lows[i] = True
        return swing_lows

    def detect_double_bottom(self, low: np.ndarray, lookback: int = 20) -> Tuple[bool, float]:
        """Detect double bottom pattern"""
        if len(low) < lookback:
            return False, 0.0
        
        recent = low[-lookback:]
        min_price = np.min(recent)
        min_count = np.sum(np.isclose(recent, min_price, rtol=0.02))
        
        if min_count >= 2:
            # Find two distinct lows
            min_indices = np.where(np.isclose(recent, min_price, rtol=0.02))[0]
            if len(min_indices) >= 2:
                # Check if they're separated by at least 5 bars
                if min_indices[-1] - min_indices[0] >= 5:
                    return True, min_price
        
        return False, 0.0

    def detect_ascending_triangle(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[bool, float]:
        """Detect ascending triangle breakout setup"""
        if len(df) < lookback:
            return False, 0.0
        
        recent = df.iloc[-lookback:]
        
        # Find higher lows trend
        lows = recent['low'].values
        highs = recent['high'].values
        
        # Check for rising lows
        low_trend = np.polyfit(np.arange(len(lows)), lows, 1)[0]
        # Check for relatively flat highs
        high_std = np.std(highs[-10:])
        
        if low_trend > 0 and high_std < np.mean(highs) * 0.02:
            breakout_level = np.mean(highs[-5:])
            return True, breakout_level
        
        return False, 0.0

    def detect_head_shoulders(self, high: np.ndarray, lookback: int = 30) -> Tuple[bool, float]:
        """Detect head and shoulders reversal pattern"""
        if len(high) < lookback:
            return False, 0.0
        
        recent = high[-lookback:]
        
        # Find peak (head)
        head_idx = np.argmax(recent)
        
        # Check for shoulders (local maxima on both sides)
        if head_idx > 5 and head_idx < len(recent) - 5:
            left_shoulder_idx = np.argmax(recent[:head_idx-2])
            right_shoulder_idx = np.argmax(recent[head_idx+2:]) + head_idx + 2
            
            head_price = recent[head_idx]
            left_shoulder = recent[left_shoulder_idx]
            right_shoulder = recent[right_shoulder_idx]
            
            # Check pattern: shoulders similar height, lower than head
            if (np.isclose(left_shoulder, right_shoulder, rtol=0.03) and 
                left_shoulder < head_price * 0.95):
                neckline = min(recent[head_idx-1], recent[head_idx+1])
                return True, neckline
        
        return False, 0.0

    def calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
        """Calculate dynamic support and resistance levels"""
        recent = df.iloc[-lookback:]
        
        support = recent['low'].min()
        resistance = recent['high'].max()
        
        # Weighted by recency (more recent = more weight)
        weights = np.linspace(0.5, 1.5, lookback)
        weighted_support = np.average(recent['low'].values, weights=weights)
        weighted_resistance = np.average(recent['high'].values, weights=weights)
        
        return weighted_support, weighted_resistance

    def confirm_volume(self, volume: np.ndarray, lookback: int = 20) -> bool:
        """Check if recent volume confirms price action"""
        if len(volume) < lookback:
            return False
        
        avg_volume = np.mean(volume[-lookback:-1])
        recent_volume = volume[-1]
        
        # Recent volume should be above average for confirmation
        return recent_volume > avg_volume * 0.9

    def scan_swing_setup(self, df: pd.DataFrame, symbol: str) -> Optional[SwingTradeSetup]:
        """Comprehensive swing trade setup analysis"""
        
        try:
            if len(df) < 50:
                return None

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            # Calculate indicators
            atr = self.calculate_atr(high, low, close)
            rsi = self.calculate_rsi(close)
            macd, macd_signal, macd_hist = self.calculate_macd(close)
            
            current_price = close[-1]
            current_atr = atr[-1]
            current_rsi = rsi[-1]
            current_macd_hist = macd_hist[-1]
            prev_macd_hist = macd_hist[-2]
            
            # Calculate support/resistance
            support, resistance = self.calculate_support_resistance(df)
            
            # Pattern detection
            double_bottom, db_price = self.detect_double_bottom(low)
            ascending_tri, tri_level = self.detect_ascending_triangle(df)
            head_shoulders, hs_neckline = self.detect_head_shoulders(high)
            
            # Volume confirmation
            vol_confirm = self.confirm_volume(volume)
            
            # Swing lows/highs
            swing_lows = self.identify_swing_lows(low)
            swing_highs = self.identify_swing_highs(high)
            
            # Initialize scoring
            signal = "No Setup"
            strength = 0.0
            confidence = 0.0
            reasons = []
            
            # ========== BUY SIGNAL LOGIC ==========
            
            buy_signals = 0
            buy_total = 0
            
            # Signal 1: Price above key moving averages
            sma20 = pd.Series(close).rolling(20).mean().values[-1]
            sma50 = pd.Series(close).rolling(50).mean().values[-1]
            
            if current_price > sma50 > sma20:
                buy_signals += 1
                reasons.append("Price above 50/20 MAs")
            buy_total += 1
            
            # Signal 2: RSI reversal from oversold
            if current_rsi < 40 and prev_macd_hist < 0 and current_macd_hist > prev_macd_hist:
                buy_signals += 1
                reasons.append("RSI bounce + MACD crossover")
            buy_total += 1
            
            # Signal 3: Double bottom pattern
            if double_bottom:
                buy_signals += 1
                reasons.append("Double bottom formed")
            buy_total += 1
            
            # Signal 4: Ascending triangle breakout
            if ascending_tri and current_price > tri_level:
                buy_signals += 1
                reasons.append("Ascending triangle breakout")
            buy_total += 1
            
            # Signal 5: MACD bullish crossover
            if prev_macd_hist < 0 and current_macd_hist > 0:
                buy_signals += 1
                reasons.append("MACD bullish crossover")
            buy_total += 1
            
            # Signal 6: Support bounce
            if current_price > support and current_price < support * 1.02:
                buy_signals += 1
                reasons.append("Support level bounce")
            buy_total += 1
            
            # Signal 7: Volume confirmation
            if vol_confirm:
                buy_signals += 1
                reasons.append("Volume confirmation")
            buy_total += 1
            
            # Calculate confidence
            confidence = (buy_signals / buy_total) * 100 if buy_total > 0 else 0
            strength = buy_signals / buy_total if buy_total > 0 else 0
            
            # Determine signal type
            if confidence >= 60 and buy_signals >= 4:
                signal = "Strong Swing Buy"
            elif confidence >= 50 and buy_signals >= 3:
                signal = "Swing Buy"
            elif confidence >= 40 and buy_signals >= 2:
                signal = "Weak Swing Buy"
            
            # ========== SELL SIGNAL LOGIC ==========
            
            sell_signals = 0
            sell_total = 0
            
            # Signal 1: Price below key MAs
            if current_price < sma20 < sma50:
                sell_signals += 1
                reasons.append("Price below 20/50 MAs")
            sell_total += 1
            
            # Signal 2: RSI overbought rejection
            if current_rsi > 70 and current_macd_hist < 0:
                sell_signals += 1
                reasons.append("RSI overbought + MACD bearish")
            sell_total += 1
            
            # Signal 3: Head and shoulders
            if head_shoulders and current_price < hs_neckline:
                sell_signals += 1
                reasons.append("Head & shoulders breakdown")
            sell_total += 1
            
            # Calculate sell confidence
            sell_confidence = (sell_signals / sell_total) * 100 if sell_total > 0 else 0
            
            if sell_confidence > confidence and sell_confidence >= 60:
                signal = "Swing Sell"
                confidence = sell_confidence
                strength = sell_signals / sell_total if sell_total > 0 else 0
            
            # If no strong setup, return None
            if confidence < self.min_confidence:
                return None
            
            # ========== RISK/REWARD CALCULATION ==========
            
            stop_loss = support - current_atr
            
            if "Buy" in signal:
                take_profit = resistance + (current_atr * 2)
                risk_amount = current_price - stop_loss
                reward_amount = take_profit - current_price
            else:  # Sell
                take_profit = support - (current_atr * 2)
                risk_amount = stop_loss - current_price
                reward_amount = current_price - take_profit
            
            risk_reward_ratio = reward_amount / (risk_amount + 1e-10)
            
            # Return setup if risk/reward is favorable
            if risk_reward_ratio < 1.5:
                return None
            
            return SwingTradeSetup(
                symbol=symbol,
                signal=signal,
                strength=strength,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                confidence=confidence,
                reasons=reasons,
                timeframe_agreement=strength * 100,
                volume_confirmation=vol_confirm,
                technical_score=confidence
            )

        except Exception as e:
            logger.error(f"Error scanning swing setup for {symbol}: {e}")
            return None

    def scan_multiple(self, symbols: Dict[str, pd.DataFrame]) -> List[SwingTradeSetup]:
        """Scan multiple stocks for swing setups"""
        setups = []
        for symbol, df in symbols.items():
            setup = self.scan_swing_setup(df, symbol)
            if setup:
                setups.append(setup)
        
        # Sort by confidence
        setups.sort(key=lambda x: x.confidence, reverse=True)
        return setups

    def format_setup(self, setup: SwingTradeSetup) -> str:
        """Format setup for display"""
        return f"""
╔════════════════════════════════════════════════════════╗
║ {setup.symbol:40} │ {setup.signal:10} ║
╠════════════════════════════════════════════════════════╣
║ Entry Price:     ${setup.entry_price:8.2f}      │ Confidence: {setup.confidence:5.1f}%  ║
║ Stop Loss:       ${setup.stop_loss:8.2f}      │ Risk/Reward: {setup.risk_reward_ratio:5.2f}x   ║
║ Take Profit:     ${setup.take_profit:8.2f}      │ Tech Score: {setup.technical_score:5.1f}%    ║
╠════════════════════════════════════════════════════════╣
║ Reasons:                                               ║
"""  + "".join([f"║   • {reason:56}║\n" for reason in setup.reasons]) + f"""╚════════════════════════════════════════════════════════╝
"""
