"""
Advanced Breakout Scanner
Features: Multi-level breakouts, pattern recognition, volume confirmation, false breakout detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BreakoutType(Enum):
    """Breakout type classification"""
    RANGE_BREAKOUT = "Range Breakout"
    RESISTANCE_BREAKOUT = "Resistance Breakout"
    SUPPORT_BREAKDOWN = "Support Breakdown"
    CONSOLIDATION_BREAKOUT = "Consolidation Breakout"
    TRIANGLE_BREAKOUT = "Triangle Breakout"


@dataclass
class BreakoutSetup:
    """Comprehensive breakout setup"""
    symbol: str
    breakout_type: str
    signal: str  # 'Breakout', 'Breakdown', 'False Breakout', 'Consolidating'
    confidence: float  # 0-100
    
    # Price levels
    breakout_level: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    
    # Volume confirmation
    volume_confirmation: bool
    volume_ratio: float  # Current vs average
    
    # Strength metrics
    breakout_strength: float  # 0-100
    consolidation_days: int
    range_size: float
    range_percentage: float
    
    # Support/Resistance
    nearest_support: float
    nearest_resistance: float
    
    # False breakout detection
    false_breakout_risk: float  # 0-100
    
    # Technical confirmation
    rsi_confirmation: bool
    macd_confirmation: bool
    trend_confirmation: bool
    
    reasons: List[str]


class BreakoutScanner:
    """Advanced breakout analysis"""

    def __init__(self):
        self.min_confidence = 60.0

    def identify_consolidation(self, high: np.ndarray, low: np.ndarray,
                              lookback: int = 20) -> Tuple[float, float, int]:
        """Identify consolidation range and duration"""
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        
        max_price = np.max(recent_high)
        min_price = np.min(recent_low)
        
        # Find how long range has been tight
        range_tightness = 0
        for i in range(len(recent_high) - 1, 0, -1):
            if recent_high[i] == max_price or recent_low[i] == min_price:
                range_tightness = len(recent_high) - i
            else:
                break
        
        return min_price, max_price, range_tightness

    def calculate_atr(self, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr[-period:])

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD histogram"""
        ema_fast = pd.Series(prices).ewm(span=12).mean().values[-1]
        ema_slow = pd.Series(prices).ewm(span=26).mean().values[-1]
        macd_line = ema_fast - ema_slow
        macd_signal = pd.Series(prices).ewm(span=9).mean().values[-1]
        macd_hist = macd_line - macd_signal
        return macd_line, macd_hist

    def find_dynamic_support_resistance(self, df: pd.DataFrame, 
                                       lookback: int = 60) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels dynamically"""
        recent = df.iloc[-lookback:]
        
        supports = []
        resistances = []
        
        # Find local minima (supports)
        for i in range(5, len(recent) - 5):
            if (recent['low'].iloc[i] < recent['low'].iloc[i-5:i].min() and
                recent['low'].iloc[i] < recent['low'].iloc[i+1:i+6].min()):
                supports.append(recent['low'].iloc[i])
        
        # Find local maxima (resistances)
        for i in range(5, len(recent) - 5):
            if (recent['high'].iloc[i] > recent['high'].iloc[i-5:i].max() and
                recent['high'].iloc[i] > recent['high'].iloc[i+1:i+6].max()):
                resistances.append(recent['high'].iloc[i])
        
        return supports, resistances

    def detect_false_breakout(self, close: np.ndarray, breakout_level: float,
                             lookback: int = 5) -> Tuple[float, bool]:
        """Calculate false breakout risk"""
        if len(close) < lookback:
            return 0.0, False
        
        recent = close[-lookback:]
        
        # Check if price quickly reverses after breakout
        time_above = np.sum(recent > breakout_level)
        
        if time_above <= 2:
            # Quick reversal = high false breakout risk
            return 80.0, True
        elif time_above <= 3:
            return 50.0, True
        else:
            return 20.0, False

    def detect_triangle(self, high: np.ndarray, low: np.ndarray,
                       lookback: int = 20) -> Tuple[bool, float]:
        """Detect ascending/descending triangle"""
        if len(high) < lookback:
            return False, 0.0
        
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        
        # Fit trend lines
        x = np.arange(len(recent_high))
        
        # Highs trend
        high_trend = np.polyfit(x, recent_high, 1)
        # Lows trend
        low_trend = np.polyfit(x, recent_low, 1)
        
        # Triangle: highs descending, lows ascending (or vice versa)
        is_triangle = (high_trend[0] < -0.01 and low_trend[0] > 0.01) or \
                     (high_trend[0] > 0.01 and low_trend[0] < -0.01)
        
        convergence = abs(high_trend[0]) + abs(low_trend[0])
        
        return is_triangle, convergence

    def scan_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[BreakoutSetup]:
        """Comprehensive breakout analysis"""
        
        try:
            if len(df) < 30:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            current_price = close[-1]
            current_high = high[-1]
            current_low = low[-1]
            current_volume = volume[-1]
            
            # Calculate indicators
            atr = self.calculate_atr(high, low, close)
            rsi = self.calculate_rsi(close)
            macd_line, macd_hist = self.calculate_macd(close)
            
            # Identify consolidation
            cons_low, cons_high, cons_days = self.identify_consolidation(high, low)
            
            # Find support/resistance
            supports, resistances = self.find_dynamic_support_resistance(df)
            nearest_support = max(supports) if supports else cons_low
            nearest_resistance = min(resistances) if resistances else cons_high
            
            # Triangle detection
            is_triangle, tri_convergence = self.detect_triangle(high, low)
            
            # Volume confirmation
            avg_volume = np.mean(volume[-20:-1])
            volume_ratio = current_volume / (avg_volume + 1e-10)
            vol_confirm = volume_ratio > 1.3
            
            # Trend confirmation
            sma50 = pd.Series(close).rolling(50).mean().values[-1]
            sma200 = pd.Series(close).rolling(200).mean().values[-1]
            trend_up = current_price > sma50 > sma200
            trend_down = current_price < sma50 < sma200
            
            # RSI and MACD confirmation
            rsi_confirm = (rsi > 50 and macd_hist > 0) or (rsi < 50 and macd_hist < 0)
            macd_confirm = macd_hist > 0 or macd_hist < 0  # Any clear direction
            
            # ========== BREAKOUT SIGNAL LOGIC ==========
            
            reasons = []
            confidence = 0.0
            signal = "Consolidating"
            breakout_type = ""
            breakout_strength = 0.0
            false_breakout_risk = 0.0
            
            # Check for resistance breakout
            if current_price > nearest_resistance * 1.005 and current_price > cons_high:
                breakout_type = BreakoutType.RESISTANCE_BREAKOUT.value
                signal = "Breakout"
                
                reasons.append(f"Broke resistance at ${nearest_resistance:.2f}")
                
                if vol_confirm:
                    reasons.append("Strong volume confirmation")
                    confidence += 15
                else:
                    reasons.append("Weak volume")
                
                if rsi > 50:
                    reasons.append("RSI above 50")
                    confidence += 10
                
                if trend_up:
                    reasons.append("In uptrend")
                    confidence += 20
                elif not trend_down:
                    reasons.append("Neutral trend")
                    confidence += 5
                
                if is_triangle and macd_hist > 0:
                    reasons.append("Triangle breakout + MACD bullish")
                    confidence += 25
                
                confidence += 30  # Base breakout score
                
                # Calculate false breakout risk
                false_risk, is_false = self.detect_false_breakout(close, nearest_resistance)
                false_breakout_risk = false_risk
                
                take_profit = nearest_resistance + (atr * 2)
                stop_loss = cons_low - atr
                
            # Check for support breakdown
            elif current_price < nearest_support * 0.995 and current_price < cons_low:
                breakout_type = BreakoutType.SUPPORT_BREAKDOWN.value
                signal = "Breakdown"
                
                reasons.append(f"Broke support at ${nearest_support:.2f}")
                
                if vol_confirm:
                    reasons.append("Strong volume confirmation")
                    confidence += 15
                
                if rsi < 50:
                    reasons.append("RSI below 50")
                    confidence += 10
                
                if trend_down:
                    reasons.append("In downtrend")
                    confidence += 20
                elif not trend_up:
                    reasons.append("Neutral trend")
                    confidence += 5
                
                if is_triangle and macd_hist < 0:
                    reasons.append("Triangle breakdown + MACD bearish")
                    confidence += 25
                
                confidence += 30
                
                false_risk, is_false = self.detect_false_breakout(close, nearest_support)
                false_breakout_risk = false_risk
                
                take_profit = nearest_support - (atr * 2)
                stop_loss = cons_high + atr
            
            # Check for consolidation with breakout potential
            elif cons_days >= 5 and (current_price > cons_high * 0.98 or current_price < cons_low * 1.02):
                breakout_type = BreakoutType.CONSOLIDATION_BREAKOUT.value
                signal = "Consolidating"
                
                reasons.append(f"Consolidating for {cons_days} days")
                range_size = cons_high - cons_low
                range_pct = (range_size / cons_low) * 100
                
                reasons.append(f"Range: ${range_size:.2f} ({range_pct:.2f}%)")
                
                if is_triangle:
                    reasons.append("Triangle formation - expecting breakout soon")
                    confidence += 35
                    breakout_type = BreakoutType.TRIANGLE_BREAKOUT.value
                
                confidence += 20
                
                # Anticipate breakout direction
                if volume_ratio > 1.2:
                    confidence += 10
                
                take_profit = cons_high + range_size if current_price > (cons_high + cons_low) / 2 else cons_low - range_size
                stop_loss = cons_low if current_price > (cons_high + cons_low) / 2 else cons_high
            
            # No clear breakout signal
            else:
                if len(reasons) < 2:
                    return None
            
            # Risk/reward calculation
            if "Breakout" in signal or "Breakdown" in signal:
                risk_amount = abs(current_price - stop_loss)
                reward_amount = abs(take_profit - current_price)
                risk_reward_ratio = reward_amount / (risk_amount + 1e-10)
            else:
                risk_reward_ratio = 0.0
                take_profit = cons_high + (cons_high - cons_low)
                stop_loss = cons_low - (cons_high - cons_low)
            
            # Determine breakout strength
            if "Breakout" in signal or "Breakdown" in signal:
                price_from_level = abs(current_price - (nearest_resistance if "Breakout" in signal else nearest_support))
                distance_from_support = abs(current_price - nearest_support) if "Breakout" in signal else abs(nearest_resistance - current_price)
                
                if price_from_level > atr * 0.5:
                    breakout_strength = min(80 + volume_ratio * 10, 100)
                else:
                    breakout_strength = min(50 + volume_ratio * 10, 80)
            else:
                breakout_strength = 30
            
            # Filter by confidence
            if confidence < self.min_confidence:
                return None
            
            return BreakoutSetup(
                symbol=symbol,
                breakout_type=breakout_type,
                signal=signal,
                confidence=confidence,
                
                breakout_level=nearest_resistance if "Breakout" in signal else nearest_support,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                
                volume_confirmation=vol_confirm,
                volume_ratio=volume_ratio,
                
                breakout_strength=breakout_strength,
                consolidation_days=cons_days,
                range_size=cons_high - cons_low,
                range_percentage=(cons_high - cons_low) / cons_low * 100,
                
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                
                false_breakout_risk=false_breakout_risk,
                
                rsi_confirmation=rsi > 50 if "Breakout" in signal else rsi < 50,
                macd_confirmation=macd_hist > 0 if "Breakout" in signal else macd_hist < 0,
                trend_confirmation=trend_up if "Breakout" in signal else trend_down if "Breakdown" in signal else False,
                
                reasons=reasons
            )
        
        except Exception as e:
            logger.error(f"Error scanning breakout for {symbol}: {e}")
            return None

    def scan_multiple(self, symbols: Dict[str, pd.DataFrame]) -> List[BreakoutSetup]:
        """Scan multiple stocks for breakout setups"""
        setups = []
        for symbol, df in symbols.items():
            setup = self.scan_breakout(df, symbol)
            if setup:
                setups.append(setup)
        
        # Sort by confidence
        setups.sort(key=lambda x: x.confidence, reverse=True)
        return setups

    def format_setup(self, setup: BreakoutSetup) -> str:
        """Format setup for display"""
        return f"""
╔════════════════════════════════════════════════════════╗
║ {setup.symbol:40} │ {setup.signal:10} ║
║ {setup.breakout_type:50} ║
╠════════════════════════════════════════════════════════╣
║ Breakout Level: ${setup.breakout_level:8.2f}   │ Confidence: {setup.confidence:6.1f}%  ║
║ Entry Price:    ${setup.entry_price:8.2f}   │ False Risk: {setup.false_breakout_risk:6.1f}%  ║
║ Stop Loss:      ${setup.stop_loss:8.2f}   │ Strength: {setup.breakout_strength:6.1f}%    ║
║ Take Profit:    ${setup.take_profit:8.2f}   │ R/R Ratio: {setup.risk_reward_ratio:6.2f}x   ║
╠════════════════════════════════════════════════════════╣
║ Consolidation: {setup.consolidation_days:2} days  │  Range: ${setup.range_size:8.2f} ({setup.range_percentage:5.2f}%)║
║ Volume: {setup.volume_ratio:5.2f}x avg   │  Support: ${setup.nearest_support:8.2f}  ║
║ Resistance: ${setup.nearest_resistance:8.2f}  │  Confirmations: RSI={'✓' if setup.rsi_confirmation else '✗'} MACD={'✓' if setup.macd_confirmation else '✗'} Trend={'✓' if setup.trend_confirmation else '✗'} ║
╠════════════════════════════════════════════════════════╣
║ ANALYSIS:                                              ║
"""  + "".join([f"║   • {reason:51}║\n" for reason in setup.reasons[:6]]) + f"""╚════════════════════════════════════════════════════════╝
"""
