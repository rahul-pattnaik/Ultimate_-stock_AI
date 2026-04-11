"""
Advanced Momentum Scanner
Features: Multi-indicator momentum, divergence detection, overbought/oversold analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MomentumRating(Enum):
    """Momentum rating classification"""
    VERY_STRONG = 5  # 5 stars
    STRONG = 4       # 4 stars
    MODERATE = 3     # 3 stars
    WEAK = 2         # 2 stars
    VERY_WEAK = 1    # 1 star


@dataclass
class MomentumAnalysis:
    """Comprehensive momentum analysis"""
    symbol: str
    rating: int  # 1-5 stars
    signal: str  # 'Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell'
    momentum_score: float  # 0-100
    trend_strength: float  # 0-100
    
    # Indicator readings
    rsi: float
    rsi_signal: str
    macd: float
    macd_signal_line: float
    macd_histogram: float
    macd_momentum: str
    stochastic_k: float
    stochastic_d: float
    stoch_momentum: str
    
    # Price action
    sma_20: float
    sma_50: float
    sma_200: float
    trend_direction: str  # 'Uptrend', 'Downtrend', 'Sideways'
    price_position: float  # 0-100 (where price is between 20 and 50 MA)
    
    # Volume analysis
    obv_momentum: str
    volume_trend: str
    
    # Divergences
    bullish_divergence: bool
    bearish_divergence: bool
    
    # Confirmations
    multi_indicator_agreement: float  # %
    
    reasons: List[str]


class MomentumScanner:
    """Advanced momentum analysis"""

    def __init__(self):
        self.min_confidence = 55.0

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

    def calculate_macd(self, prices: np.ndarray, fast: int = 12,
                      slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray,
                            close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        lowest_low = pd.Series(low).rolling(window=period).min().values
        highest_high = pd.Series(high).rolling(window=period).max().values
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d_percent = pd.Series(k_percent).rolling(window=3).mean().values
        
        return k_percent, d_percent

    def calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv

    def detect_rsi_divergence(self, close: np.ndarray, rsi: np.ndarray,
                             lookback: int = 20) -> Tuple[bool, bool]:
        """Detect RSI bullish/bearish divergences"""
        bullish = False
        bearish = False
        
        if len(close) < lookback:
            return bullish, bearish
        
        recent_close = close[-lookback:]
        recent_rsi = rsi[-lookback:]
        
        # Find local price lows and RSI lows (bullish divergence)
        price_low_idx = np.argmin(recent_close)
        rsi_low_idx = np.argmin(recent_rsi)
        
        if price_low_idx < rsi_low_idx:
            # Price made lower low but RSI didn't - bullish divergence
            if recent_close[price_low_idx] < np.min(close[:-lookback-1:-1][:5]):
                if recent_rsi[rsi_low_idx] > recent_rsi[price_low_idx]:
                    bullish = True
        
        # Find local price highs and RSI highs (bearish divergence)
        price_high_idx = np.argmax(recent_close)
        rsi_high_idx = np.argmax(recent_rsi)
        
        if price_high_idx < rsi_high_idx:
            # Price made higher high but RSI didn't - bearish divergence
            if recent_close[price_high_idx] > np.max(close[:-lookback-1:-1][:5]):
                if recent_rsi[rsi_high_idx] < recent_rsi[price_high_idx]:
                    bearish = True
        
        return bullish, bearish

    def detect_macd_divergence(self, close: np.ndarray, macd: np.ndarray,
                              lookback: int = 20) -> Tuple[bool, bool]:
        """Detect MACD bullish/bearish divergences"""
        bullish = False
        bearish = False
        
        if len(close) < lookback:
            return bullish, bearish
        
        recent_close = close[-lookback:]
        recent_macd = macd[-lookback:]
        
        # Find local lows
        price_low_idx = np.argmin(recent_close)
        macd_low_idx = np.argmin(recent_macd)
        
        if price_low_idx < macd_low_idx and price_low_idx > 5:
            if recent_close[price_low_idx] < np.min(close[:-lookback-1:-1][:5]):
                if recent_macd[macd_low_idx] > recent_macd[price_low_idx]:
                    bullish = True
        
        # Find local highs
        price_high_idx = np.argmax(recent_close)
        macd_high_idx = np.argmax(recent_macd)
        
        if price_high_idx < macd_high_idx and price_high_idx > 5:
            if recent_close[price_high_idx] > np.max(close[:-lookback-1:-1][:5]):
                if recent_macd[macd_high_idx] < recent_macd[price_high_idx]:
                    bearish = True
        
        return bullish, bearish

    def analyze_momentum(self, df: pd.DataFrame, symbol: str) -> Optional[MomentumAnalysis]:
        """Comprehensive momentum analysis"""
        
        try:
            if len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calculate indicators
            rsi = self.calculate_rsi(close)
            macd, macd_signal, macd_hist = self.calculate_macd(close)
            stoch_k, stoch_d = self.calculate_stochastic(high, low, close)
            obv = self.calculate_obv(close, volume)
            
            # Moving averages
            sma20 = pd.Series(close).rolling(20).mean().values[-1]
            sma50 = pd.Series(close).rolling(50).mean().values[-1]
            sma200 = pd.Series(close).rolling(200).mean().values[-1]
            
            # Current values
            current_price = close[-1]
            current_rsi = rsi[-1]
            current_macd = macd[-1]
            current_macd_signal = macd_signal[-1]
            current_macd_hist = macd_hist[-1]
            current_stoch_k = stoch_k[-1]
            current_stoch_d = stoch_d[-1]
            current_obv = obv[-1]
            prev_obv = obv[-2]
            
            # Divergence detection
            rsi_bull_div, rsi_bear_div = self.detect_rsi_divergence(close, rsi)
            macd_bull_div, macd_bear_div = self.detect_macd_divergence(close, macd)
            
            # ========== SIGNAL ANALYSIS ==========
            
            reasons = []
            buy_signals = 0
            sell_signals = 0
            total_indicators = 0
            
            # RSI analysis
            if current_rsi < 30:
                buy_signals += 1
                reasons.append("RSI oversold (<30)")
            elif current_rsi < 40:
                buy_signals += 0.5
                reasons.append("RSI undervalued (30-40)")
            total_indicators += 1
            
            if current_rsi > 70:
                sell_signals += 1
                reasons.append("RSI overbought (>70)")
            elif current_rsi > 60:
                sell_signals += 0.5
                reasons.append("RSI elevated (60-70)")
            total_indicators += 1
            
            rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            
            # MACD analysis
            if current_macd > current_macd_signal and macd_hist[-2] < 0:
                buy_signals += 1
                reasons.append("MACD bullish crossover")
            elif current_macd > current_macd_signal and current_macd_hist > 0:
                buy_signals += 0.5
                reasons.append("MACD above signal line")
            total_indicators += 1
            
            if current_macd < current_macd_signal and macd_hist[-2] > 0:
                sell_signals += 1
                reasons.append("MACD bearish crossover")
            elif current_macd < current_macd_signal and current_macd_hist < 0:
                sell_signals += 0.5
                reasons.append("MACD below signal line")
            total_indicators += 1
            
            macd_momentum = "Bullish" if current_macd_hist > 0 else "Bearish"
            
            # Stochastic analysis
            if current_stoch_k < 20:
                buy_signals += 1
                reasons.append("Stochastic oversold (<20)")
            elif current_stoch_k < 40:
                buy_signals += 0.5
                reasons.append("Stochastic undervalued (20-40)")
            total_indicators += 1
            
            if current_stoch_k > 80:
                sell_signals += 1
                reasons.append("Stochastic overbought (>80)")
            elif current_stoch_k > 60:
                sell_signals += 0.5
                reasons.append("Stochastic elevated (60-80)")
            total_indicators += 1
            
            stoch_momentum = "Overbought" if current_stoch_k > 80 else "Oversold" if current_stoch_k < 20 else "Neutral"
            
            # Trend analysis
            if current_price > sma50 > sma200:
                buy_signals += 1.5
                reasons.append("Strong uptrend (Price > 50MA > 200MA)")
                trend_direction = "Uptrend"
            elif current_price > sma200:
                buy_signals += 0.5
                reasons.append("Price above 200MA")
                trend_direction = "Uptrend"
            else:
                trend_direction = "Downtrend"
            
            if current_price < sma50 < sma200:
                sell_signals += 1.5
                reasons.append("Strong downtrend (Price < 50MA < 200MA)")
                trend_direction = "Downtrend"
            elif current_price < sma200:
                sell_signals += 0.5
                reasons.append("Price below 200MA")
                trend_direction = "Downtrend"
            
            total_indicators += 2  # Trend MA's
            
            # Price position between 20 and 50 MA
            if sma50 > sma20:
                price_position = min((current_price - sma20) / (sma50 - sma20) * 100, 100)
            else:
                price_position = 50.0
            
            # Volume analysis
            avg_volume = pd.Series(volume).rolling(20).mean().values[-1]
            volume_ratio = volume[-1] / (avg_volume + 1e-10)
            
            if volume_ratio > 1.3:
                buy_signals += 0.5
                reasons.append("Volume confirmation")
                volume_trend = "Increasing"
            elif volume_ratio < 0.7:
                volume_trend = "Decreasing"
            else:
                volume_trend = "Average"
            
            obv_momentum = "Bullish" if current_obv > prev_obv else "Bearish"
            
            # Divergence signals
            if rsi_bull_div:
                buy_signals += 2
                reasons.append("RSI bullish divergence")
            if rsi_bear_div:
                sell_signals += 2
                reasons.append("RSI bearish divergence")
            
            if macd_bull_div:
                buy_signals += 2
                reasons.append("MACD bullish divergence")
            if macd_bear_div:
                sell_signals += 2
                reasons.append("MACD bearish divergence")
            
            # ========== FINAL SCORING ==========
            
            multi_agreement = (buy_signals + sell_signals) / (total_indicators * 2) * 100
            
            # Net momentum
            net_signals = buy_signals - sell_signals
            momentum_score = (net_signals / (total_indicators * 2)) * 100
            momentum_score = np.clip(momentum_score + 50, 0, 100)  # Convert to 0-100
            
            # Determine rating and signal
            if net_signals > total_indicators * 1.5:
                rating = 5
                signal = "Strong Buy"
            elif net_signals > total_indicators * 0.5:
                rating = 4
                signal = "Buy"
            elif net_signals > -total_indicators * 0.5:
                rating = 3
                signal = "Neutral"
            elif net_signals > -total_indicators * 1.5:
                rating = 2
                signal = "Sell"
            else:
                rating = 1
                signal = "Strong Sell"
            
            # Trend strength
            trend_strength = abs(momentum_score - 50) * 2
            trend_strength = np.clip(trend_strength, 0, 100)
            
            # Confidence filter
            if rating == 3 and len(reasons) < 3:
                return None
            
            return MomentumAnalysis(
                symbol=symbol,
                rating=rating,
                signal=signal,
                momentum_score=momentum_score,
                trend_strength=trend_strength,
                
                rsi=current_rsi,
                rsi_signal=rsi_signal,
                macd=current_macd,
                macd_signal_line=current_macd_signal,
                macd_histogram=current_macd_hist,
                macd_momentum=macd_momentum,
                stochastic_k=current_stoch_k,
                stochastic_d=current_stoch_d,
                stoch_momentum=stoch_momentum,
                
                sma_20=sma20,
                sma_50=sma50,
                sma_200=sma200,
                trend_direction=trend_direction,
                price_position=price_position,
                
                obv_momentum=obv_momentum,
                volume_trend=volume_trend,
                
                bullish_divergence=rsi_bull_div or macd_bull_div,
                bearish_divergence=rsi_bear_div or macd_bear_div,
                
                multi_indicator_agreement=multi_agreement,
                
                reasons=reasons
            )
        
        except Exception as e:
            logger.error(f"Error analyzing momentum for {symbol}: {e}")
            return None

    def scan_stocks(self, symbols: Dict[str, pd.DataFrame]) -> List[MomentumAnalysis]:
        """Scan multiple stocks for momentum"""
        analyses = []
        for symbol, df in symbols.items():
            analysis = self.analyze_momentum(df, symbol)
            if analysis and analysis.rating >= 4:  # Only strong signals
                analyses.append(analysis)
        
        # Sort by rating then momentum score
        analyses.sort(key=lambda x: (x.rating, x.momentum_score), reverse=True)
        return analyses

    def format_analysis(self, analysis: MomentumAnalysis) -> str:
        """Format analysis for display"""
        stars = "⭐" * analysis.rating + "☆" * (5 - analysis.rating)
        return f"""
╔════════════════════════════════════════════════════════╗
║ {analysis.symbol:35} {stars:25} ║
║ Signal: {analysis.signal:43} ║
╠════════════════════════════════════════════════════════╣
║ MOMENTUM SCORE: {analysis.momentum_score:6.1f}/100  │  TREND STRENGTH: {analysis.trend_strength:6.1f}/100 ║
║ MULTI-INDICATOR: {analysis.multi_indicator_agreement:5.1f}%  │  TREND: {analysis.trend_direction:30} ║
╠════════════════════════════════════════════════════════╣
║ RSI: {analysis.rsi:6.1f} ({analysis.rsi_signal:12}) │ STOCH K: {analysis.stochastic_k:6.1f}              ║
║ MACD: {'Bullish' if analysis.macd_momentum == 'Bullish' else 'Bearish':15} │ OBV: {'↑ Bullish' if analysis.obv_momentum == 'Bullish' else '↓ Bearish':18} ║
║ 20MA: ${analysis.sma_20:8.2f}  │  50MA: ${analysis.sma_50:8.2f}  │  200MA: ${analysis.sma_200:8.2f} ║
╠════════════════════════════════════════════════════════╣
║ DIVERGENCES: {'✓ Bullish' if analysis.bullish_divergence else '✗'}  {'✓ Bearish' if analysis.bearish_divergence else '✗'} ║
╠════════════════════════════════════════════════════════╣
║ KEY SIGNALS:                                           ║
"""  + "".join([f"║   • {reason:51}║\n" for reason in analysis.reasons[:6]]) + f"""╚════════════════════════════════════════════════════════╝
"""
