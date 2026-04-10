"""
Advanced Financial Data Module for AI Stock Analysis
Features: Real-time processing, caching, batch operations, robust error handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import lru_cache
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)


class PeriodType(Enum):
    """Financial period types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class OHLCV:
    """OHLCV data structure with validation"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

    def __post_init__(self):
        """Validate OHLCV integrity"""
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            logger.warning(f"Invalid OHLCV at {self.timestamp}: H={self.high}, L={self.low}, O={self.open}, C={self.close}")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")

    @property
    def body_size(self) -> float:
        """Candlestick body size"""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """High-low range"""
        return self.high - self.low

    @property
    def hl_ratio(self) -> float:
        """Body to range ratio"""
        return self.body_size / (self.range + 1e-10)


@dataclass
class FinancialMetrics:
    """Comprehensive financial metrics"""
    revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    ebitda: float
    assets: float
    liabilities: float
    equity: float
    operating_cash_flow: float
    free_cash_flow: float
    debt: float
    cash: float

    @property
    def gross_margin(self) -> float:
        return (self.gross_profit / (self.revenue + 1e-10)) * 100

    @property
    def operating_margin(self) -> float:
        return (self.operating_income / (self.revenue + 1e-10)) * 100

    @property
    def net_margin(self) -> float:
        return (self.net_income / (self.revenue + 1e-10)) * 100

    @property
    def roa(self) -> float:
        """Return on Assets"""
        return (self.net_income / (self.assets + 1e-10)) * 100

    @property
    def roe(self) -> float:
        """Return on Equity"""
        return (self.net_income / (self.equity + 1e-10)) * 100

    @property
    def debt_to_equity(self) -> float:
        return self.debt / (self.equity + 1e-10)

    @property
    def current_ratio(self) -> float:
        """Simplified - should use current assets/liabilities"""
        return self.assets / (self.liabilities + 1e-10)

    @property
    def fcf_margin(self) -> float:
        return (self.free_cash_flow / (self.revenue + 1e-10)) * 100


class FinancialDataHandler:
    """Optimized financial data handler with caching and batch operations"""

    def __init__(self, max_cache_size: int = 1000):
        self.ohlcv_data: Dict[str, List[OHLCV]] = {}
        self.financial_data: Dict[str, List[FinancialMetrics]] = {}
        self.max_cache_size = max_cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def add_ohlcv(self, symbol: str, ohlcv_list: List[OHLCV]) -> None:
        """Add OHLCV data with validation"""
        try:
            if symbol in self.ohlcv_data:
                existing = self.ohlcv_data[symbol]
                # Merge and remove duplicates based on timestamp
                merged = {candle.timestamp: candle for candle in existing + ohlcv_list}
                self.ohlcv_data[symbol] = sorted(merged.values(), key=lambda x: x.timestamp)
            else:
                self.ohlcv_data[symbol] = sorted(ohlcv_list, key=lambda x: x.timestamp)
            logger.info(f"Added {len(ohlcv_list)} OHLCV records for {symbol}")
        except Exception as e:
            logger.error(f"Error adding OHLCV for {symbol}: {e}")
            raise

    def add_financial_metrics(self, symbol: str, metrics_list: List[FinancialMetrics]) -> None:
        """Add financial metrics"""
        self.financial_data[symbol] = sorted(
            metrics_list,
            key=lambda x: getattr(x, '_date', datetime.now()),
            reverse=True
        )
        logger.info(f"Added {len(metrics_list)} financial records for {symbol}")

    def get_ohlcv_dataframe(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Convert OHLCV to optimized DataFrame"""
        if symbol not in self.ohlcv_data:
            return pd.DataFrame()

        data = self.ohlcv_data[symbol][-limit:] if limit else self.ohlcv_data[symbol]
        
        df = pd.DataFrame({
            'timestamp': [c.timestamp for c in data],
            'open': [c.open for c in data],
            'high': [c.high for c in data],
            'low': [c.low for c in data],
            'close': [c.close for c in data],
            'volume': [c.volume for c in data]
        })
        
        df.set_index('timestamp', inplace=True)
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest closing price"""
        if symbol not in self.ohlcv_data or not self.ohlcv_data[symbol]:
            return None
        return self.ohlcv_data[symbol][-1].close

    def get_price_range(self, symbol: str, days: int = 252) -> Tuple[float, float]:
        """Get 52-week or custom range"""
        if symbol not in self.ohlcv_data:
            return 0, 0

        recent = self.ohlcv_data[symbol][-days:]
        if not recent:
            return 0, 0

        lows = [c.low for c in recent]
        highs = [c.high for c in recent]
        return min(lows), max(highs)

    @lru_cache(maxsize=128)
    def calculate_daily_returns(self, symbol: str) -> np.ndarray:
        """Calculate daily returns with caching"""
        df = self.get_ohlcv_dataframe(symbol)
        if len(df) < 2:
            return np.array([])
        return df['close'].pct_change().dropna().values

    def get_volatility(self, symbol: str, periods: int = 252) -> float:
        """Annualized volatility"""
        returns = self.calculate_daily_returns(symbol)
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(periods) * 100

    def batch_get_metrics(self, symbols: List[str]) -> Dict[str, Dict]:
        """Efficiently get metrics for multiple symbols"""
        results = {}
        for symbol in symbols:
            try:
                latest = self.get_latest_price(symbol)
                low_52w, high_52w = self.get_price_range(symbol)
                volatility = self.get_volatility(symbol)
                
                results[symbol] = {
                    'latest_price': latest,
                    '52w_low': low_52w,
                    '52w_high': high_52w,
                    'volatility': volatility
                }
            except Exception as e:
                logger.error(f"Error getting metrics for {symbol}: {e}")
                results[symbol] = None

        return results

    def get_price_momentum(self, symbol: str, periods: List[int] = [20, 50, 200]) -> Dict[str, float]:
        """Calculate momentum across multiple periods"""
        latest = self.get_latest_price(symbol)
        if not latest:
            return {}

        df = self.get_ohlcv_dataframe(symbol)
        if len(df) == 0:
            return {}

        momentum = {}
        for period in periods:
            if len(df) >= period:
                past_price = df['close'].iloc[-period]
                change_pct = ((latest - past_price) / past_price) * 100
                momentum[f'{period}d'] = change_pct
            else:
                momentum[f'{period}d'] = 0.0

        return momentum

    def detect_gaps(self, symbol: str) -> List[Dict]:
        """Detect price gaps for trading signals"""
        if symbol not in self.ohlcv_data or len(self.ohlcv_data[symbol]) < 2:
            return []

        gaps = []
        candles = self.ohlcv_data[symbol]

        for i in range(1, len(candles)):
            prev_close = candles[i-1].close
            curr_open = candles[i].open
            gap_pct = ((curr_open - prev_close) / prev_close) * 100

            if abs(gap_pct) > 1:  # Only significant gaps
                gaps.append({
                    'timestamp': candles[i].timestamp,
                    'gap_pct': gap_pct,
                    'direction': 'up' if gap_pct > 0 else 'down'
                })

        return gaps[-10:]  # Return last 10 gaps

    def to_json(self, symbol: str) -> str:
        """Export data to JSON"""
        df = self.get_ohlcv_dataframe(symbol)
        if df.empty:
            return "{}"
        return df.to_json(orient='records', default_handler=str)

    def get_stats(self) -> Dict:
        """Get handler statistics"""
        total_symbols = len(self.ohlcv_data)
        total_candles = sum(len(v) for v in self.ohlcv_data.values())
        return {
            'total_symbols': total_symbols,
            'total_candles': total_candles,
            'cache_size': len(self.ohlcv_data)
        }
