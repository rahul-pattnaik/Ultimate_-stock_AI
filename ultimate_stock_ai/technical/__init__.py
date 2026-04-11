# technical/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# Technical Analysis Module Exports
# ─────────────────────────────────────────────────────────────────────────────

from .trend_detection import detect_trend, trend_analysis
from .breakout import breakout_signal, breakout_analysis
from .support_resistance import get_support_resistance, get_sr_zones
from .volume_profile import volume_profile, volume_profile_full
from .momentum import momentum_report, add_momentum_indicators
from .moving_averages import ma_signal, add_moving_averages
from .supertrend import supertrend_signal
from .volatility import volatility_report
from .vwap import vwap_bands
from .fibonacci import fibonacci_retracements
from .ichimoku import ichimoku_signal
from .elliott_wave import detect_elliott_waves

# New modules (Professional Trading Features)
from .rsi_divergence import detect_rsi_divergence, rsi_divergence_report
from .candlestick import detect_candlestick_patterns, candlestick_report
from .volume_analysis import volume_analysis, volume_report, compute_obv
from .confluence import ConfluenceEngine, confluence_report
from .multiple_timeframe import multi_timeframe_analysis, mtf_report
from .sector_rotation import sector_rotation_analysis, sector_report, get_sector_for_stock

__all__ = [
    # Original
    'detect_trend', 'trend_analysis',
    'breakout_signal', 'breakout_analysis',
    'get_support_resistance', 'get_sr_zones',
    'volume_profile', 'volume_profile_full',
    'momentum_report', 'add_momentum_indicators',
    'ma_signal', 'add_moving_averages',
    'supertrend_signal',
    'volatility_report',
    'vwap_bands',
    'fibonacci_retracements',
    'ichimoku_signal',
    'detect_elliott_waves',
    # New
    'detect_rsi_divergence', 'rsi_divergence_report',
    'detect_candlestick_patterns', 'candlestick_report',
    'volume_analysis', 'volume_report', 'compute_obv',
    'ConfluenceEngine', 'confluence_report',
    'multi_timeframe_analysis', 'mtf_report',
    'sector_rotation_analysis', 'sector_report', 'get_sector_for_stock',
]
