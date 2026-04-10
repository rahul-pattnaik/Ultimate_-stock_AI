"""
Advanced Valuation Models & ML-Powered Predictive Analytics
Features: DCF, comparables, growth forecasting, anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValuationMethod(Enum):
    """Valuation methodology types"""
    DCF = "discounted_cash_flow"
    COMPARABLE = "comparable_companies"
    ASSET_BASED = "asset_based"
    PRECEDENT = "precedent_transaction"
    SUM_OF_PARTS = "sum_of_parts"


@dataclass
class ValuationResult:
    """Comprehensive valuation output"""
    method: ValuationMethod
    intrinsic_value: float
    current_price: float
    upside_downside: float
    confidence: float  # 0-100
    key_assumptions: Dict[str, float]
    sensitivity_analysis: Dict[str, Dict[str, float]]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def recommendation(self) -> str:
        """Investment recommendation based on valuation"""
        if self.upside_downside > 25:
            return "STRONG BUY"
        elif self.upside_downside > 15:
            return "BUY"
        elif self.upside_downside > -15:
            return "HOLD"
        elif self.upside_downside > -25:
            return "SELL"
        else:
            return "STRONG SELL"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'method': self.method.value,
            'intrinsic_value': round(self.intrinsic_value, 2),
            'current_price': round(self.current_price, 2),
            'upside_downside': round(self.upside_downside, 2),
            'confidence': round(self.confidence, 1),
            'recommendation': self.recommendation,
            'timestamp': self.timestamp.isoformat()
        }


class DCFValuation:
    """Discounted Cash Flow Valuation Model"""

    def __init__(self, wacc: float = 0.08, terminal_growth_rate: float = 0.03):
        self.wacc = wacc
        self.terminal_growth_rate = terminal_growth_rate
        self.projection_years = 5

    def project_fcf(self, base_fcf: float, growth_rates: List[float]) -> np.ndarray:
        """Project Free Cash Flow"""
        fcf = np.zeros(len(growth_rates))
        fcf[0] = base_fcf * (1 + growth_rates[0])

        for i in range(1, len(growth_rates)):
            fcf[i] = fcf[i-1] * (1 + growth_rates[i])

        return fcf

    def calculate_pv_fcf(self, fcf_projections: np.ndarray) -> float:
        """Present Value of FCF"""
        pv = 0
        for i, fcf in enumerate(fcf_projections):
            pv += fcf / ((1 + self.wacc) ** (i + 1))
        return pv

    def calculate_terminal_value(self, final_fcf: float) -> float:
        """Terminal Value (Gordon Growth Model)"""
        return (final_fcf * (1 + self.terminal_growth_rate)) / (self.wacc - self.terminal_growth_rate + 1e-10)

    def calculate_pv_terminal(self, terminal_value: float, periods: int) -> float:
        """Present Value of Terminal Value"""
        return terminal_value / ((1 + self.wacc) ** periods)

    def calculate_enterprise_value(self, base_fcf: float, growth_rates: List[float],
                                   shares_outstanding: float, net_debt: float) -> Tuple[float, Dict]:
        """Calculate Enterprise Value and Equity Value Per Share"""
        
        # Project FCF
        fcf_projections = self.project_fcf(base_fcf, growth_rates)
        pv_fcf = self.calculate_pv_fcf(fcf_projections)

        # Terminal Value
        terminal_value = self.calculate_terminal_value(fcf_projections[-1])
        pv_terminal = self.calculate_pv_terminal(terminal_value, len(growth_rates))

        # Enterprise Value
        enterprise_value = pv_fcf + pv_terminal

        # Equity Value
        equity_value = enterprise_value - net_debt
        value_per_share = equity_value / (shares_outstanding + 1e-10)

        assumptions = {
            'base_fcf': base_fcf,
            'wacc': self.wacc,
            'terminal_growth_rate': self.terminal_growth_rate,
            'shares_outstanding': shares_outstanding,
            'net_debt': net_debt
        }

        return value_per_share, assumptions

    def sensitivity_analysis(self, base_fcf: float, growth_rates: List[float],
                           shares_outstanding: float, net_debt: float,
                           wacc_range: Tuple[float, float],
                           tgr_range: Tuple[float, float]) -> Dict[str, Dict]:
        """Sensitivity Analysis on WACC and Terminal Growth Rate"""
        
        sensitivity = {}
        wacc_values = np.linspace(wacc_range[0], wacc_range[1], 5)
        tgr_values = np.linspace(tgr_range[0], tgr_range[1], 5)

        for wacc in wacc_values:
            for tgr in tgr_values:
                old_wacc = self.wacc
                old_tgr = self.terminal_growth_rate
                self.wacc = wacc
                self.terminal_growth_rate = tgr

                value_per_share, _ = self.calculate_enterprise_value(
                    base_fcf, growth_rates, shares_outstanding, net_debt
                )

                self.wacc = old_wacc
                self.terminal_growth_rate = old_tgr

                key = f"WACC_{wacc:.2%}_TGR_{tgr:.2%}"
                sensitivity[key] = {'value': value_per_share}

        return sensitivity


class ComparableValuation:
    """Comparable Companies Valuation"""

    @staticmethod
    def calculate_average_multiple(multiples: List[float]) -> float:
        """Calculate average of multiples, excluding outliers"""
        if len(multiples) < 2:
            return np.mean(multiples)

        # Remove outliers (values beyond 2 std dev)
        multiples = np.array(multiples)
        mean = np.mean(multiples)
        std = np.std(multiples)
        filtered = multiples[np.abs(multiples - mean) <= 2 * std]

        return float(np.mean(filtered)) if len(filtered) > 0 else float(mean)

    @staticmethod
    def calculate_metrics_from_comparables(
        comparable_pe: List[float],
        comparable_pb: List[float],
        comparable_ps: List[float],
        target_eps: float,
        target_bvps: float,
        target_sales_ps: float
    ) -> Dict[str, float]:
        """Calculate valuation using comparable multiples"""

        avg_pe = ComparableValuation.calculate_average_multiple(comparable_pe)
        avg_pb = ComparableValuation.calculate_average_multiple(comparable_pb)
        avg_ps = ComparableValuation.calculate_average_multiple(comparable_ps)

        valuations = {
            'pe_valuation': target_eps * avg_pe if avg_pe else 0,
            'pb_valuation': target_bvps * avg_pb if avg_pb else 0,
            'ps_valuation': target_sales_ps * avg_ps if avg_ps else 0,
            'avg_pe': avg_pe,
            'avg_pb': avg_pb,
            'avg_ps': avg_ps
        }

        return valuations

    @staticmethod
    def weighted_valuation(valuations: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted average valuation"""
        weighted_sum = 0
        total_weight = 0

        for key, weight in weights.items():
            if key in valuations:
                weighted_sum += valuations[key] * weight
                total_weight += weight

        return weighted_sum / (total_weight + 1e-10)


class GrowthForecasting:
    """AI-powered growth forecasting and trend analysis"""

    @staticmethod
    def linear_regression_forecast(historical_values: List[float], periods: int = 4) -> np.ndarray:
        """Linear regression-based forecast"""
        x = np.arange(len(historical_values))
        y = np.array(historical_values)

        # Fit linear model
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        # Forecast
        future_x = np.arange(len(historical_values), len(historical_values) + periods)
        forecast = p(future_x)

        return np.maximum(forecast, 0)  # Ensure non-negative

    @staticmethod
    def exponential_smoothing(values: List[float], alpha: float = 0.3, periods: int = 4) -> np.ndarray:
        """Exponential smoothing forecast"""
        values = np.array(values)
        result = [values[0]]

        for i in range(1, len(values)):
            result.append(alpha * values[i] + (1 - alpha) * result[i-1])

        # Forecast using last smoothed value
        forecast = np.full(periods, result[-1])
        return forecast

    @staticmethod
    def moving_average_forecast(values: List[float], window: int = 4, periods: int = 4) -> np.ndarray:
        """Moving average-based forecast"""
        ma = np.mean(values[-window:])
        return np.full(periods, ma)

    @staticmethod
    def arima_simple_forecast(values: List[float], periods: int = 4) -> np.ndarray:
        """Simplified ARIMA-like forecast using differencing"""
        values = np.array(values)
        
        if len(values) < 2:
            return np.full(periods, values[-1])

        # Calculate first differences
        diffs = np.diff(values)
        avg_diff = np.mean(diffs[-4:])  # Use last 4 differences

        # Forecast
        last_value = values[-1]
        forecast = np.zeros(periods)
        for i in range(periods):
            forecast[i] = last_value + (avg_diff * (i + 1))

        return np.maximum(forecast, 0)


class AnomalyDetector:
    """Detect anomalies and unusual patterns in stock data"""

    @staticmethod
    def zscore_anomalies(prices: np.ndarray, threshold: float = 3.0) -> List[Tuple[int, float]]:
        """Detect price anomalies using Z-score"""
        returns = np.diff(prices) / prices[:-1]
        mean = np.mean(returns)
        std = np.std(returns)

        anomalies = []
        for i, ret in enumerate(returns):
            zscore = abs((ret - mean) / (std + 1e-10))
            if zscore > threshold:
                anomalies.append((i + 1, zscore))

        return anomalies

    @staticmethod
    def volume_anomalies(volumes: np.ndarray, threshold: float = 2.0) -> List[Tuple[int, float]]:
        """Detect unusual volume spikes"""
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)

        anomalies = []
        for i, vol in enumerate(volumes):
            zscore = (vol - mean_vol) / (std_vol + 1e-10)
            if zscore > threshold:
                anomalies.append((i, zscore))

        return anomalies

    @staticmethod
    def volatility_regimes(returns: np.ndarray, window: int = 20) -> List[str]:
        """Identify volatility regimes (Low, Normal, High)"""
        rolling_std = pd.Series(returns).rolling(window=window).std().values

        q1 = np.percentile(rolling_std[~np.isnan(rolling_std)], 33)
        q3 = np.percentile(rolling_std[~np.isnan(rolling_std)], 67)

        regimes = []
        for std in rolling_std:
            if np.isnan(std):
                regimes.append('NA')
            elif std <= q1:
                regimes.append('Low')
            elif std <= q3:
                regimes.append('Normal')
            else:
                regimes.append('High')

        return regimes


class ValuationEngine:
    """Unified valuation engine combining multiple methods"""

    def __init__(self):
        self.dcf_model = DCFValuation()
        self.comparable_model = ComparableValuation()
        self.forecaster = GrowthForecasting()
        self.anomaly_detector = AnomalyDetector()

    def comprehensive_valuation(self, current_price: float, **kwargs) -> List[ValuationResult]:
        """Generate comprehensive valuation from multiple methods"""
        results = []

        # DCF Valuation
        if 'base_fcf' in kwargs and 'growth_rates' in kwargs:
            try:
                value_per_share, assumptions = self.dcf_model.calculate_enterprise_value(
                    kwargs['base_fcf'],
                    kwargs['growth_rates'],
                    kwargs.get('shares_outstanding', 1),
                    kwargs.get('net_debt', 0)
                )

                dcf_result = ValuationResult(
                    method=ValuationMethod.DCF,
                    intrinsic_value=value_per_share,
                    current_price=current_price,
                    upside_downside=((value_per_share - current_price) / current_price) * 100,
                    confidence=85.0,
                    key_assumptions=assumptions,
                    sensitivity_analysis={}
                )
                results.append(dcf_result)
            except Exception as e:
                logger.error(f"DCF valuation error: {e}")

        # Comparable Valuation
        if 'comparable_multiples' in kwargs:
            multiples = kwargs['comparable_multiples']
            try:
                valuations = self.comparable_model.calculate_metrics_from_comparables(
                    multiples.get('pe', []),
                    multiples.get('pb', []),
                    multiples.get('ps', []),
                    kwargs.get('eps', 1),
                    kwargs.get('bvps', 1),
                    kwargs.get('sales_ps', 1)
                )

                comparable_value = self.comparable_model.weighted_valuation(
                    valuations,
                    {'pe_valuation': 0.4, 'pb_valuation': 0.3, 'ps_valuation': 0.3}
                )

                comp_result = ValuationResult(
                    method=ValuationMethod.COMPARABLE,
                    intrinsic_value=comparable_value,
                    current_price=current_price,
                    upside_downside=((comparable_value - current_price) / current_price) * 100,
                    confidence=75.0,
                    key_assumptions=valuations,
                    sensitivity_analysis={}
                )
                results.append(comp_result)
            except Exception as e:
                logger.error(f"Comparable valuation error: {e}")

        return results
