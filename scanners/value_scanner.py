"""
Advanced Value Investing Scanner
Features: Multi-metric analysis, intrinsic value calculation, margin of safety
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValueGrade(Enum):
    """Value grade classification"""
    DEEP_VALUE = "A+"
    STRONG_VALUE = "A"
    MODERATE_VALUE = "B+"
    FAIR_VALUE = "B"
    SLIGHTLY_OVERVALUED = "C"
    OVERVALUED = "D"


@dataclass
class ValueAnalysis:
    """Comprehensive value analysis result"""
    symbol: str
    grade: str
    score: float  # 0-100
    intrinsic_value: float
    current_price: float
    margin_of_safety: float  # %
    confidence: float  # 0-100
    
    # Valuation metrics
    pe_ratio: float
    pb_ratio: float
    ps_ratio: float
    ev_ebitda: float
    fcf_yield: float
    
    # Quality metrics
    roe: float
    debt_to_equity: float
    current_ratio: float
    fcf_to_net_income: float
    
    # Growth metrics
    revenue_growth: float
    earnings_growth: float
    fcf_growth: float
    
    # Scores
    valuation_score: float  # 0-100
    quality_score: float    # 0-100
    growth_score: float     # 0-100
    
    reasons: List[str]


class ValueScanner:
    """Advanced value investing analysis"""

    def __init__(self):
        self.min_confidence = 60.0

    def calculate_intrinsic_value_dcf(self, fcf: float, growth_rate: float,
                                      wacc: float = 0.08, 
                                      terminal_growth: float = 0.03,
                                      periods: int = 5) -> float:
        """Calculate intrinsic value using DCF"""
        if fcf <= 0 or wacc <= terminal_growth:
            return 0.0
        
        pv_fcf = 0.0
        for i in range(1, periods + 1):
            fcf_year = fcf * ((1 + growth_rate) ** i)
            pv_fcf += fcf_year / ((1 + wacc) ** i)
        
        terminal_fcf = fcf * ((1 + growth_rate) ** periods) * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** periods)
        
        enterprise_value = pv_fcf + pv_terminal
        return enterprise_value

    def calculate_pe_percentile(self, pe: float, industry_pe: float,
                               market_pe: float = 20) -> float:
        """Calculate PE ratio percentile (0-100, higher is cheaper)"""
        if pe <= 0:
            return 0.0
        
        # Compare to industry average
        ratio = pe / (industry_pe + 1e-10)
        
        # Normalize to 0-100 scale
        if ratio < 0.5:
            percentile = 90.0  # Very cheap
        elif ratio < 0.7:
            percentile = 75.0  # Cheap
        elif ratio < 0.9:
            percentile = 60.0  # Fair
        elif ratio < 1.1:
            percentile = 40.0  # Slightly expensive
        else:
            percentile = 20.0  # Expensive
        
        return percentile

    def calculate_pb_percentile(self, pb: float, industry_pb: float) -> float:
        """Calculate P/B ratio percentile"""
        if pb <= 0:
            return 0.0
        
        ratio = pb / (industry_pb + 1e-10)
        
        if ratio < 0.6:
            percentile = 90.0
        elif ratio < 0.8:
            percentile = 75.0
        elif ratio < 1.0:
            percentile = 60.0
        elif ratio < 1.2:
            percentile = 40.0
        else:
            percentile = 20.0
        
        return percentile

    def calculate_ps_percentile(self, ps: float, industry_ps: float) -> float:
        """Calculate P/S ratio percentile"""
        if ps <= 0:
            return 0.0
        
        ratio = ps / (industry_ps + 1e-10)
        
        if ratio < 0.6:
            percentile = 90.0
        elif ratio < 0.8:
            percentile = 75.0
        elif ratio < 1.0:
            percentile = 60.0
        elif ratio < 1.3:
            percentile = 40.0
        else:
            percentile = 20.0
        
        return percentile

    def quality_score_calculation(self, roe: float, debt_to_equity: float,
                                 current_ratio: float,
                                 fcf_to_ni: float) -> float:
        """Calculate quality score (0-100)"""
        score = 0.0
        max_score = 0.0
        
        # ROE contribution (25 points max)
        roe_score = min(roe * 2.5, 25.0) if roe > 0 else 0.0
        score += roe_score
        max_score += 25
        
        # Debt contribution (25 points max)
        de_score = 0.0
        if debt_to_equity < 0.5:
            de_score = 25.0
        elif debt_to_equity < 1.0:
            de_score = 20.0
        elif debt_to_equity < 1.5:
            de_score = 15.0
        elif debt_to_equity < 2.0:
            de_score = 10.0
        score += de_score
        max_score += 25
        
        # Liquidity contribution (25 points max)
        cr_score = 0.0
        if current_ratio > 2.0:
            cr_score = 25.0
        elif current_ratio > 1.5:
            cr_score = 20.0
        elif current_ratio > 1.0:
            cr_score = 15.0
        elif current_ratio > 0.5:
            cr_score = 5.0
        score += cr_score
        max_score += 25
        
        # FCF quality contribution (25 points max)
        fcf_score = 0.0
        if fcf_to_ni > 1.0:
            fcf_score = 25.0
        elif fcf_to_ni > 0.8:
            fcf_score = 20.0
        elif fcf_to_ni > 0.6:
            fcf_score = 15.0
        elif fcf_to_ni > 0.4:
            fcf_score = 10.0
        score += fcf_score
        max_score += 25
        
        return (score / max_score * 100) if max_score > 0 else 0.0

    def growth_score_calculation(self, revenue_growth: float,
                                earnings_growth: float,
                                fcf_growth: float) -> float:
        """Calculate growth score (0-100)"""
        score = 0.0
        max_score = 0.0
        
        # Revenue growth (35 points)
        rev_score = min(revenue_growth * 3.5, 35.0) if revenue_growth > 0 else 0.0
        score += rev_score
        max_score += 35
        
        # Earnings growth (35 points)
        earn_score = min(earnings_growth * 3.5, 35.0) if earnings_growth > 0 else 0.0
        score += earn_score
        max_score += 35
        
        # FCF growth (30 points)
        fcf_score = min(fcf_growth * 3.0, 30.0) if fcf_growth > 0 else 0.0
        score += fcf_score
        max_score += 30
        
        return (score / max_score * 100) if max_score > 0 else 0.0

    def calculate_valuation_score(self, pe_pct: float, pb_pct: float,
                                 ps_pct: float, ev_ebitda_pct: float) -> float:
        """Calculate composite valuation score"""
        scores = [pe_pct, pb_pct, ps_pct, ev_ebitda_pct]
        valid_scores = [s for s in scores if s > 0]
        return np.mean(valid_scores) if valid_scores else 0.0

    def determine_grade(self, overall_score: float) -> str:
        """Determine value grade"""
        if overall_score >= 90:
            return ValueGrade.DEEP_VALUE.value
        elif overall_score >= 80:
            return ValueGrade.STRONG_VALUE.value
        elif overall_score >= 70:
            return ValueGrade.MODERATE_VALUE.value
        elif overall_score >= 60:
            return ValueGrade.FAIR_VALUE.value
        elif overall_score >= 40:
            return ValueGrade.SLIGHTLY_OVERVALUED.value
        else:
            return ValueGrade.OVERVALUED.value

    def scan_value_stock(self, symbol: str,
                        current_price: float,
                        pe: float,
                        pb: float,
                        ps: float,
                        ev_ebitda: float,
                        market_cap: float,
                        fcf: float,
                        
                        # Quality metrics
                        roe: float,
                        debt_to_equity: float,
                        current_ratio: float,
                        
                        # Growth metrics
                        revenue_growth: float,
                        earnings_growth: float,
                        fcf_growth: float,
                        
                        # Industry benchmarks
                        industry_pe: float = 20,
                        industry_pb: float = 3,
                        industry_ps: float = 2,
                        fcf_yield_pct: float = 5) -> Optional[ValueAnalysis]:
        """Comprehensive value analysis"""
        
        try:
            reasons = []
            
            # Calculate FCF yield
            fcf_yield = (fcf / market_cap * 100) if market_cap > 0 else 0.0
            fcf_to_ni = fcf / (current_price * 0.05) if current_price > 0 else 0.0  # Rough estimate
            
            # Valuation metrics percentiles
            pe_pct = self.calculate_pe_percentile(pe, industry_pe)
            pb_pct = self.calculate_pb_percentile(pb, industry_pb)
            ps_pct = self.calculate_ps_percentile(ps, industry_ps)
            
            # EV/EBITDA percentile
            if ev_ebitda > 0 and ev_ebitda < 30:
                ev_pct = min(max((30 - ev_ebitda) / 20 * 100, 0), 100)
            else:
                ev_pct = 20.0
            
            # Calculate scores
            valuation_score = self.calculate_valuation_score(pe_pct, pb_pct, ps_pct, ev_pct)
            quality_score = self.quality_score_calculation(roe, debt_to_equity, 
                                                          current_ratio, fcf_to_ni)
            growth_score = self.growth_score_calculation(revenue_growth, 
                                                        earnings_growth, fcf_growth)
            
            # Weighted overall score
            overall_score = (valuation_score * 0.5) + (quality_score * 0.3) + (growth_score * 0.2)
            
            # Calculate intrinsic value estimate
            intrinsic_value = self.calculate_intrinsic_value_dcf(
                fcf=fcf,
                growth_rate=min(revenue_growth, 0.15),  # Cap at 15%
                wacc=0.08,
                terminal_growth=0.03,
                periods=5
            )
            
            # Margin of safety
            if intrinsic_value > 0:
                margin_of_safety = ((intrinsic_value - current_price) / current_price) * 100
            else:
                margin_of_safety = 0.0
            
            # Build reasons
            if pe < industry_pe * 0.7:
                reasons.append("Cheap P/E vs industry")
            if pb < industry_pb * 0.8:
                reasons.append("Discount to book value")
            if ps < industry_ps * 0.8:
                reasons.append("Low P/S ratio")
            if roe > 15:
                reasons.append("Strong ROE (>15%)")
            if debt_to_equity < 1.0:
                reasons.append("Conservative leverage")
            if current_ratio > 1.5:
                reasons.append("Strong liquidity")
            if fcf_yield > fcf_yield_pct:
                reasons.append(f"High FCF yield ({fcf_yield:.1f}%)")
            if margin_of_safety > 20:
                reasons.append(f"Margin of safety: {margin_of_safety:.1f}%")
            if revenue_growth > 5:
                reasons.append(f"Revenue growth: {revenue_growth:.1f}%")
            
            # Determine confidence
            confidence = overall_score
            
            # Only return if has enough reasons or strong score
            if len(reasons) < 3 and confidence < 60:
                return None
            
            grade = self.determine_grade(overall_score)
            
            return ValueAnalysis(
                symbol=symbol,
                grade=grade,
                score=overall_score,
                intrinsic_value=intrinsic_value,
                current_price=current_price,
                margin_of_safety=margin_of_safety,
                confidence=confidence,
                
                pe_ratio=pe,
                pb_ratio=pb,
                ps_ratio=ps,
                ev_ebitda=ev_ebitda,
                fcf_yield=fcf_yield,
                
                roe=roe,
                debt_to_equity=debt_to_equity,
                current_ratio=current_ratio,
                fcf_to_net_income=fcf_to_ni,
                
                revenue_growth=revenue_growth,
                earnings_growth=earnings_growth,
                fcf_growth=fcf_growth,
                
                valuation_score=valuation_score,
                quality_score=quality_score,
                growth_score=growth_score,
                
                reasons=reasons
            )
        
        except Exception as e:
            logger.error(f"Error scanning value for {symbol}: {e}")
            return None

    def scan_portfolio(self, stocks: Dict[str, Dict]) -> List[ValueAnalysis]:
        """Scan multiple stocks for value"""
        analyses = []
        for symbol, data in stocks.items():
            analysis = self.scan_value_stock(symbol, **data)
            if analysis:
                analyses.append(analysis)
        
        # Sort by score
        analyses.sort(key=lambda x: x.score, reverse=True)
        return analyses

    def format_analysis(self, analysis: ValueAnalysis) -> str:
        """Format analysis for display"""
        return f"""
╔════════════════════════════════════════════════════════════╗
║ {analysis.symbol:50} {analysis.grade:>6} ║
╠════════════════════════════════════════════════════════════╣
║ VALUE SCORE:        {analysis.score:6.1f}/100  │  VALUATION:  {analysis.valuation_score:6.1f}  ║
║ QUALITY SCORE:      {analysis.quality_score:6.1f}/100  │  QUALITY:    {analysis.quality_score:6.1f}  ║
║ GROWTH SCORE:       {analysis.growth_score:6.1f}/100  │  GROWTH:     {analysis.growth_score:6.1f}  ║
╠════════════════════════════════════════════════════════════╣
║ INTRINSIC VALUE:    ${analysis.intrinsic_value:10.2f}  │  CURRENT:    ${analysis.current_price:8.2f}║
║ MARGIN OF SAFETY:   {analysis.margin_of_safety:10.2f}% │  CONFIDENCE: {analysis.confidence:6.1f}% ║
╠════════════════════════════════════════════════════════════╣
║ P/E: {analysis.pe_ratio:6.2f}  │  P/B: {analysis.pb_ratio:6.2f}  │  P/S: {analysis.ps_ratio:6.2f}  │  EV/EBITDA: {analysis.ev_ebitda:6.2f}  ║
║ ROE: {analysis.roe:6.1f}% │  D/E: {analysis.debt_to_equity:6.2f}  │  CR: {analysis.current_ratio:6.2f}  │  FCF Yield: {analysis.fcf_yield:5.1f}% ║
║ Rev Growth: {analysis.revenue_growth:5.1f}% │ Earn Growth: {analysis.earnings_growth:5.1f}% │ FCF Growth: {analysis.fcf_growth:5.1f}% ║
╠════════════════════════════════════════════════════════════╣
║ KEY REASONS:                                               ║
"""  + "".join([f"║   • {reason:56}║\n" for reason in analysis.reasons[:8]]) + f"""╚════════════════════════════════════════════════════════════╝
"""
