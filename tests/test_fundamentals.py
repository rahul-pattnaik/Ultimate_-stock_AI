from __future__ import annotations

import pandas as pd

from fundamentals.analysis_engine import build_fundamental_report
from portfolio.premium_optimizer import portfolio_health_report, sip_simulation


def test_build_fundamental_report_contains_core_metrics() -> None:
    snapshot = {
        "current_price": 150,
        "eps": 10,
        "bvps": 40,
        "sales_ps": 60,
        "industry_pe": 18,
        "industry_pb": 2.5,
        "industry_ps": 2.0,
        "pe": 15,
        "pb": 2.1,
        "roe": 18,
        "debt_to_equity": 0.4,
        "revenue_growth": 12,
        "earnings_growth": 15,
        "operating_income": 1200,
        "net_income": 900,
        "assets": 8000,
        "liabilities": 3000,
    }
    report = build_fundamental_report(snapshot)
    assert report["fair_value"] > 0
    assert report["debt_health"] == "Healthy"
    assert 0 <= report["fundamental_score"] <= 100


def test_build_fundamental_report_uses_peer_context() -> None:
    snapshot = {
        "current_price": 100,
        "eps": 5,
        "bvps": 20,
        "sales_ps": 30,
        "industry_pe": 20,
        "industry_pb": 3,
        "industry_ps": 2,
        "pe": 14,
        "pb": 1.8,
        "roe": 16,
        "debt_to_equity": 0.5,
        "revenue_growth": 10,
        "earnings_growth": 9,
        "operating_income": 700,
        "net_income": 500,
        "assets": 5000,
        "liabilities": 1800,
    }
    peers = [{"pe": 18, "pb": 2.4}, {"pe": 19, "pb": 2.2}]
    report = build_fundamental_report(snapshot, peers=peers)
    assert report["sector_comparison"] in {"Attractive", "Premium"}


def test_portfolio_health_report_returns_diversification(bullish_df: pd.DataFrame, bearish_df: pd.DataFrame) -> None:
    result = portfolio_health_report(
        {"INFY.NS": bullish_df, "HDFCBANK.NS": bearish_df},
        {"INFY.NS": 0.5, "HDFCBANK.NS": 0.5},
    )
    assert "diversification_score" in result
    assert "sector_exposure" in result


def test_sip_simulation_compounds_over_time() -> None:
    result = sip_simulation(monthly_investment=10000, annual_return_pct=12, years=10)
    assert result["estimated_value"] > result["invested_amount"]
