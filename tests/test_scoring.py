from __future__ import annotations

import pandas as pd

from ai.premium_scoring import generate_premium_scores
from ai.ranking_engine import detailed_score_report, stock_score


def test_stock_score_stays_in_0_to_100_range(bullish_df: pd.DataFrame) -> None:
    score = stock_score(bullish_df)
    assert 0 <= score <= 100


def test_bullish_data_scores_higher_than_bearish_data(bullish_df: pd.DataFrame, bearish_df: pd.DataFrame) -> None:
    assert stock_score(bullish_df) > stock_score(bearish_df)


def test_detailed_score_report_contains_breakdown(bullish_df: pd.DataFrame) -> None:
    report = detailed_score_report(bullish_df)
    assert "breakdown" in report
    assert "indicators" in report


def test_premium_scores_are_bounded() -> None:
    result = generate_premium_scores(
        technical_report={
            "confluence_score": 82,
            "momentum_score": 76,
            "volatility_regime": "Normal Volatility",
            "trend": "Strong Uptrend",
        },
        fundamental_report={
            "fundamental_score": 74,
            "fair_value_upside_pct": 18,
            "debt_health": "Healthy",
        },
        ai_signal_result={"score": 70},
        prediction={"change_percent": 6.5},
    )
    for key in ["buy_score", "swing_score", "long_term_score", "risk_score", "confidence"]:
        assert 0 <= result[key] <= 100
    assert result["action"] in {"BUY", "ACCUMULATE", "HOLD", "AVOID"}
