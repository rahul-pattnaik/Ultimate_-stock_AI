from __future__ import annotations

import pandas as pd

import main_terminal
from ai.ranking_engine import stock_score
from core.reporting import render_ascii_table
from technical.breakout import breakout_analysis, breakout_signal
from technical.trend_detection import detect_trend


def test_no_ambiguous_truth_value_in_breakout_logic(breakout_df: pd.DataFrame) -> None:
    result = breakout_analysis(breakout_df)
    assert isinstance(result["triggered"], bool)
    assert breakout_signal(breakout_df) is True


def test_detect_trend_handles_multiindex_without_sequence_error(multiindex_df: pd.DataFrame) -> None:
    trend = detect_trend(multiindex_df)
    assert isinstance(trend, str)


def test_ranking_engine_regression_score_stays_bounded_on_sideways_data(sideways_df: pd.DataFrame) -> None:
    score = stock_score(sideways_df)
    assert 0 <= score <= 100


def test_cli_formatting_regression_table_shape() -> None:
    table = render_ascii_table("Report", [("Buy Score", "82/100"), ("Risk", "Medium")])
    assert "Report" in table
    assert "| Buy Score | 82/100 |" in table


def test_resolve_symbol_lowercase_regression(monkeypatch, bullish_df: pd.DataFrame) -> None:
    monkeypatch.setattr(main_terminal, "_quiet_yfinance_download", lambda ticker: bullish_df if ticker == "ITC.NS" else pd.DataFrame())
    assert main_terminal.resolve_symbol("itc") == "ITC.NS"
