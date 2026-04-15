from __future__ import annotations

import pandas as pd

from technical.trend_detection import detect_trend, trend_analysis


def test_detect_trend_identifies_bullish_trend(bullish_df: pd.DataFrame) -> None:
    assert "Uptrend" in detect_trend(bullish_df)


def test_detect_trend_identifies_bearish_trend(bearish_df: pd.DataFrame) -> None:
    assert "Downtrend" in detect_trend(bearish_df)


def test_trend_analysis_handles_multiindex_dataframe(multiindex_df: pd.DataFrame) -> None:
    report = trend_analysis(multiindex_df)
    assert "trend" in report
    assert "timeframes" in report


def test_trend_analysis_handles_short_input(short_df: pd.DataFrame) -> None:
    assert "error" in trend_analysis(short_df)
