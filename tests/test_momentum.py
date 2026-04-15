from __future__ import annotations

import math

import pandas as pd

from technical.momentum import _rsi, add_momentum_indicators, momentum_report


def test_add_momentum_indicators_adds_expected_columns(bullish_df: pd.DataFrame) -> None:
    result = add_momentum_indicators(bullish_df)
    for column in ["rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "cci", "williams_r", "roc10", "mfi"]:
        assert column in result.columns


def test_roc10_matches_expected_percentage(bullish_df: pd.DataFrame) -> None:
    result = add_momentum_indicators(bullish_df)
    expected = bullish_df["Close"].pct_change(10).iloc[-1] * 100
    assert math.isclose(float(result["roc10"].iloc[-1]), float(expected), rel_tol=1e-9)


def test_rsi_on_falling_series_is_low() -> None:
    close = pd.Series([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5], dtype=float)
    rsi = _rsi(close, period=14).iloc[-1]
    assert rsi < 10


def test_momentum_report_handles_insufficient_data(short_df: pd.DataFrame) -> None:
    assert "error" in momentum_report(short_df)


def test_momentum_report_returns_indicator_bundle(bullish_df: pd.DataFrame) -> None:
    result = momentum_report(bullish_df)
    assert "signal" in result
    assert "indicators" in result
