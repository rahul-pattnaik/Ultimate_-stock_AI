from __future__ import annotations

import math

from technical.moving_averages import add_moving_averages, ma_signal


def test_add_moving_averages_adds_expected_columns(bullish_df) -> None:
    result = add_moving_averages(bullish_df)
    for column in ["ma20", "ma50", "ma200", "ema9", "wma20", "dema20", "hma20", "golden_cross", "death_cross", "ma_trend"]:
        assert column in result.columns


def test_sma20_matches_expected_mean(bullish_df) -> None:
    result = add_moving_averages(bullish_df)
    expected = bullish_df["Close"].tail(20).mean()
    assert math.isclose(float(result["ma20"].iloc[-1]), float(expected), rel_tol=1e-9)


def test_ma_signal_returns_actionable_snapshot(bullish_df) -> None:
    result = ma_signal(bullish_df)
    assert "signal" in result
    assert "current_price" in result
    assert "moving_averages" in result
