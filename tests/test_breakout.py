from __future__ import annotations

import pandas as pd

from technical.breakout import breakout_analysis, breakout_signal


def test_breakout_signal_detects_price_breakout(breakout_df: pd.DataFrame) -> None:
    assert breakout_signal(breakout_df) is True


def test_breakout_analysis_returns_bounded_score(breakout_df: pd.DataFrame) -> None:
    result = breakout_analysis(breakout_df)
    assert 0 <= result["score"] <= 100
    assert isinstance(result["triggered"], bool)


def test_breakout_analysis_handles_insufficient_data(short_df: pd.DataFrame) -> None:
    assert "error" in breakout_analysis(short_df)
