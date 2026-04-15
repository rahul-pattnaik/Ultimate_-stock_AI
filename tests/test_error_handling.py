from __future__ import annotations

import pandas as pd

from ai.ranking_engine import stock_score
from core.market_data_engine import _prepare_actions
from core.utils import safe_divide
from technical.breakout import breakout_signal


def test_safe_divide_returns_default_on_zero_division() -> None:
    assert safe_divide(10, 0, default=99) == 99


def test_stock_score_returns_zero_for_short_input(short_df: pd.DataFrame) -> None:
    assert stock_score(short_df) == 0


def test_breakout_signal_returns_false_for_empty_dataframe() -> None:
    assert breakout_signal(pd.DataFrame()) is False


def test_prepare_actions_handles_malformed_response() -> None:
    result = _prepare_actions("bad-response")
    assert result.empty


def test_non_positive_prices_are_rejected(bullish_df: pd.DataFrame) -> None:
    from core.data_fetcher import _validate

    df = bullish_df.copy()
    df.loc[df.index[-1], "Close"] = 0
    ok, reason = _validate(df, "TEST.NS", min_bars=10)
    assert ok is False
    assert "Non-positive prices detected" in reason
