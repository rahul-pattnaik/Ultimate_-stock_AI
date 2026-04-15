from __future__ import annotations

import pandas as pd

from technical.volatility import add_volatility, volatility_report


def test_add_volatility_adds_expected_columns(bullish_df: pd.DataFrame) -> None:
    result = add_volatility(bullish_df)
    for column in ["atr", "atr_pct", "hist_vol_20", "bb_width", "kc_upper", "kc_lower"]:
        assert column in result.columns


def test_volatility_report_contains_regime_and_keltner(bullish_df: pd.DataFrame) -> None:
    report = volatility_report(bullish_df)
    assert "volatility_regime" in report
    assert "keltner_position" in report
    assert report["current_price"] > 0


def test_volatility_report_handles_short_input(short_df: pd.DataFrame) -> None:
    assert "error" in volatility_report(short_df)
