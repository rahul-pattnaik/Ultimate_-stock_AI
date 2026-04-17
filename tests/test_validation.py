from __future__ import annotations

import pandas as pd

import main_terminal


def test_resolve_symbol_keeps_existing_suffix() -> None:
    assert main_terminal.resolve_symbol("INFY.NS") == "INFY.NS"


def test_resolve_symbol_returns_empty_for_blank() -> None:
    assert main_terminal.resolve_symbol("   ") == ""


def test_resolve_symbol_adds_nse_suffix_when_probe_succeeds(monkeypatch, bullish_df: pd.DataFrame) -> None:
    monkeypatch.setattr(
        main_terminal,
        "_quiet_yfinance_download",
        lambda ticker: bullish_df if ticker == "RELIANCE.NS" else pd.DataFrame(),
    )
    assert main_terminal.resolve_symbol("reliance") == "RELIANCE.NS"


def test_resolve_symbol_returns_original_when_no_probe_matches(monkeypatch) -> None:
    monkeypatch.setattr(main_terminal, "_quiet_yfinance_download", lambda ticker: pd.DataFrame())
    assert main_terminal.resolve_symbol("unknown") == "UNKNOWN"


def test_resolve_symbol_maps_legacy_indian_alias_without_probe() -> None:
    assert main_terminal.resolve_symbol("zomato") == "ETERNAL"


def test_validate_symbol_allows_valid_ampersand_ticker() -> None:
    from core.data_fetcher import validate_symbol

    result = validate_symbol("M&M.NS")
    assert result.is_valid is True
    assert result.normalized == "M&M.NS"


def test_lowercase_ohlcv_converts_columns(bullish_df: pd.DataFrame) -> None:
    lowered = main_terminal._lowercase_ohlcv(bullish_df)
    assert list(lowered.columns) == ["open", "high", "low", "close", "volume"]


def test_safe_wrapper_returns_error_dict() -> None:
    result = main_terminal._safe(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert result == {"error": "boom"}
