from __future__ import annotations

import pandas as pd

from core import data_fetcher


def test_prepare_ohlcv_flattens_multiindex_and_sorts(multiindex_df: pd.DataFrame) -> None:
    prepared = data_fetcher._prepare_ohlcv(multiindex_df)
    assert list(prepared.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert prepared.index.is_monotonic_increasing


def test_validate_rejects_missing_columns(missing_cols_df: pd.DataFrame) -> None:
    ok, reason = data_fetcher._validate(missing_cols_df, "TEST.NS", min_bars=10)
    assert ok is False
    assert "Missing columns" in reason


def test_validate_rejects_excessive_close_nans(nan_df: pd.DataFrame) -> None:
    ok, reason = data_fetcher._validate(nan_df, "TEST.NS", min_bars=10)
    assert ok is False
    assert "Too many nulls in Close" in reason


def test_get_stock_data_uses_cache(monkeypatch, bullish_df: pd.DataFrame) -> None:
    monkeypatch.setattr(data_fetcher, "_cache_valid", lambda path: True)
    monkeypatch.setattr(data_fetcher, "_load_cache", lambda path: bullish_df)
    monkeypatch.setattr(data_fetcher, "_quiet_download", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not run")))
    result = data_fetcher.get_stock_data("TEST.NS", use_cache=True, verbose=False)
    pd.testing.assert_frame_equal(result, bullish_df)


def test_get_stock_data_handles_timeout_gracefully(monkeypatch, capsys, no_sleep) -> None:
    monkeypatch.setattr(data_fetcher, "_quiet_download", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timeout")))
    result = data_fetcher.get_stock_data("BAD.NS", use_cache=False, verbose=False)
    captured = capsys.readouterr().out
    assert result is None
    assert "[ERROR] No data for BAD.NS" in captured


def test_get_multiple_stocks_falls_back_to_individual(monkeypatch, bullish_df: pd.DataFrame) -> None:
    monkeypatch.setattr(data_fetcher, "_quiet_download", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("batch failed")))

    def fake_get_stock_data(symbol: str, *args, **kwargs):
        return bullish_df if symbol == "A.NS" else None

    monkeypatch.setattr(data_fetcher, "get_stock_data", fake_get_stock_data)
    result = data_fetcher.get_multiple_stocks(["A.NS", "B.NS"], use_cache=False, verbose=False, min_bars=10)
    assert list(result.keys()) == ["A.NS"]


def test_get_latest_price_returns_error_when_data_missing(monkeypatch) -> None:
    monkeypatch.setattr(data_fetcher, "get_stock_data", lambda *args, **kwargs: None)
    result = data_fetcher.get_latest_price("MISSING.NS")
    assert "error" in result
