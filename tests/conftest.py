from __future__ import annotations

import builtins
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pytest


def make_ohlcv_dataframe(
    periods: int = 260,
    trend: str = "bullish",
    volume_base: float = 1_000_000,
    add_nans: bool = False,
) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    base = np.linspace(100, 180, periods)
    if trend == "bearish":
        base = np.linspace(180, 100, periods)
    elif trend == "sideways":
        base = 140 + np.sin(np.linspace(0, 10 * np.pi, periods)) * 8
    elif trend == "breakout":
        base = np.linspace(100, 118, periods)
        base[-5:] = np.array([118, 119, 120, 121, 135], dtype=float)

    wave = np.sin(np.linspace(0, 6 * np.pi, periods))
    close = base + wave
    open_ = close - 0.8
    high = close + 1.5
    low = close - 1.5
    volume = np.full(periods, volume_base, dtype=float)
    if trend == "breakout":
        volume[-1] = volume_base * 3.5

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )
    if add_nans:
        df.loc[df.index[10:15], "Close"] = np.nan
    return df


@pytest.fixture
def bullish_df() -> pd.DataFrame:
    return make_ohlcv_dataframe(trend="bullish")


@pytest.fixture
def bearish_df() -> pd.DataFrame:
    return make_ohlcv_dataframe(trend="bearish")


@pytest.fixture
def sideways_df() -> pd.DataFrame:
    return make_ohlcv_dataframe(trend="sideways")


@pytest.fixture
def breakout_df() -> pd.DataFrame:
    return make_ohlcv_dataframe(trend="breakout")


@pytest.fixture
def multiindex_df(bullish_df: pd.DataFrame) -> pd.DataFrame:
    df = bullish_df.copy()
    df.columns = pd.MultiIndex.from_product([df.columns, ["TEST.NS"]])
    return df


@pytest.fixture
def short_df() -> pd.DataFrame:
    return make_ohlcv_dataframe(periods=15, trend="bullish")


@pytest.fixture
def missing_cols_df(bullish_df: pd.DataFrame) -> pd.DataFrame:
    return bullish_df.drop(columns=["Volume"])


@pytest.fixture
def nan_df(bullish_df: pd.DataFrame) -> pd.DataFrame:
    df = bullish_df.copy()
    df.loc[df.index[:40], "Close"] = np.nan
    return df


@pytest.fixture
def dummy_market_snapshot():
    class Snapshot:
        symbol = "RELIANCE.NS"
        source = "mock"
        live_price = {"price": 150.0}
        corporate_actions = pd.DataFrame()
        metadata = {"history_source": "mock", "actions_source": "mock"}

    return Snapshot()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)


def feed_inputs(monkeypatch: pytest.MonkeyPatch, values: Iterable[str]) -> None:
    iterator = iter(values)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(iterator))


@pytest.fixture
def input_feeder(monkeypatch: pytest.MonkeyPatch) -> Callable[[Iterable[str]], None]:
    def _apply(values: Iterable[str]) -> None:
        feed_inputs(monkeypatch, values)

    return _apply
