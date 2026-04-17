from __future__ import annotations

import io
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from .config import FETCH_RETRIES, FETCH_TIMEOUT, INTERVAL, MIN_BARS, PERIOD
from .data_fetcher import (
    SourceAttempt,
    get_latest_price,
    get_multiple_stocks,
    get_stock_data,
    validate_symbol,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketDataSnapshot:
    symbol: str
    source: str
    live_price: dict[str, Any]
    historical: Optional[pd.DataFrame]
    corporate_actions: pd.DataFrame
    metadata: dict[str, Any]


def _quiet_call(fn, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return fn(*args, **kwargs)


def _prepare_actions(actions: Any) -> pd.DataFrame:
    if actions is None:
        return pd.DataFrame(columns=["Dividends", "Stock Splits"])
    if isinstance(actions, pd.Series):
        actions = actions.to_frame(name="Actions")
    if not isinstance(actions, pd.DataFrame):
        return pd.DataFrame(columns=["Dividends", "Stock Splits"])
    prepared = actions.copy()
    prepared.index = pd.to_datetime(prepared.index)
    return prepared.sort_index()


class MarketDataEngine:
    def __init__(self, retries: int = FETCH_RETRIES, timeout: int = FETCH_TIMEOUT):
        self.retries = retries
        self.timeout = timeout

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        period: str = PERIOD,
        interval: str = INTERVAL,
        min_bars: int = MIN_BARS,
        use_cache: bool = True,
    ) -> tuple[str, Optional[pd.DataFrame], str]:
        validation = validate_symbol(symbol)
        candidates = validation.candidates
        attempts: list[SourceAttempt] = []
        if not validation.is_valid:
            return validation.normalized, None, "validation_failed"

        for candidate in candidates:
            data = get_stock_data(
                candidate,
                period=period,
                interval=interval,
                use_cache=use_cache,
                verbose=False,
                min_bars=min_bars,
            )
            if data is not None and not data.empty:
                return candidate, data, "core.data_fetcher"
            attempts.append(SourceAttempt(candidate, "core.data_fetcher", 1, False, "primary fetch unavailable"))

        for candidate in candidates:
            for attempt in range(1, self.retries + 1):
                try:
                    ticker = yf.Ticker(candidate)
                    raw = _quiet_call(
                        ticker.history,
                        period=period,
                        interval=interval,
                        auto_adjust=True,
                        actions=False,
                    )
                    if raw is not None and not raw.empty:
                        prepared = raw[[col for col in ["Open", "High", "Low", "Close", "Volume"] if col in raw.columns]].copy()
                        prepared.index = pd.to_datetime(prepared.index)
                        prepared = prepared.dropna().sort_index()
                        if len(prepared) >= min_bars:
                            return candidate, prepared, "yfinance.Ticker.history"
                        attempts.append(SourceAttempt(candidate, "yfinance.Ticker.history", attempt, False, "insufficient rows"))
                except Exception as exc:
                    attempts.append(SourceAttempt(candidate, "yfinance.Ticker.history", attempt, False, str(exc)))
                    if attempt < self.retries:
                        time.sleep(2 ** attempt)

        logger.warning("Historical OHLCV fetch failed for %s: %s", symbol, "; ".join(f"{item.symbol}:{item.source}:{item.message}" for item in attempts[-4:]))
        return validation.normalized, None, "unavailable"

    def fetch_live_price(self, symbol: str) -> dict[str, Any]:
        candidates = validate_symbol(symbol).candidates
        for candidate in candidates:
            quote = get_latest_price(candidate)
            if "error" not in quote:
                quote["source"] = "core.data_fetcher"
                return quote

        for candidate in candidates:
            try:
                ticker = yf.Ticker(candidate)
                info = _quiet_call(lambda: ticker.fast_info)
                price = float(info.get("lastPrice") or info.get("regularMarketPrice") or 0.0)
                previous = float(info.get("previousClose") or price or 0.0)
                if price > 0:
                    change = price - previous
                    return {
                        "symbol": candidate,
                        "price": round(price, 2),
                        "change": round(change, 2),
                        "change_pct": round((change / previous * 100) if previous else 0.0, 2),
                        "volume": int(info.get("lastVolume") or 0),
                        "high": round(float(info.get("dayHigh") or price), 2),
                        "low": round(float(info.get("dayLow") or price), 2),
                        "date": str(pd.Timestamp.utcnow().date()),
                        "trend": "Live Snapshot",
                        "source": "yfinance.fast_info",
                    }
            except Exception:
                continue

        return {"error": f"Could not fetch live price for {symbol}"}

    def fetch_corporate_actions(self, symbol: str) -> tuple[pd.DataFrame, str]:
        for candidate in validate_symbol(symbol).candidates:
            try:
                ticker = yf.Ticker(candidate)
                actions = _prepare_actions(_quiet_call(lambda: ticker.actions))
                if not actions.empty:
                    return actions.tail(10), "yfinance.actions"
            except Exception:
                continue
        return pd.DataFrame(columns=["Dividends", "Stock Splits"]), "unavailable"

    def fetch_snapshot(
        self,
        symbol: str,
        period: str = PERIOD,
        interval: str = INTERVAL,
        min_bars: int = MIN_BARS,
    ) -> MarketDataSnapshot:
        resolved_symbol, historical, history_source = self.fetch_historical_ohlcv(
            symbol=symbol,
            period=period,
            interval=interval,
            min_bars=min_bars,
        )
        live_price = self.fetch_live_price(resolved_symbol)
        corporate_actions, actions_source = self.fetch_corporate_actions(resolved_symbol)

        return MarketDataSnapshot(
            symbol=resolved_symbol,
            source=history_source,
            live_price=live_price,
            historical=historical,
            corporate_actions=corporate_actions,
            metadata={
                "history_source": history_source,
                "actions_source": actions_source,
                "supports_nse_bse": True,
                "cache_enabled": True,
                "fallback_chain": ["core.data_fetcher", "yfinance.Ticker.history", "yfinance.fast_info"],
            },
        )


def fetch_market_snapshot(symbol: str, period: str = PERIOD, interval: str = INTERVAL) -> MarketDataSnapshot:
    return MarketDataEngine().fetch_snapshot(symbol=symbol, period=period, interval=interval)


def fetch_batch_snapshots(symbols: list[str], period: str = PERIOD, interval: str = INTERVAL) -> dict[str, MarketDataSnapshot]:
    snapshots: dict[str, MarketDataSnapshot] = {}
    batch = get_multiple_stocks(symbols, period=period, interval=interval, verbose=False)
    for symbol in symbols:
        resolved = symbol.strip().upper()
        historical = batch.get(resolved)
        if historical is None:
            snapshots[resolved] = MarketDataEngine().fetch_snapshot(symbol=resolved, period=period, interval=interval)
            continue
        quote = get_latest_price(resolved)
        actions, source = MarketDataEngine().fetch_corporate_actions(resolved)
        snapshots[resolved] = MarketDataSnapshot(
            symbol=resolved,
            source="core.data_fetcher.batch",
            live_price=quote if "error" not in quote else {"symbol": resolved, "source": "batch"},
            historical=historical,
            corporate_actions=actions,
            metadata={
                "history_source": "core.data_fetcher.batch",
                "actions_source": source,
                "supports_nse_bse": True,
                "cache_enabled": True,
            },
        )
    return snapshots
