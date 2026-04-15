import hashlib
import io
import logging
import os
import pickle
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import (
    CACHE_DIR,
    CACHE_TTL_HOURS,
    FETCH_RETRIES,
    FETCH_TIMEOUT,
    INTERVAL,
    MIN_BARS,
    PERIOD,
)

logger = logging.getLogger(__name__)


def _cache_key(symbol: str, period: str, interval: str) -> str:
    return hashlib.md5(f"{symbol}_{period}_{interval}".encode()).hexdigest()


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def _cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) / 3600 < CACHE_TTL_HOURS


def _save_cache(path: str, df: pd.DataFrame) -> None:
    try:
        with open(path, "wb") as handle:
            pickle.dump(df, handle)
    except Exception as exc:
        logger.warning(f"Cache write failed: {exc}")


def _load_cache(path: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _quiet_download(*args, **kwargs) -> pd.DataFrame:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return yf.download(*args, **kwargs)


def _validate(df: pd.DataFrame, symbol: str, min_bars: int = MIN_BARS) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "Empty DataFrame"
    if len(df) < min_bars:
        return False, f"Only {len(df)} rows (need {min_bars})"
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    if df["Close"].isnull().mean() > 0.1:
        return False, "Too many nulls in Close"
    if (df["Close"].dropna() <= 0).any():
        return False, "Non-positive prices detected"
    return True, "OK"


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    available = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in df.columns]
    cleaned = df[available].copy()
    cleaned = cleaned.dropna(how="all")
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned.sort_index(inplace=True)
    cleaned = cleaned.ffill().dropna()
    return cleaned


def get_stock_data(
    symbol: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
    verbose: bool = False,
    min_bars: int = MIN_BARS,
) -> Optional[pd.DataFrame]:
    symbol = symbol.upper().strip()

    if use_cache:
        key = _cache_key(symbol, period, interval)
        path = _cache_path(key)
        if _cache_valid(path):
            cached = _load_cache(path)
            if cached is not None:
                if verbose:
                    print(f"[CACHE] {symbol} ({len(cached)} rows)")
                return cached

    df = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            raw = _quiet_download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                timeout=FETCH_TIMEOUT,
                auto_adjust=True,
            )
            if raw is not None and not raw.empty:
                df = raw
                break
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(f"[{symbol}] Attempt {attempt} failed: {exc}. Retry in {wait}s")
            if attempt < FETCH_RETRIES:
                time.sleep(wait)

    if df is None or df.empty:
        logger.error(f"[{symbol}] All {FETCH_RETRIES} download attempts failed")
        print(f"[ERROR] No data for {symbol}. Possible reasons:")
        print("   - Wrong symbol (Indian stocks need .NS suffix, e.g. RELIANCE.NS)")
        print("   - Delisted or unavailable")
        return None

    df = _prepare_ohlcv(df)

    ok, reason = _validate(df, symbol, min_bars=min_bars)
    if not ok:
        logger.error(f"[{symbol}] Validation failed: {reason}")
        print(f"[ERROR] Data validation failed for {symbol}: {reason}")
        return None

    if verbose:
        print(f"[FETCH] {symbol} -> {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")

    if use_cache:
        _save_cache(_cache_path(_cache_key(symbol, period, interval)), df)

    return df


def get_multiple_stocks(
    symbols: list,
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
    verbose: bool = True,
    min_bars: int = MIN_BARS,
) -> dict:
    symbols = [symbol.upper().strip() for symbol in symbols]
    result = {}
    to_fetch = []

    if use_cache:
        for symbol in symbols:
            key = _cache_key(symbol, period, interval)
            path = _cache_path(key)
            if _cache_valid(path):
                cached = _load_cache(path)
                if cached is not None:
                    result[symbol] = cached
                    if verbose:
                        print(f"[CACHE] {symbol}")
                    continue
            to_fetch.append(symbol)
    else:
        to_fetch = list(symbols)

    if not to_fetch:
        return result

    if verbose:
        print(f"[FETCH] Downloading {len(to_fetch)} symbols...")

    try:
        raw = _quiet_download(
            to_fetch,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
            timeout=FETCH_TIMEOUT,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.error(f"Batch download failed: {exc}. Falling back to individual.")
        raw = None

    if raw is not None and not raw.empty:
        for symbol in to_fetch:
            try:
                if len(to_fetch) == 1:
                    df = raw.copy()
                elif symbol in raw.columns.get_level_values(0):
                    df = raw[symbol].copy()
                else:
                    df = None

                if df is None or df.empty:
                    continue

                df = _prepare_ohlcv(df)
                ok, reason = _validate(df, symbol, min_bars=min_bars)
                if ok:
                    result[symbol] = df
                    if use_cache:
                        _save_cache(_cache_path(_cache_key(symbol, period, interval)), df)
                    if verbose:
                        print(f"  [OK] {symbol} ({len(df)} rows)")
                elif verbose:
                    print(f"  [SKIP] {symbol}: {reason}")
            except Exception as exc:
                logger.warning(f"  [SKIP] {symbol}: {exc}")
    else:
        for symbol in to_fetch:
            df = get_stock_data(symbol, period, interval, use_cache, verbose)
            if df is not None:
                result[symbol] = df

    return result


def get_latest_price(symbol: str) -> dict:
    df = get_stock_data(symbol, period="5d", interval="1d", use_cache=False, verbose=False, min_bars=2)
    if df is None or df.empty:
        return {"error": f"Could not fetch {symbol}"}

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest
    close = float(latest["Close"])
    previous_close = float(previous["Close"])
    change = close - previous_close
    change_pct = change / previous_close * 100

    return {
        "symbol": symbol,
        "price": round(close, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "volume": int(latest["Volume"]),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "date": str(df.index[-1].date()),
        "trend": "UP" if change > 0 else "DOWN",
    }


def clear_cache(symbol: str = None, period: str = PERIOD, interval: str = INTERVAL) -> None:
    if symbol:
        key = _cache_key(symbol.upper(), period, interval)
        path = _cache_path(key)
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleared cache for {symbol}")
        return

    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, filename))
        print(f"Cleared all cache ({CACHE_DIR})")
