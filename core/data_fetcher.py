import hashlib
import io
import logging
import os
import pickle
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
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
from .nse_universe import canonical_symbol

logger = logging.getLogger(__name__)

SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-^_&]{1,20}$")


@dataclass
class SymbolValidationResult:
    is_valid: bool
    normalized: str
    candidates: list[str]
    reason: str = ""


@dataclass
class SourceAttempt:
    symbol: str
    source: str
    attempt: int
    success: bool
    message: str


def validate_symbol(symbol: str) -> SymbolValidationResult:
    normalized = canonical_symbol(symbol)
    if not normalized:
        return SymbolValidationResult(False, "", [], "Blank symbol")
    if not SYMBOL_PATTERN.match(normalized):
        return SymbolValidationResult(False, normalized, [], "Unsupported symbol format")
    if normalized.startswith(".") or normalized.endswith("."):
        return SymbolValidationResult(False, normalized, [], "Malformed ticker")
    if "." in normalized or normalized.startswith("^"):
        return SymbolValidationResult(True, normalized, [normalized], "OK")
    return SymbolValidationResult(True, normalized, [normalized, f"{normalized}.NS", f"{normalized}.BO"], "OK")


def _describe_no_data(symbol: str, reason: str = "") -> None:
    print(f"[ERROR] No data for {symbol}. Possible reasons:")
    print("   - Wrong symbol (Indian stocks often need .NS or .BO)")
    print("   - Delisted, unavailable, or temporarily unsupported")
    if reason:
        print(f"   - Validation / fetch detail: {reason}")


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


def _quiet_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)


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


def _fetch_download_source(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return _quiet_download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        timeout=FETCH_TIMEOUT,
        auto_adjust=True,
    )


def _fetch_ticker_history_source(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return _quiet_history(symbol, period=period, interval=interval)


def _run_source_with_retries(
    source_name: str,
    symbol: str,
    period: str,
    interval: str,
    retries: int,
    fetch_fn,
) -> tuple[Optional[pd.DataFrame], list[SourceAttempt]]:
    attempts: list[SourceAttempt] = []
    for attempt in range(1, retries + 1):
        try:
            raw = fetch_fn(symbol, period, interval)
            if raw is not None and not raw.empty:
                attempts.append(SourceAttempt(symbol, source_name, attempt, True, f"{len(raw)} rows"))
                return raw, attempts
            attempts.append(SourceAttempt(symbol, source_name, attempt, False, "Empty response"))
        except Exception as exc:
            attempts.append(SourceAttempt(symbol, source_name, attempt, False, str(exc)))
            wait = 2 ** attempt
            logger.warning("[%s] %s attempt %s failed: %s. Retry in %ss", symbol, source_name, attempt, exc, wait)
            if attempt < retries:
                time.sleep(wait)
    return None, attempts


def get_stock_data(
    symbol: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
    verbose: bool = False,
    min_bars: int = MIN_BARS,
) -> Optional[pd.DataFrame]:
    validation = validate_symbol(symbol)
    symbol = validation.normalized
    if not validation.is_valid:
        logger.error("[%s] Validation failed: %s", symbol or symbol, validation.reason)
        print(f"[ERROR] Symbol validation failed for {symbol or '<blank>'}: {validation.reason}")
        return None

    if use_cache:
        for candidate in validation.candidates:
            key = _cache_key(candidate, period, interval)
            path = _cache_path(key)
            if _cache_valid(path):
                cached = _load_cache(path)
                if cached is not None:
                    if verbose:
                        print(f"[CACHE] {candidate} ({len(cached)} rows)")
                    return cached

    source_chain = [
        ("yfinance.download", _fetch_download_source),
        ("yfinance.Ticker.history", _fetch_ticker_history_source),
    ]
    history: list[SourceAttempt] = []

    for candidate in validation.candidates:
        for source_name, fetch_fn in source_chain:
            raw, attempts = _run_source_with_retries(
                source_name=source_name,
                symbol=candidate,
                period=period,
                interval=interval,
                retries=FETCH_RETRIES,
                fetch_fn=fetch_fn,
            )
            history.extend(attempts)
            if raw is None or raw.empty:
                continue

            df = _prepare_ohlcv(raw)
            ok, reason = _validate(df, candidate, min_bars=min_bars)
            if not ok:
                logger.warning("[%s] %s validation failed: %s", candidate, source_name, reason)
                history.append(SourceAttempt(candidate, source_name, len(attempts), False, f"Validation failed: {reason}"))
                continue

            if verbose:
                print(f"[FETCH] {candidate} via {source_name} -> {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")

            if use_cache:
                _save_cache(_cache_path(_cache_key(candidate, period, interval)), df)
            return df

    logger.error("[%s] All fallback sources failed", symbol)
    last_reason = history[-1].message if history else validation.reason
    _describe_no_data(symbol, last_reason)
    return None


def get_multiple_stocks(
    symbols: list,
    period: str = PERIOD,
    interval: str = INTERVAL,
    use_cache: bool = True,
    verbose: bool = True,
    min_bars: int = MIN_BARS,
) -> dict:
    validated = [validate_symbol(symbol) for symbol in symbols]
    symbols = [item.normalized for item in validated if item.is_valid]
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

    unresolved = [item.normalized for item in validated if item.is_valid and item.normalized not in result and "." not in item.normalized and not item.normalized.startswith("^")]
    for symbol in unresolved:
        if symbol in result:
            continue
        df = get_stock_data(symbol, period=period, interval=interval, use_cache=use_cache, verbose=verbose, min_bars=min_bars)
        if df is not None:
            matching = next((candidate for candidate in validate_symbol(symbol).candidates if _cache_valid(_cache_path(_cache_key(candidate, period, interval)))), symbol)
            result[matching] = df

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
