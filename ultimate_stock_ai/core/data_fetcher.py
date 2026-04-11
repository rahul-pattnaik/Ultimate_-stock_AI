# core/data_fetcher.py
# ─────────────────────────────────────────────────────────────────────────────
# Data Fetcher — yfinance wrapper
# FIXES:
#   ✅ MultiIndex column flattening (yfinance >= 0.2.18)
#   ✅ Auto-retry with exponential backoff
#   ✅ Disk caching (avoids re-downloading same data)
#   ✅ Data validation (min rows, positive prices, null check)
#   ✅ Single & batch download support
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import pickle
import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .config import (PERIOD, INTERVAL, MIN_BARS, CACHE_DIR,
                     CACHE_TTL_HOURS, FETCH_RETRIES, FETCH_TIMEOUT)

logger = logging.getLogger(__name__)


# ── Cache ─────────────────────────────────────────────────────────────────────

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
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")

def _load_cache(path: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ── Column Flattening (KEY FIX) ───────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance ≥ 0.2.18 returns MultiIndex columns even for single tickers:
        ('Close', 'RELIANCE.NS') → 'Close'
    This fix is applied EVERYWHERE before any computation.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame, symbol: str) -> tuple:
    if df is None or df.empty:
        return False, "Empty DataFrame"
    if len(df) < MIN_BARS:
        return False, f"Only {len(df)} rows (need {MIN_BARS})"
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    if df["Close"].isnull().mean() > 0.1:
        return False, "Too many nulls in Close"
    if (df["Close"].dropna() <= 0).any():
        return False, "Non-positive prices detected"
    return True, "OK"


# ── Core Fetch ────────────────────────────────────────────────────────────────

def get_stock_data(symbol: str,
                   period:    str  = PERIOD,
                   interval:  str  = INTERVAL,
                   use_cache: bool = True,
                   verbose:   bool = False) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single symbol.

    Args:
        symbol    : ticker e.g. "RELIANCE.NS" or "AAPL"
        period    : yfinance period string
        interval  : yfinance interval string
        use_cache : use disk cache
        verbose   : print status

    Returns:
        Clean DataFrame with [Open, High, Low, Close, Volume]
        or None on failure.
    """
    symbol = symbol.upper().strip()

    # Cache check
    if use_cache:
        key  = _cache_key(symbol, period, interval)
        path = _cache_path(key)
        if _cache_valid(path):
            df = _load_cache(path)
            if df is not None:
                if verbose:
                    print(f"[CACHE] {symbol} ({len(df)} rows)")
                return df

    # Download with retry
    df = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            raw = yf.download(
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
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"[{symbol}] Attempt {attempt} failed: {e}. Retry in {wait}s")
            if attempt < FETCH_RETRIES:
                time.sleep(wait)

    if df is None or df.empty:
        logger.error(f"[{symbol}] All {FETCH_RETRIES} download attempts failed")
        print(f"❌ No data for {symbol}. Possible reasons:")
        print(f"   • Wrong symbol (Indian stocks need .NS suffix, e.g. RELIANCE.NS)")
        print(f"   • Delisted or unavailable")
        return None

    # ── THE KEY FIX: flatten MultiIndex columns ───────────────────────
    df = _flatten_columns(df)

    # Keep only OHLCV
    available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[available].copy()
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.ffill().dropna()

    # Validate
    ok, reason = _validate(df, symbol)
    if not ok:
        logger.error(f"[{symbol}] Validation failed: {reason}")
        print(f"❌ Data validation failed for {symbol}: {reason}")
        return None

    if verbose:
        print(f"[FETCH] {symbol} → {len(df)} rows | "
              f"{df.index[0].date()} to {df.index[-1].date()}")

    # Cache save
    if use_cache:
        key  = _cache_key(symbol, period, interval)
        _save_cache(_cache_path(key), df)

    return df


# ── Batch Download ────────────────────────────────────────────────────────────

def get_multiple_stocks(symbols: list,
                        period:    str  = PERIOD,
                        interval:  str  = INTERVAL,
                        use_cache: bool = True,
                        verbose:   bool = True) -> dict:
    """
    Download data for multiple symbols efficiently.
    Returns dict {symbol: DataFrame} — only successful downloads.
    """
    symbols  = [s.upper().strip() for s in symbols]
    result   = {}
    to_fetch = []

    # Check cache first
    if use_cache:
        for sym in symbols:
            key  = _cache_key(sym, period, interval)
            path = _cache_path(key)
            if _cache_valid(path):
                df = _load_cache(path)
                if df is not None:
                    result[sym] = df
                    if verbose:
                        print(f"[CACHE] {sym}")
                    continue
            to_fetch.append(sym)
    else:
        to_fetch = list(symbols)

    if not to_fetch:
        return result

    if verbose:
        print(f"[FETCH] Downloading {len(to_fetch)} symbols...")

    try:
        raw = yf.download(
            to_fetch,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
            timeout=FETCH_TIMEOUT,
            auto_adjust=True,
        )
    except Exception as e:
        logger.error(f"Batch download failed: {e}. Falling back to individual.")
        raw = None

    if raw is not None and not raw.empty:
        for sym in to_fetch:
            try:
                if len(to_fetch) == 1:
                    df = raw.copy()
                elif sym in raw.columns.get_level_values(0):
                    df = raw[sym].copy()
                else:
                    df = None

                if df is None or (hasattr(df, 'empty') and df.empty):
                    continue

                df = _flatten_columns(df)
                available = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                             if c in df.columns]
                df = df[available].dropna()
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                df = df.ffill().dropna()

                ok, reason = _validate(df, sym)
                if ok:
                    result[sym] = df
                    if use_cache:
                        key  = _cache_key(sym, period, interval)
                        _save_cache(_cache_path(key), df)
                    if verbose:
                        print(f"  ✅ {sym} ({len(df)} rows)")
                else:
                    if verbose:
                        print(f"  ❌ {sym}: {reason}")

            except Exception as e:
                logger.warning(f"  ❌ {sym}: {e}")
    else:
        for sym in to_fetch:
            df = get_stock_data(sym, period, interval, use_cache, verbose)
            if df is not None:
                result[sym] = df

    return result


# ── Latest Price Snapshot ─────────────────────────────────────────────────────

def get_latest_price(symbol: str) -> dict:
    """Fetch latest price info (fast 5d window)."""
    df = get_stock_data(symbol, period="5d", interval="1d",
                        use_cache=False, verbose=False)
    if df is None or df.empty:
        return {"error": f"Could not fetch {symbol}"}

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest

    close  = float(latest["Close"])
    prev_c = float(prev["Close"])
    change = close - prev_c
    pct    = change / prev_c * 100

    return {
        "symbol":     symbol,
        "price":      round(close, 2),
        "change":     round(change, 2),
        "change_pct": round(pct, 2),
        "volume":     int(latest["Volume"]),
        "high":       round(float(latest["High"]), 2),
        "low":        round(float(latest["Low"]), 2),
        "date":       str(df.index[-1].date()),
        "trend":      "UP 📈" if change > 0 else "DOWN 📉",
    }


# ── Cache Management ──────────────────────────────────────────────────────────

def clear_cache(symbol: str = None, period: str = PERIOD,
                interval: str = INTERVAL) -> None:
    if symbol:
        key  = _cache_key(symbol.upper(), period, interval)
        path = _cache_path(key)
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleared cache for {symbol}")
    else:
        if os.path.exists(CACHE_DIR):
            for f in os.listdir(CACHE_DIR):
                os.remove(os.path.join(CACHE_DIR, f))
            print(f"Cleared all cache ({CACHE_DIR})")
