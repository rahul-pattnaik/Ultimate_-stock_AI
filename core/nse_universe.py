# core/nse_universe.py
# ─────────────────────────────────────────────────────────────────────────────
# NSE Universe Fetcher
# Gets stock lists from NSE India with proper session cookies (required).
# Includes fallbacks for: Nifty 50 · Nifty 500 · sector-wise lists.
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging
import json
from typing import Optional

import requests

from .config import (NSE_HEADERS, NSE_SESSION_URL,
                     NSE_NIFTY500_URL, NSE_NIFTY50_URL,
                     NIFTY50_STOCKS)

logger = logging.getLogger(__name__)


# ── Hardcoded Fallbacks ───────────────────────────────────────────────────────
# NSE blocks automated requests aggressively. These are reliable fallbacks.

_NIFTY50_FALLBACK = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "ZOMATO.NS",
]

_NIFTY_IT_FALLBACK = [
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "OFSS.NS",
]

_NIFTY_BANK_FALLBACK = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "AUBANK.NS", "PNB.NS", "BANKBARODA.NS",
]

_NIFTY_PHARMA_FALLBACK = [
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS", "TORNTPHARM.NS", "ALKEM.NS", "AUROPHARMA.NS",
    "GLAND.NS", "IPCALAB.NS",
]

_NIFTY_AUTO_FALLBACK = [
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTORS.NS", "BALKRISIND.NS",
    "MOTHERSON.NS", "BOSCHLTD.NS",
]


# ── NSE Session Handler ───────────────────────────────────────────────────────

def _get_nse_session() -> requests.Session:
    """
    NSE requires a browser session with cookies.
    First hit the homepage to get cookies, then call the API.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Get cookies by visiting the homepage first
        session.get(NSE_SESSION_URL, timeout=15)
        time.sleep(1)   # polite delay
    except Exception as e:
        logger.warning(f"NSE session init failed: {e}")
    return session


def _fetch_nse_index(url: str) -> Optional[list]:
    """Fetch stock list from NSE API. Returns list of .NS symbols or None."""
    try:
        session = _get_nse_session()
        response = session.get(url, timeout=20)
        response.raise_for_status()
        body = response.text.strip()
        content_type = response.headers.get("Content-Type", "").lower()
        if not body:
            logger.info("NSE API returned an empty response; using fallback universe.")
            return None
        if "json" not in content_type and not body.startswith("{") and not body.startswith("["):
            logger.info("NSE API returned non-JSON content; using fallback universe.")
            return None
        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.info("NSE API returned invalid JSON; using fallback universe.")
            return None
        symbols = [x["symbol"] + ".NS" for x in data.get("data", [])]
        return symbols if symbols else None
    except Exception as e:
        logger.warning(f"NSE API call failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_nifty50(use_fallback: bool = True) -> list:
    """
    Get Nifty 50 stocks (50 symbols with .NS suffix).

    Args:
        use_fallback : if NSE API fails, use hardcoded list (default True)
    """
    symbols = _fetch_nse_index(NSE_NIFTY50_URL)
    if symbols:
        print(f"✅ Fetched {len(symbols)} Nifty 50 stocks from NSE")
        return symbols
    if use_fallback:
        print(f"⚠️  NSE API unavailable — using fallback Nifty 50 list ({len(_NIFTY50_FALLBACK)} stocks)")
        return _NIFTY50_FALLBACK
    return []


def get_nifty500(use_fallback: bool = True) -> list:
    """
    Get Nifty 500 stocks.
    Falls back to Nifty 50 if NSE API is unreachable.

    Args:
        use_fallback : if NSE API fails, use Nifty 50 as fallback
    """
    symbols = _fetch_nse_index(NSE_NIFTY500_URL)
    if symbols:
        print(f"✅ Fetched {len(symbols)} Nifty 500 stocks from NSE")
        return symbols
    if use_fallback:
        print(f"⚠️  NSE API unavailable — using Nifty 50 fallback")
        return _NIFTY50_FALLBACK
    return []


def get_sector(sector: str) -> list:
    """
    Get stocks for a specific NSE sector index.

    Supported sectors:
        'it' · 'bank' · 'pharma' · 'auto' · 'nifty50' · 'nifty500'

    Args:
        sector : sector name string (case-insensitive)

    Returns list of .NS ticker symbols.
    """
    sector = sector.lower().strip()

    sector_map = {
        "it":       ("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20IT",
                     _NIFTY_IT_FALLBACK),
        "bank":     ("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK",
                     _NIFTY_BANK_FALLBACK),
        "pharma":   ("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20PHARMA",
                     _NIFTY_PHARMA_FALLBACK),
        "auto":     ("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20AUTO",
                     _NIFTY_AUTO_FALLBACK),
        "nifty50":  (NSE_NIFTY50_URL,  _NIFTY50_FALLBACK),
        "nifty500": (NSE_NIFTY500_URL, _NIFTY50_FALLBACK),
    }

    if sector not in sector_map:
        available = ", ".join(sector_map.keys())
        print(f"❌ Unknown sector '{sector}'. Available: {available}")
        return []

    url, fallback = sector_map[sector]
    symbols = _fetch_nse_index(url)

    if symbols:
        print(f"✅ {sector.upper()}: {len(symbols)} stocks")
        return symbols

    print(f"⚠️  NSE API failed for {sector} — using fallback ({len(fallback)} stocks)")
    return fallback


def get_custom_watchlist(symbols: list) -> list:
    """
    Validate and normalise a custom list of symbols.
    Adds .NS suffix to bare NSE symbols if missing.

    Args:
        symbols : list of ticker strings

    Returns cleaned list.
    """
    cleaned = []
    for s in symbols:
        s = s.upper().strip()
        # If no exchange suffix and looks like NSE symbol, add .NS
        if "." not in s and not s.startswith("^"):
            s += ".NS"
        cleaned.append(s)
    return cleaned


def search_symbol(query: str) -> list:
    """
    Search NSE for a symbol by company name or partial ticker.

    Args:
        query : search string

    Returns list of matching {symbol, name} dicts.
    """
    try:
        session = _get_nse_session()
        url     = f"https://www.nseindia.com/api/search/autocomplete?q={query}"
        resp    = session.get(url, timeout=15)
        data    = resp.json()
        matches = data.get("symbols", [])
        return [{"symbol": m.get("symbol", "") + ".NS",
                 "name":   m.get("symbol_info", "")}
                for m in matches[:10]]
    except Exception as e:
        logger.warning(f"Symbol search failed: {e}")
        return []
