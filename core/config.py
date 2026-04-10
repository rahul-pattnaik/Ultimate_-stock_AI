# core/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Ultimate Stock AI — Central Configuration
# All tunable parameters in one place. Import from here everywhere.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Data Settings ─────────────────────────────────────────────────────────────

PERIOD   = "1y"       # yfinance period: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
INTERVAL = "1d"       # yfinance interval: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo

# Minimum bars required for full analysis
MIN_BARS = 60

# ── Default Watchlists ────────────────────────────────────────────────────────

# Large-cap NSE blue chips (always reliable)
NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "POWERGRID.NS", "NTPC.NS",
    "ONGC.NS", "COALINDIA.NS", "TATAMOTORS.NS", "M&M.NS", "TATASTEEL.NS",
]

# Quick default list for fast testing
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

# US stocks for reference / comparison
US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# ── Scoring Thresholds ────────────────────────────────────────────────────────

SCORE_STRONG_BUY = 80
SCORE_BUY        = 60
SCORE_HOLD       = 40
SCORE_SELL       = 20   # below this = STRONG SELL

# ── Technical Indicator Parameters ───────────────────────────────────────────

RSI_PERIOD        = 14
RSI_OVERSOLD      = 30
RSI_OVERBOUGHT    = 70

MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9

BB_PERIOD         = 20
BB_STD            = 2.0

ATR_PERIOD        = 14

SUPERTREND_PERIOD = 10
SUPERTREND_MULT   = 3.0

ICHIMOKU_TENKAN   = 9
ICHIMOKU_KIJUN    = 26
ICHIMOKU_SENKOU_B = 52

MA_PERIODS        = [20, 50, 200]
EMA_PERIODS       = [9, 21, 50]

VWAP_SD_BANDS     = [1.0, 2.0]

FIBONACCI_LOOKBACK = 60
ZIGZAG_THRESHOLD   = 0.03

# ── Backtester Defaults ───────────────────────────────────────────────────────

BACKTEST_INITIAL_CAPITAL = 100_000   # INR / USD
BACKTEST_COMMISSION      = 0.001     # 0.1% per trade
BACKTEST_SLIPPAGE        = 0.001     # 0.1% slippage
BACKTEST_POSITION_SIZE   = 0.10      # 10% of capital per trade (risk management)
BACKTEST_STOP_LOSS       = 0.05      # 5% stop-loss
BACKTEST_TAKE_PROFIT     = 0.15      # 15% take-profit

# ── Data Fetcher Settings ─────────────────────────────────────────────────────

FETCH_RETRIES     = 3
FETCH_TIMEOUT     = 30              # seconds
CACHE_DIR         = ".cache"        # local disk cache directory
CACHE_TTL_HOURS   = 4               # refresh data after N hours

# ── NSE Universe ─────────────────────────────────────────────────────────────

NSE_SESSION_URL   = "https://www.nseindia.com"
NSE_NIFTY500_URL  = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
NSE_NIFTY50_URL   = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

# ── Display / Output ──────────────────────────────────────────────────────────

DISPLAY_SEPARATOR = "=" * 50
CURRENCY_SYMBOL   = "₹"            # change to "$" for US stocks

# ── Environment / API Keys ────────────────────────────────────────────────────

# Load from .env or environment variables (never hardcode secrets)
ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_KEY",  "")
NEWS_API_KEY       = os.getenv("NEWS_API_KEY",        "")
FINNHUB_KEY        = os.getenv("FINNHUB_KEY",         "")
