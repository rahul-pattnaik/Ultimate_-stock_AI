# technical/moving_averages.py
# ─────────────────────────────────────────────────────────────────────────────
# Moving Averages Engine
# Types: SMA · EMA · WMA · DEMA · TEMA · HMA (Hull MA)
# Output: DataFrame with all MAs + crossover signals + Golden/Death cross
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── MA Implementations ────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average — linear weights, recent bars get more weight."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: (x * weights).sum() / weights.sum(), raw=True
    )


def _dema(series: pd.Series, period: int) -> pd.Series:
    """Double EMA — reduces lag vs standard EMA."""
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    return 2 * e1 - e2


def _tema(series: pd.Series, period: int) -> pd.Series:
    """Triple EMA — even lower lag than DEMA."""
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return 3*e1 - 3*e2 + e3


def _hma(series: pd.Series, period: int) -> pd.Series:
    """
    Hull Moving Average — smoothest, lowest-lag MA.
    HMA(n) = WMA(2*WMA(n/2) − WMA(n)), sqrt(n)
    """
    half    = max(1, period // 2)
    sq_root = max(1, int(np.sqrt(period)))
    raw     = 2 * _wma(series, half) - _wma(series, period)
    return _wma(raw, sq_root)


# ── Public API ────────────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all MA columns to DataFrame — backward compatible.

    Adds columns:
        ma20, ma50, ma200              : Simple MAs
        ema9, ema21, ema50             : Exponential MAs
        wma20                          : Weighted MA
        dema20                         : Double EMA
        hma20                          : Hull MA
        golden_cross, death_cross      : bool signals
        ma_trend                       : label string
    """
    df   = df.copy()
    close = df["Close"]

    # SMA
    for p in [20, 50, 200]:
        df[f"ma{p}"] = close.rolling(p).mean()

    # EMA
    for span in [9, 21, 50]:
        df[f"ema{span}"] = _ema(close, span)

    # Advanced MAs
    df["wma20"]  = _wma(close, 20)
    df["dema20"] = _dema(close, 20)
    df["hma20"]  = _hma(close, 20)

    # Golden / Death Cross (MA50 crosses MA200)
    ma50_prev  = df["ma50"].shift(1)
    ma200_prev = df["ma200"].shift(1)
    df["golden_cross"] = (df["ma50"] > df["ma200"]) & (ma50_prev <= ma200_prev)
    df["death_cross"]  = (df["ma50"] < df["ma200"]) & (ma50_prev >= ma200_prev)

    # Overall MA trend label
    latest = df.iloc[-1]
    price  = float(latest["Close"])
    above  = sum([
        price > latest["ma20"],
        price > latest["ma50"],
        price > latest["ema21"],
    ])
    if   above == 3: df["ma_trend"] = "Bullish"
    elif above == 0: df["ma_trend"] = "Bearish"
    else:            df["ma_trend"] = "Mixed"

    return df


def ma_signal(df: pd.DataFrame) -> dict:
    """
    Moving average signal snapshot for the latest bar.

    Returns all MA values, crossover status, and actionable signal.
    """
    if df is None or len(df) < 50:
        return {"error": "Need at least 50 rows"}

    df_ma = add_moving_averages(df)
    row   = df_ma.iloc[-1]
    price = float(row["Close"])

    def safe(col):
        try:    return round(float(row[col]), 2)
        except: return None

    mas = {
        "SMA20":  safe("ma20"),  "SMA50":  safe("ma50"),  "SMA200": safe("ma200"),
        "EMA9":   safe("ema9"),  "EMA21":  safe("ema21"), "EMA50":  safe("ema50"),
        "WMA20":  safe("wma20"), "DEMA20": safe("dema20"),"HMA20":  safe("hma20"),
    }

    above = {k: (price > v) for k, v in mas.items() if v is not None}
    bull_count = sum(above.values())
    total      = len(above)

    if   bull_count == total:   signal = "STRONG BUY 🚀"
    elif bull_count >= total*0.7: signal = "BUY 📈"
    elif bull_count >= total*0.4: signal = "HOLD ➡️"
    elif bull_count >= total*0.2: signal = "SELL 📉"
    else:                         signal = "STRONG SELL 🔴"

    # Recent cross events (last 5 bars)
    recent_golden = bool(df_ma["golden_cross"].iloc[-5:].any())
    recent_death  = bool(df_ma["death_cross"].iloc[-5:].any())

    return {
        "signal":        signal,
        "price_vs_mas":  above,
        "bull_count":    f"{bull_count}/{total} MAs bullish",
        "moving_averages": mas,
        "golden_cross":  "⚡ Recent Golden Cross!" if recent_golden else "None",
        "death_cross":   "⚠️ Recent Death Cross!" if recent_death  else "None",
        "current_price": round(price, 2),
    }
