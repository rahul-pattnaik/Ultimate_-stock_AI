# technical/momentum.py
# ─────────────────────────────────────────────────────────────────────────────
# Momentum Indicators Engine (no `ta` dependency)
# Indicators: RSI · MACD · Stochastic · CCI · Williams %R · ROC · MFI
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── Indicator Functions ───────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist   = line - signal
    return line, signal, hist


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    mean    = typical.rolling(period).mean()
    mad     = typical.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (typical - mean) / (0.015 * mad + 1e-9)


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["High"].rolling(period).max()
    low_min  = df["Low"].rolling(period).min()
    return -100 * (high_max - df["Close"]) / (high_max - low_min + 1e-9)


def _roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change — momentum as % change over N bars."""
    return close.pct_change(period) * 100


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    mf      = typical * df["Volume"]
    pos_mf  = mf.where(typical > typical.shift(1), 0)
    neg_mf  = mf.where(typical < typical.shift(1), 0)
    mfr     = pos_mf.rolling(period).sum() / (neg_mf.rolling(period).sum() + 1e-9)
    return 100 - (100 / (1 + mfr))


# ── Public API ────────────────────────────────────────────────────────────────

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all momentum indicator columns — backward compatible.
    Replaces `ta` library dependency.

    Adds: rsi, macd, macd_signal, macd_hist, stoch_k, stoch_d,
          cci, williams_r, roc10, mfi
    """
    df = df.copy()

    df["rsi"]         = _rsi(df["Close"])
    macd, sig, hist   = _macd(df["Close"])
    df["macd"]        = macd
    df["macd_signal"] = sig
    df["macd_hist"]   = hist

    k, d = _stochastic(df)
    df["stoch_k"] = k
    df["stoch_d"] = d

    df["cci"]       = _cci(df)
    df["williams_r"]= _williams_r(df)
    df["roc10"]     = _roc(df["Close"], 10)
    df["mfi"]       = _mfi(df)

    return df


def momentum_report(df: pd.DataFrame) -> dict:
    """
    Full momentum snapshot for the latest bar.

    Scores each indicator and returns composite momentum verdict.
    """
    if df is None or len(df) < 30:
        return {"error": "Need at least 30 rows"}

    df_m = add_momentum_indicators(df)
    row  = df_m.iloc[-1]

    def s(col):
        try: return round(float(row[col]), 4)
        except: return None

    rsi        = s("rsi")
    macd_v     = s("macd")
    macd_sig   = s("macd_signal")
    macd_h     = s("macd_hist")
    stoch_k    = s("stoch_k")
    stoch_d    = s("stoch_d")
    cci_v      = s("cci")
    will_r     = s("williams_r")
    roc        = s("roc10")
    mfi_v      = s("mfi")

    # Simple scoring
    bulls = 0
    if rsi     and rsi < 50:         bulls += 1   # RSI oversold/neutral = potential
    if macd_h  and macd_h > 0:       bulls += 1   # MACD histogram positive
    if stoch_k and stoch_k < 20:     bulls += 1   # Stoch oversold
    if cci_v   and cci_v < -100:     bulls += 1   # CCI oversold
    if will_r  and will_r < -80:     bulls += 1   # Williams oversold
    if roc     and roc > 0:          bulls += 1   # Positive ROC
    if mfi_v   and mfi_v < 20:       bulls += 1   # MFI oversold

    total = 7
    if   bulls >= 6: momentum_signal = "STRONG BULLISH MOMENTUM 🚀"
    elif bulls >= 4: momentum_signal = "BULLISH MOMENTUM 📈"
    elif bulls >= 3: momentum_signal = "NEUTRAL MOMENTUM ➡️"
    elif bulls >= 1: momentum_signal = "BEARISH MOMENTUM 📉"
    else:            momentum_signal = "STRONG BEARISH MOMENTUM 🔴"

    return {
        "signal":       momentum_signal,
        "bull_count":   f"{bulls}/{total} indicators bullish",
        "indicators": {
            "RSI_14":       rsi,
            "MACD":         macd_v,
            "MACD_signal":  macd_sig,
            "MACD_hist":    macd_h,
            "Stoch_K":      stoch_k,
            "Stoch_D":      stoch_d,
            "CCI_20":       cci_v,
            "Williams_%R":  will_r,
            "ROC_10":       roc,
            "MFI_14":       mfi_v,
        },
        "overbought": {
            "RSI":     rsi > 70     if rsi    else False,
            "Stoch":   stoch_k > 80 if stoch_k else False,
            "CCI":     cci_v > 100  if cci_v  else False,
            "MFI":     mfi_v > 80   if mfi_v  else False,
        },
        "oversold": {
            "RSI":     rsi < 30     if rsi    else False,
            "Stoch":   stoch_k < 20 if stoch_k else False,
            "CCI":     cci_v < -100 if cci_v  else False,
            "MFI":     mfi_v < 20   if mfi_v  else False,
        },
    }
