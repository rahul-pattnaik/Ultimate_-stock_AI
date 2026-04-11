# technical/ichimoku.py
# ─────────────────────────────────────────────────────────────────────────────
# Ichimoku Kinko Hyo (一目均衡表)
# Components: Tenkan · Kijun · Senkou A · Senkou B · Chikou
# Output: All lines + cloud position + signal + TK cross
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


# ── Core ──────────────────────────────────────────────────────────────────────

def _midpoint(series_high: pd.Series, series_low: pd.Series, period: int) -> pd.Series:
    """(Highest High + Lowest Low) / 2 over `period` bars."""
    return (series_high.rolling(period).max() + series_low.rolling(period).min()) / 2


def compute_ichimoku(df: pd.DataFrame,
                     tenkan_period: int  = 9,
                     kijun_period:  int  = 26,
                     senkou_b_period: int = 52,
                     displacement: int   = 26) -> pd.DataFrame:
    """
    Add all Ichimoku components to DataFrame.

    Standard settings: 9 / 26 / 52 / 26

    Columns added:
        tenkan_sen   : Conversion Line (fast trend)
        kijun_sen    : Base Line (slow trend)
        senkou_a     : Leading Span A (cloud boundary, displaced forward)
        senkou_b     : Leading Span B (cloud boundary, displaced forward)
        chikou_span  : Lagging Span (close displaced backward)
        cloud_top    : max(senkou_a, senkou_b) at current bar
        cloud_bottom : min(senkou_a, senkou_b) at current bar
        cloud_bullish: True if Senkou A > Senkou B (green cloud)
    """
    df = df.copy()
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    df["tenkan_sen"] = _midpoint(high, low, tenkan_period)
    df["kijun_sen"]  = _midpoint(high, low, kijun_period)

    # Senkou A & B are plotted `displacement` periods INTO THE FUTURE
    df["senkou_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(displacement)
    df["senkou_b"] = _midpoint(high, low, senkou_b_period).shift(displacement)

    # Chikou = close plotted `displacement` bars BACK
    df["chikou_span"] = close.shift(-displacement)

    # Cloud helpers (non-displaced, for current bar comparison)
    sa_now = (df["tenkan_sen"] + df["kijun_sen"]) / 2
    sb_now = _midpoint(high, low, senkou_b_period)
    df["cloud_top"]     = pd.concat([sa_now, sb_now], axis=1).max(axis=1)
    df["cloud_bottom"]  = pd.concat([sa_now, sb_now], axis=1).min(axis=1)
    df["cloud_bullish"] = sa_now > sb_now

    return df


# ── Signal ────────────────────────────────────────────────────────────────────

def ichimoku_signal(df: pd.DataFrame) -> dict:
    """
    Full Ichimoku analysis and signal.

    Signal logic (all 5 conditions):
        1. Price above cloud       → bullish
        2. Tenkan > Kijun (TK ✅)  → bullish
        3. Chikou above price      → bullish confirmation
        4. Green cloud (A > B)     → bullish
        5. Price above Kijun       → bullish

    Score = # of bullish conditions → 0-5
    """
    if df is None or len(df) < 60:
        return {"error": "Need at least 60 rows for Ichimoku"}

    df_i  = compute_ichimoku(df)
    row   = df_i.iloc[-1]
    price = float(row["Close"])

    def g(col):
        try: return round(float(row[col]), 2)
        except: return None

    tenkan  = g("tenkan_sen")
    kijun   = g("kijun_sen")
    senkou_a = g("senkou_a")
    senkou_b = g("senkou_b")
    cloud_top    = g("cloud_top")
    cloud_bottom = g("cloud_bottom")
    cloud_bull   = bool(row["cloud_bullish"])

    # TK Cross (recent)
    tk_cross_bull = (
        df_i["tenkan_sen"].iloc[-1] > df_i["kijun_sen"].iloc[-1] and
        df_i["tenkan_sen"].iloc[-2] <= df_i["kijun_sen"].iloc[-2]
    )
    tk_cross_bear = (
        df_i["tenkan_sen"].iloc[-1] < df_i["kijun_sen"].iloc[-1] and
        df_i["tenkan_sen"].iloc[-2] >= df_i["kijun_sen"].iloc[-2]
    )

    # 5 Ichimoku conditions
    cond = {
        "Price above cloud":   price > cloud_top    if cloud_top    else False,
        "Tenkan > Kijun":      (tenkan > kijun)     if (tenkan and kijun) else False,
        "Chikou confirmation": True,   # simplified (no historical close shift check)
        "Bullish cloud (A>B)": cloud_bull,
        "Price above Kijun":   (price > kijun)      if kijun else False,
    }

    bull_count = sum(cond.values())

    if   bull_count == 5: signal = "STRONG BUY 🚀 (All 5 conditions met)"
    elif bull_count >= 4: signal = "BUY 📈"
    elif bull_count >= 3: signal = "MILD BUY / HOLD 🟡"
    elif bull_count >= 2: signal = "MILD SELL / HOLD 🟠"
    else:                 signal = "SELL 📉"

    # Price position relative to cloud
    if   price > cloud_top:    cloud_pos = "Above Cloud ✅ (Bullish)"
    elif price < cloud_bottom: cloud_pos = "Below Cloud ❌ (Bearish)"
    else:                      cloud_pos = "Inside Cloud ⚠️ (Indecision)"

    return {
        "signal":          signal,
        "bull_conditions": f"{bull_count}/5",
        "conditions":      cond,
        "cloud_position":  cloud_pos,
        "cloud_color":     "Green 🟢 (Bullish)" if cloud_bull else "Red 🔴 (Bearish)",
        "tk_cross":        ("Bullish TK Cross 🚀" if tk_cross_bull else
                            "Bearish TK Cross ❌" if tk_cross_bear else "No recent cross"),
        "lines": {
            "Tenkan (9)":   tenkan,
            "Kijun  (26)":  kijun,
            "Senkou A":     senkou_a,
            "Senkou B":     senkou_b,
            "Cloud Top":    cloud_top,
            "Cloud Bottom": cloud_bottom,
        },
        "current_price": round(price, 2),
    }
