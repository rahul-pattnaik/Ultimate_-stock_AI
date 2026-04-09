import pandas as pd

def ai_signal(df):

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    if df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
        return "BUY"

    elif df["MA20"].iloc[-1] < df["MA50"].iloc[-1]:
        return "SELL"

    else:
        return "HOLD"