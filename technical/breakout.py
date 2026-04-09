def breakout_signal(df):

    if len(df) < 20:
        return False

    resistance = df["High"].rolling(20).max()

    last_close = df["Close"].iloc[-1].item()
    last_resistance = resistance.iloc[-2].item()

    return last_close > last_resistance