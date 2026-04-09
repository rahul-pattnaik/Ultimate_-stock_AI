def breakout(df):

    resistance = df["High"].rolling(20).max()

    if df["Close"].iloc[-1] > resistance.iloc[-2]:
        return True

    return False