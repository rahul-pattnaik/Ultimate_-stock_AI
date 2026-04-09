import ta

def add_volatility(df):

    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"],
        df["Low"],
        df["Close"]
    ).average_true_range()

    return df