def swing_trade(df):

    if df["Close"].iloc[-1] > df["ma50"].iloc[-1]:
        return "Swing Buy"

    return "No Setup"