def add_moving_averages(df):

    df["ma50"] = df["Close"].rolling(50).mean()

    df["ma200"] = df["Close"].rolling(200).mean()

    return df