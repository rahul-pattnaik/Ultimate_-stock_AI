def detect_trend(df):

    close = df["Close"].squeeze()   # ensure 1D series

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    ma50_last = float(ma50.iloc[-1])
    ma200_last = float(ma200.iloc[-1])

    if ma50_last > ma200_last:
        return "Uptrend"

    elif ma50_last < ma200_last:
        return "Downtrend"

    else:
        return "Sideways"