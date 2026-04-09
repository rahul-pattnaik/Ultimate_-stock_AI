import ta

def add_momentum_indicators(df):

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    df["macd"] = ta.trend.MACD(df["Close"]).macd()

    df["macd_signal"] = ta.trend.MACD(df["Close"]).macd_signal()

    return df