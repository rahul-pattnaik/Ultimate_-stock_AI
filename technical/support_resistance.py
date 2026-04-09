import numpy as np

def get_support_resistance(df):

    prices = df["Close"]

    support = prices.rolling(20).min().iloc[-1]
    resistance = prices.rolling(20).max().iloc[-1]

    return support, resistance