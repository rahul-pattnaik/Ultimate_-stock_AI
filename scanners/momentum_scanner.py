def momentum(df):

    rsi = df["rsi"].iloc[-1]

    if rsi > 60:
        return "Strong Momentum"

    return "Weak"