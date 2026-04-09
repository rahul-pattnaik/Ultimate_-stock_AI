def stock_score(df):

    score = 0

    price = float(df["Close"].iloc[-1])

    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    ma200 = float(df["Close"].rolling(200).mean().iloc[-1])

    # Trend
    if price > ma20:
        score += 20

    if ma20 > ma50:
        score += 20

    if ma50 > ma200:
        score += 20

    # Momentum
    if float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-5]):
        score += 20

    # Breakout
    high20 = float(df["High"].rolling(20).max().iloc[-2])
    if price > high20:
        score += 20

    return score