def insider_signal(insider_buy, insider_sell):

    if insider_buy > insider_sell:
        return "Bullish"

    if insider_sell > insider_buy:
        return "Bearish"

    return "Neutral"