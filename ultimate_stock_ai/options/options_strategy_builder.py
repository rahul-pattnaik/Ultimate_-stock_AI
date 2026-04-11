def bull_call_spread(price):

    buy_call = price * 1.02

    sell_call = price * 1.05

    return {

        "buy_call": buy_call,
        "sell_call": sell_call
    }