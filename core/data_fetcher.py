import yfinance as yf

def get_stock_data(symbol):

    df = yf.download(symbol, period="1y")

    # Fix multi-index columns
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    return df