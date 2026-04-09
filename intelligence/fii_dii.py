import pandas as pd

def analyze_fii_dii(df):

    df["fii_trend"] = df["FII_Net"].rolling(5).mean()

    df["dii_trend"] = df["DII_Net"].rolling(5).mean()

    if df["fii_trend"].iloc[-1] > 0:
        return "FII Buying"

    return "FII Selling"