import numpy as np

def volume_profile(df, bins=20):

    price = df["Close"]

    volume = df["Volume"]

    hist, edges = np.histogram(price, bins=bins, weights=volume)

    poc = edges[np.argmax(hist)]

    return poc