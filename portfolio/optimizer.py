import numpy as np

def sharpe_ratio(returns):

    mean = np.mean(returns)

    std = np.std(returns)

    sharpe = mean / std

    return sharpe