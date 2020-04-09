import numpy as np

def ema(data, length):
    """Exponential Moving Average
    
    Arguments:
        data {list} -- List of price data
        length {int} -- Lookback period for ema
    
    Returns:
        list -- EMA of given data
    """
    weights = np.exp(np.linspace(-1., .0, length))
    weights /= weights.sum()

    a = np.convolve(data, weights, mode="full")[:len(data)]
    a[:length] = a[length]
    return a