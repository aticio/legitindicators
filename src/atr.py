from src.truerange import trueRange
from src.rma import rma

def atr(data, length):
    """Average True Range indicator
    
    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
        length {int} -- Lookback period for atr indicator
    
    Returns:
        list -- ATR of given ohlc data
    """
    tr = trueRange(data)
    a = rma(tr, length)
    return a
