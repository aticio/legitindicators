import requests
from legitindicators.atr import atr

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_atr():
    response = requests.get(url = BINANCE_URL, params = PARAMS)
    data = response.json()
    open = [float(o[1]) for o in data]
    high = [float(h[2]) for h in data]
    low = [float(l[3]) for l in data]
    close = [float(c[4]) for c in data]

    inputData = []
    for i in range(0,len(data)):
        ohlc = [open[i], high[i], low[i], close[i]]
        inputData.append(ohlc)
    a = atr(inputData,14)
    print(a)
    assert len(a) == len(a)
