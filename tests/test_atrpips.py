import requests
from src.legitindicators import atrpips

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_atrpips():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    open = [float(o[1]) for o in data]
    high = [float(h[2]) for h in data]
    low = [float(l[3]) for l in data]
    close = [float(c[4]) for c in data]

    input_data = []
    for i in range(0, len(data)):
        ohlc = [open[i], high[i], low[i], close[i]]
        input_data.append(ohlc)
    apips = atrpips(input_data, 14)
    print(apips)
    assert len(apips) == len(close)
