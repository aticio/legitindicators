import requests
from legitindicators import atr

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_atr():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    open = [float(o[1]) for o in data]
    high = [float(h[2]) for h in data]
    low = [float(lo[3]) for lo in data]
    close = [float(c[4]) for c in data]

    input_data = []
    for i in range(0, len(data)):
        ohlc = [open[i], high[i], low[i], close[i]]
        input_data.append(ohlc)
    a = atr(input_data, 14)
    print(a)
    assert len(a) == len(close)
