import requests
from legitindicators import linreg_slope

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_linreg_slope():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    lrs = linreg_slope(close, 24)
    print(lrs)
    assert len(lrs) == len(close)
