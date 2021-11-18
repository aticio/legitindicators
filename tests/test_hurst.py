import requests
from legitindicators import hurst_coefficient

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_hurst_coefficient():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    hurst = hurst_coefficient(close, 30)
    print(hurst[-20:])
    assert len(hurst) == len(close)
