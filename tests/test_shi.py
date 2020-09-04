import requests
from legitindicators import simple_harmonic_index

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_simple_harmonic_index():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    shi = simple_harmonic_index(close, 14)
    print(shi)
    assert len(shi) == len(close)
