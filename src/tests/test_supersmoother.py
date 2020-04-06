import requests
from src.supersmoother import superSmoother

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_supersmoother():
    response = requests.get(url = BINANCE_URL, params = PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    ssf = superSmoother(close,10)
    print(ssf)
    assert len(ssf) == len(close)
