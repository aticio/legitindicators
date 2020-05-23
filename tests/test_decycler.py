import requests
from legitindicators import decycler

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_decycler():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    dec = decycler(close, 125)
    print(dec)
    assert len(dec) == len(close)
