import requests
from legitindicators import double_decycler, agc

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_double_decycler():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    ddec = double_decycler(close, 125, 10)
    a = agc(ddec)
    print(a)
    assert len(a) == len(close)
