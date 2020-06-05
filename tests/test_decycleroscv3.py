import requests
from legitindicators import decycler_oscillator_v3

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_decycler_oscillator_v3():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    decosc = decycler_oscillator_v3(close, 75, 100)
    print(decosc)
    assert len(decosc) == len(close)
