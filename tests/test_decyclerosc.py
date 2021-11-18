import requests
from legitindicators import decycler_oscillator

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_decycler_oscillator():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    decosc = decycler_oscillator(close, 54, 1, 64, 1.2)
    print(decosc)
    assert len(decosc) == len(close)
