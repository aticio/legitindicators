import requests
from legitindicators import decycler_oscillator, agc

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "DOGEUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_agc():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    dec = decycler_oscillator(close, 67, 1, 88, 1.2)
    a = agc(dec)
    print(a)
    assert len(a) == len(close)
