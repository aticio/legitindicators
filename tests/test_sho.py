import requests
from legitindicators import simple_harmonic_oscillator

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_simple_harmonic_oscillator():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    sho = simple_harmonic_oscillator(close, 14)
    print(sho)
    assert len(sho) == len(close)
