import requests
from legitindicators import smoothed_simple_harmonic_oscillator

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "YFIUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_smoothed_simple_harmonic_oscillator():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    ssho = smoothed_simple_harmonic_oscillator(close, 14)
    print(ssho)
    assert len(ssho) == len(close)
