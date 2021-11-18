import requests
from legitindicators import decycler_oscillator, cube_transform

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_cube():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    dec = decycler_oscillator(close, 54, 1, 66, 1.2)
    cube = cube_transform(dec)
    print(cube)
    assert len(cube) == len(close)
