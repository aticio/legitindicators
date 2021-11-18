import requests
from legitindicators import volume_heat

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL, "limit": 1000}


def test_volume_heat():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    volume = [float(v[5]) for v in data]
    vh = volume_heat(volume, 610)
    print(vh)
    assert len(vh) == len(volume)
