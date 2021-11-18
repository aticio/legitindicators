import requests
from legitindicators import bollinger_bands_pb

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_bollinger_bands_pb():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    bbr = bollinger_bands_pb(close, 20, 2)
    print(bbr)
    assert len(bbr) == len(close)
