import requests
from legitindicators import kaufman_er

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_hurst_coefficient():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    ker = kaufman_er(close, 16)
    print(ker)
    assert len(ker) == len(close)
