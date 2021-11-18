import requests
from legitindicators import roofing_filter

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_roofingfilter():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    r_f = roofing_filter(close, 45, 30)
    print(r_f)
    assert len(r_f) == len(close)
