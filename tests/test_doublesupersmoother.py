import requests
from legitindicators import double_super_smoother

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "MTLUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_doublesupersmoother():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    dssf = double_super_smoother(close, 50, 200)
    print(dssf)
    assert len(dssf) == len(close)
