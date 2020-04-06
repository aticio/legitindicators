import requests
from src.roofingfilter import roofingFilter

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_roofingfilter():
    response = requests.get(url = BINANCE_URL, params = PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    rf = roofingFilter(close,45,30)
    print(rf)
    assert len(rf) == len(close)
