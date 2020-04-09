import requests
from src.ema import ema

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_ema():
    response = requests.get(url = BINANCE_URL, params = PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    exp = ema(close,10)
    print(exp)
    assert len(exp) == len(exp)
