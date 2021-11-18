import requests
from legitindicators import momentum_normalized

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_momnom():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(c[4]) for c in data]

    mn = momentum_normalized(close, 10)

    print(mn)
    assert len(mn) == len(data)
