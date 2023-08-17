import requests
from legitindicators import bollinger_bands_width_normalized

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_bollinger_bands_width_normalized():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    bbwn = bollinger_bands_width_normalized(close, 20, 2)
    print(bbwn)
    assert len(bbwn) == len(close)
