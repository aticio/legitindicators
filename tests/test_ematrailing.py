import requests
from legitindicators import ema_trailing

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_ematrailing():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    close = [float(d[4]) for d in data]
    emavg, ts = ema_trailing(close, 14, 0.02)

    for i, emvg in enumerate(emavg):
        print(close[i], emavg[i], ts[i])
    assert len(emavg) == len(close)
