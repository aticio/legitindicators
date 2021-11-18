import requests
from legitindicators import szladx

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL}


def test_szladx():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    high = [float(h[2]) for h in data]
    low = [float(lo[3]) for lo in data]
    close = [float(c[4]) for c in data]

    input_data = []
    for i in range(0, len(data)):
        hlc = [high[i], low[i], close[i]]
        input_data.append(hlc)

    szladx_data = szladx(input_data, 14)
    print(data[-1][0], szladx_data[-1])
    assert len(szladx_data) == len(data)
