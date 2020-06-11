import requests
from legitindicators import szladx_v2

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_szladxv2():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    high = [float(h[2]) for h in data]
    low = [float(l[3]) for l in data]
    close = [float(c[4]) for c in data]

    input_data = []
    for i in range(0,len(data)):
        hlc = [high[i], low[i], close[i]]
        input_data.append(hlc)

    szladx_data = szladx_v2(input_data, 14)
    print(szladx_data)
    assert len(szladx_data) == len(data)