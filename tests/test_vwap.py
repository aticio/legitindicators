import requests
from legitindicators import vwap

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
PARAMS = {"symbol": SYMBOL, "interval": INTERVAL, "limit": 26}


def test_vwap():
    response = requests.get(url=BINANCE_URL, params=PARAMS)
    data = response.json()
    high = [float(h[2]) for h in data]
    low = [float(lo[3]) for lo in data]
    close = [float(c[4]) for c in data]
    volume = [float(v[5]) for v in data]

    input_data = []
    for i in range(0, len(data)):
        hlcv = [high[i], low[i], close[i], volume[i]]
        input_data.append(hlcv)
    
    vwap_data = vwap(input_data)
    print(vwap_data, len(vwap_data))
    assert len(vwap_data) == len(close)


