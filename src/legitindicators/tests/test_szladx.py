import requests
from src.szladx import szladx

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
PARAMS = {"symbol":SYMBOL, "interval":INTERVAL}

def test_szladx():
    response = requests.get(url = BINANCE_URL, params = PARAMS)
    data = response.json()
    high = [float(h[2]) for h in data]
    low = [float(l[3]) for l in data]
    close = [float(c[4]) for c in data]

    inputData = []
    for i in range(0,len(data)):
        hlc = [high[i], low[i], close[i]]
        inputData.append(hlc)
    
    szladxData = szladx(inputData,14,20)
    print(szladxData)
    assert len(szladxData) == len(data)
