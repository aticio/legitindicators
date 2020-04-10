import math
from legitindicators.supersmoother import superSmoother

def szladx(data, length, treshold):
    """A low lagging upgrade of ADX indicator.
    
    Arguments:
        data {list} -- list data consists of [high, low, close]
        length {int} -- lookback period of adx
        treshold {int} -- threshold line for adx
    
    Returns:
        [list] -- list of low lag adx indicator data
    """
    lag = (length - 1) / 2
    ssf = []
    smoothedTrueRange = []
    smoothedDirectionalMovementPlus = []
    smoothedDirectionalMovementMinus = []
    dx = []
    szladxi = []

    for i in range(0, len(data)):
        if i < round(lag):
            ssf.append(1)
            smoothedTrueRange.append(1)
            smoothedDirectionalMovementMinus.append(1)
            smoothedDirectionalMovementPlus.append(1)
            dx.append(1)
            szladxi.append(1)
        else:
            high = data[i][0]
            high1 = data[i-1][0]
            low = data[i][1]
            low1 = data[i-1][1]
            close = data[i][2]
            close1 = data[i-1][2]

            trueRange = max(max(high - low, abs(high - close1)), abs(low - close1))
            if high - high1 > low1 - low:
                directionalMovementPlus = max(high - high1, 0)
            else:
                directionalMovementPlus = 0

            if low1 - low > high - high1:
                directionalMovementMinus = max(low1 - low, 0)
            else:
                directionalMovementMinus = 0

            smoothedTrueRange.append(smoothedTrueRange[i-1] - (smoothedTrueRange[i-1] / length) + trueRange)
            smoothedDirectionalMovementPlus.append(smoothedDirectionalMovementPlus[i-1] - (smoothedDirectionalMovementPlus[i-1] / length) + directionalMovementPlus)
            smoothedDirectionalMovementMinus.append(smoothedDirectionalMovementMinus[i-1] - (smoothedDirectionalMovementMinus[i-1]/ length) + directionalMovementMinus) 
            
            diPlus = smoothedDirectionalMovementPlus[i] / smoothedTrueRange[i] * 100
            diMinus = smoothedDirectionalMovementMinus[i] / smoothedTrueRange[i] * 100
            dx.append(abs(diPlus - diMinus) / (diPlus+diMinus) * 100)
            
            szladxi.append(dx[i] + (dx[i] - dx[i-round(lag)]))
    
    ssf = superSmoother(szladxi,10)
    return ssf
