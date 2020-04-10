import numpy as np
import math

def ema(data, length):
    """Exponential Moving Average
    
    Arguments:
        data {list} -- List of price data
        length {int} -- Lookback period for ema
    
    Returns:
        list -- EMA of given data
    """
    weights = np.exp(np.linspace(-1., .0, length))
    weights /= weights.sum()

    a = np.convolve(data, weights, mode="full")[:len(data)]
    a[:length] = a[length]
    return a

def atr(data, length):
    """Average True Range indicator
    
    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
        length {int} -- Lookback period for atr indicator
    
    Returns:
        list -- ATR of given ohlc data
    """
    tr = trueRange(data)
    a = rma(tr, length)
    return a

def rma(data, length):
    """Rolled moving average
    
    Arguments:
        data {list} -- List of price data
        length {int} -- Lookback period for rma
    
    Returns:
        list -- RMA of given data
    """
    alpha = 1 / length
    rma = []
    for i in range(0, len(data)):
        if i < 1:
            rma.append(0)
        else:
            rma.append(alpha * data[i] + (1 - alpha) * rma[i - 1])
    return rma

def roofingFilter(data, hpLength, ssLength):
    """Python implementation of the Roofing Filter indicator created by John Ehlers
    
    Arguments:
        data {list} -- list of price data
        hpLength {int} -- High Pass filter length
        ssLength {int} -- period for super smoother
    
    Returns:
        list -- roofin filter applied data
    """
    hp = []

    for i in range(0, len(data)):
        if i < 2:
            hp.append(0)
        else:
            alphaArg = 2 * 3.14159 / (hpLength * 1.414)
            alpha1 = (math.cos(alphaArg) + math.sin(alphaArg) - 1) / math.cos(alphaArg)
            hp.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hp[i-1] - math.pow(1-alpha1, 2)*hp[i-2])
    return superSmoother(hp,ssLength)

def superSmoother(data, length):
    """Python implementation of the Super Smoother indicator created by John Ehlers 
    
    Arguments:
        data {list} -- list of price data 
        length {int} -- period
    
    Returns:
        list -- super smoothed price data
    """
    ssf = []
    for i in range(0, len(data)):
        if i < 2:
            ssf.append(0)
        else:
            arg = 1.414 * 3.14159 / length
            a1 = math.exp(-arg)
            b1 = 2 * a1 * math.cos(4.44/float(length))
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            ssf.append(c1 * (data[i] + data[i-1]) / 2 + c2 * ssf[i-1] + c3 * ssf[i-2])
    return ssf

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

def trueRange(data):
    """True range
    
    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
    
    Returns:
        list -- True range of given data
    """
    tr = []
    for i in range(0,len(data)):
        if i < 1:
            tr.append(0)
        else:
            x = data[i][1] - data[i][2]
            y = abs(data[i][1] - data[i-1][3])
            z = abs(data[i][2] - data[i-1][3])

            if y <= x >= z:
                tr.append(x)
            elif x <= y >= z:
                tr.append(y)
            elif x <= z >= y:
                tr.append(z)
    return tr