import math
from legitindicators.supersmoother import superSmoother

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

