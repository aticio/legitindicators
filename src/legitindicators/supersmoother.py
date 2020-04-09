import math

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

