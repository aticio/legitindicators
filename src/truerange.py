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