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