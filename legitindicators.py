"""legitindicators"""
import math
import numpy as np

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

    res = np.convolve(data, weights, mode="full")[:len(data)]
    res[:length] = res[length]
    return res

def atr(data, length):
    """Average True Range indicator

    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
        length {int} -- Lookback period for atr indicator

    Returns:
        list -- ATR of given ohlc data
    """
    trng = true_range(data)
    res = rma(trng, length)
    return res

def rma(data, length):
    """Rolled moving average

    Arguments:
        data {list} -- List of price data
        length {int} -- Lookback period for rma

    Returns:
        list -- RMA of given data
    """
    alpha = 1 / length
    romoav = []
    for i, _ in enumerate(data):
        if i < 1:
            romoav.append(0)
        else:
            romoav.append(alpha * data[i] + (1 - alpha) * romoav[i - 1])
    return romoav

def atrpips(data, length):
    """Average True Range indicator in pips

    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
        length {int} -- Lookback period for atr indicator

    Returns:
        list -- ATR (in pips) of given ohlc data
    """
    atr_pips = []
    avtr = atr(data, length)
    close = [d[3] for d in data]

    for i, _ in enumerate(avtr):
        lclose = int(close[i])
        ldigits = 0
        if lclose == 0:
            ldigits = 1
        else:
            ldigits = int(math.log10(lclose))+1
        rdigits = 5 - ldigits
        if rdigits == 0:
            rdigits = 1
        atrpip = avtr[i] * pow(10, rdigits)
        atr_pips.append(atrpip)

    return atr_pips

def roofing_filter(data, hp_length, ss_length):
    """Python implementation of the Roofing Filter indicator created by John Ehlers

    Arguments:
        data {list} -- list of price data
        hp_length {int} -- High Pass filter length
        ss_length {int} -- period for super smoother

    Returns:
        list -- roofin filter applied data
    """
    hpf = []

    for i, _ in enumerate(data):
        if i < 2:
            hpf.append(0)
        else:
            alpha_arg = 2 * 3.14159 / (hp_length * 1.414)
            alpha1 = (math.cos(alpha_arg) + math.sin(alpha_arg) - 1) / math.cos(alpha_arg)
            hpf.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hpf[i-1] - math.pow(1-alpha1, 2)*hpf[i-2])
    return super_smoother(hpf, ss_length)

def super_smoother(data, length):
    """Python implementation of the Super Smoother indicator created by John Ehlers

    Arguments:
        data {list} -- list of price data
        length {int} -- period

    Returns:
        list -- super smoothed price data
    """
    ssf = []
    for i, _ in enumerate(data):
        if i < 2:
            ssf.append(0)
        else:
            arg = 1.414 * 3.14159 / length
            a_1 = math.exp(-arg)
            b_1 = 2 * a_1 * math.cos(4.44/float(length))
            c_2 = b_1
            c_3 = -a_1 * a_1
            c_1 = 1 - c_2 - c_3
            ssf.append(c_1 * (data[i] + data[i-1]) / 2 + c_2 * ssf[i-1] + c_3 * ssf[i-2])
    return ssf

def szladx(data, length):
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
    smoothed_true_range = []
    smoothed_directional_movement_plus = []
    smoothed_directional_movement_minus = []
    dxi = []
    szladxi = []

    for i, _ in enumerate(data):
        if i < round(lag):
            ssf.append(1)
            smoothed_true_range.append(1)
            smoothed_directional_movement_minus.append(1)
            smoothed_directional_movement_plus.append(1)
            dxi.append(1)
            szladxi.append(1)
        else:
            high = data[i][0]
            high1 = data[i-1][0]
            low = data[i][1]
            low1 = data[i-1][1]
            close1 = data[i-1][2]

            trng = max(max(high - low, abs(high - close1)), abs(low - close1))
            if high - high1 > low1 - low:
                directional_movement_plus = max(high - high1, 0)
            else:
                directional_movement_plus = 0

            if low1 - low > high - high1:
                directional_movement_minus = max(low1 - low, 0)
            else:
                directional_movement_minus = 0

            smoothed_true_range.append(smoothed_true_range[i-1] - (smoothed_true_range[i-1] / length) + trng)
            smoothed_directional_movement_plus.append(smoothed_directional_movement_plus[i-1] - (smoothed_directional_movement_plus[i-1] / length) + directional_movement_plus)
            smoothed_directional_movement_minus.append(smoothed_directional_movement_minus[i-1] - (smoothed_directional_movement_minus[i-1]/ length) + directional_movement_minus)

            di_plus = smoothed_directional_movement_plus[i] / smoothed_true_range[i] * 100
            di_minus = smoothed_directional_movement_minus[i] / smoothed_true_range[i] * 100
            dxi.append(abs(di_plus - di_minus) / (di_plus + di_minus) * 100)

            szladxi.append(dxi[i] + (dxi[i] - dxi[i-round(lag)]))

    ssf = super_smoother(szladxi, 10)
    return ssf

def true_range(data):
    """True range

    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]

    Returns:
        list -- True range of given data
    """
    trng = []
    for i, _ in enumerate(data):
        if i < 1:
            trng.append(0)
        else:
            val1 = data[i][1] - data[i][2]
            val2 = abs(data[i][1] - data[i-1][3])
            val3 = abs(data[i][2] - data[i-1][3])

            if val2 <= val1 >= val3:
                trng.append(val1)
            elif val1 <= val2 >= val3:
                trng.append(val2)
            elif val1 <= val3 >= val2:
                trng.append(val3)
    return trng

def decycler(data, hp_length):
    """Python implementation of Simple Decycler indicator created by John Ehlers

    Arguments:
        data {list} -- list of price data
        hp_length {int} -- high Pass filter length

    Returns:
        list -- Decycler applied price data
    """
    hpf = []

    for i, _ in enumerate(data):
        if i < 2:
            hpf.append(0)
        else:
            alpha_arg = 2 * 3.14159 / (hp_length * 1.414)
            alpha1 = (math.cos(alpha_arg) + math.sin(alpha_arg) - 1) / math.cos(alpha_arg)
            hpf.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hpf[i-1] - math.pow(1-alpha1, 2)*hpf[i-2])

    dec = []
    for i, _ in enumerate(data):
        dec.append(data[i] - hpf[i])

    return dec

def decycler_oscillator(data, hp_length, k_multiplier, hp_length2, k_multiplier2):
    """Python implementation of Decycler Oscillator created by John Ehlers

    Arguments:
        data {list} -- list of price data
        hp_length {int} -- high pass length for first filter
        k_multiplier {float} -- multiplier for first filter
        hp_length2 {int} -- high pass length for second filter
        k_multiplier2 {float} -- multiplier for second filter

    Returns:
        list -- Decycler oscillator data
    """
    hpf = high_pass_filter(data, hp_length, 1)

    dec = []
    for i, _ in enumerate(data):
        dec.append(data[i] - hpf[i])

    decosc = []
    dec_hp = high_pass_filter(dec, hp_length, 0.5)
    for i, _ in enumerate(data):
        decosc.append(100 * k_multiplier * dec_hp[i] / data[i])

    hpf2 = high_pass_filter(data, hp_length2, 1)

    dec2 = []
    for i, _ in enumerate(data):
        dec2.append(data[i] - hpf2[i])

    decosc2 = []
    dec_hp2 = high_pass_filter(dec2, hp_length2, 0.5)
    for i, _ in enumerate(data):
        decosc2.append(100 * k_multiplier2 * dec_hp2[i] / data[i])

    decosc_final = []
    for i, _ in enumerate(decosc):
        decosc_final.append(decosc2[i] - decosc[i])

    return decosc_final

def high_pass_filter(data, hp_length, multiplier):
    """Applies high pass filter to given data

    Arguments:
        data {list} -- list of price data
        hp_length {int} -- high pass length
        multiplier {float} -- multiplier

    Returns:
        list -- high pass filter applied price data
    """
    hpf = []

    for i, _ in enumerate(data):
        if i < 2:
            hpf.append(0)
        else:
            alpha_arg = 2 * 3.14159 / (multiplier * hp_length * 1.414)
            alpha1 = (math.cos(alpha_arg) + math.sin(alpha_arg) - 1) / math.cos(alpha_arg)
            hpf.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hpf[i-1] - math.pow(1-alpha1, 2)*hpf[i-2])

    return hpf

def damiani_volatmeter(data, vis_atr, vis_std, sed_atr, sed_std, threshold):
    """Ptyhon implementation of Damiani Volatmeter

    Args:
        data (list): List of ohlc data [open, high, low, close]
        vis_atr (int): atr length of viscosity
        vis_std (int): std length of viscosity
        sed_atr (int): atr length of sedimentation
        sed_std (int): std length of sedimentation
        threshold (float): threshold

    Returns:
        list: list of damiani volatmeter data
    """
    lag_s = 0.5

    vol = []
    vol_m = []

    close = [float(c[3]) for c in data]

    atrvis = atr(data, vis_atr)
    atrsed = atr(data, sed_atr)
    for i, _ in enumerate(data):
        if i < sed_std:
            vol.append(0)
            vol_m.append(0)
        else:
            vol.append(atrvis[i] / atrsed[i] + lag_s * (vol[i - 1] - vol[i - 3]))
            anti_thres = np.std(close[i - vis_std:i]) / np.std(close[i-sed_std:i])
            t = threshold - anti_thres
            if vol[i] > t:
                vol_m.append(1)
            else:
                vol_m.append(0)

    return vol_m

def voss(data, period, predict, bandwith):
    """Python implementation of Voss indicator created by John Ehlers

    Arguments:
        data {list} -- list of price data
        period {int} -- period
        predict {int} -- predict
        bandwith {float} -- bandwith

    Returns:
        list -- Voss indicator data
    """
    voss = []
    filt = []
    vf = []
    sumcs = []

    pi = 3.14159

    order = 3 * predict
    f1 = math.cos(2 * pi / period)
    g1 = math.cos(bandwith * 2 * pi / period)
    s1 = 1 / g1 - math.sqrt(1 / (g1 * g1) - 1)

    for i, _ in enumerate(data):
        if i <= period or i <= 5 or i <= order:
            filt.append(0)
        else:
            filt.append(0.5 *(1 - s1) * (data[i] - data[i - 2]) + f1 * (1 + s1) * filt[i - 1] - s1 * filt[i - 2])

    for i, _ in enumerate(data):
        if i <= period or i <= 5 or i <= order:
            voss.append(0)
        else:
            sumc = 0
            for count in range(order):
                sumc = sumc + ((count + 1) / float(order)) * voss[i - (order - count)]

            voss.append(((3 + order) / 2) * filt[i] - sumc)

    for i, _ in enumerate(data):
        vf.append(voss[i] - filt[i])
    return vf

def hurst_coefficient(data, length):
    dimen = []
    hurst = []
    n1 = []
    n2 = []
    n3 = []
    ll = []
    hh = []
    half_length = math.ceil(length / 2)

    for i, _ in enumerate(data):
        if i < length:
            dimen.append(0)
            hurst.append(0)
            n1.append(0)
            n2.append(0)
            n3.append(0)
            ll.append(0)
            hh.append(0)
        else:
            n3.append(round((max(data[i - length:i]) - min(data[i - length:i])) / length, 2))
            hh.append(data[i-1])
            ll.append(data[i-1])

            for t in range(half_length):
                price = data[i - t - 1]
                if price > hh[-1]:
                    hh[-1] = price
                if price < ll[-1]:
                    ll[-1] = price

            n1.append(round((hh[-1] - ll[-1]) / half_length, 2))
            hh[-1] = data[i - 1 - half_length]
            ll[-1] = data[i - 1 - half_length]

            for z in range(half_length, length):
                price = data[i - z - 1]
                if price > hh[-1]:
                    hh[-1] = price
                if price < ll[-1]:
                    ll[-1] = price

            n2.append(round((hh[-1] - ll[-1]) / half_length, 2))

            if n1[-1] > 0 and n2[-1] > 0 and n3[-1] > 0:
                dimen.append(round(0.5 * (((math.log(n1[-1] + n2[-1]) - math.log(n3[-1])) / math.log(2)) + dimen[i - 1]), 2))
            else:
                dimen.append(0)

            hurst.append(round(2 - dimen[i], 2))
    return super_smoother(hurst, 20)
