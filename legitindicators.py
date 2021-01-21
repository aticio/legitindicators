"""legitindicators"""
import math
import statistics
import numpy as np
from scipy import stats

def sma(data, length):
    """Simple Moving Average

    Arguments:
        data {list} -- List of price data
        length {int} -- Lookback period for ema

    Returns:
        list -- SMA of given data
    """
    res = []
    for i, _ in reversed(list(enumerate(data))):
        sum = 0
        for t in range(i - length + 1, i + 1):
            sum = sum + data[t] / length
        res.insert(0, sum)
    return res

def ema(data, length):
    """Exponential Moving Average

    Arguments:
        data {list} -- List of price data
        length {int} -- Lookback period for ema

    Returns:
        list -- EMA of given data
    """
    res = []
    alpha = 2 / (length + 1)
    for i, _ in enumerate(data):
        if i < length:
            res.append(1)
        elif i == length:
            res.append(sma(data[0:i], length)[-1])
        else:
            res.append(alpha * data[i] + (1 - alpha) * res[i - 1])
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

def smoothed_atr(data, length):
    """Average True Range indicator smoothed with super smoother

    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
        length {int} -- Lookback period for atr indicator

    Returns:
        list -- Smoothed ATR of given ohlc data
    """
    trng = true_range(data)
    res = super_smoother(trng, length)
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

def atrlimit(data, length, limit, coef):
    """Average True Range implementation with a limit for using as a volatility indicator

    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
        length {int} -- Lookback period for atr indicator
        limit{int} -- average limit number to be used as threashold
        coef{float} -- threshold coefficient

    Returns:
        list -- List of ones and zeros to be used as volatility indicator
    """
    atrl = []
    th = []

    avgtr = atr(data, length)
    for i, _ in enumerate(data):
        if  i < limit:
            th.append(0)
        else:
            mean = statistics.mean(avgtr[i - limit:i + 1])
            th.append(mean * coef)

    for t, _ in enumerate(avgtr):
        if avgtr[t] >= th[t]:
            atrl.append(1)
        else:
            atrl.append(0)
    return atrl

def smoothed_atrlimit(data, length, limit, coef):
    """Smoothed Average True Range implementation with a limit for using as a volatility indicator

    Arguments:
        data {list} -- List of ohlc data [open, high, low, close]
        length {int} -- Lookback period for atr indicator
        limit{int} -- average limit number to be used as threashold
        coef{float} -- threshold coefficient

    Returns:
        list -- List of ones and zeros to be used as volatility indicator
    """
    atrl = []
    th = []

    avgtr = smoothed_atr(data, length)
    for i, _ in enumerate(data):
        if  i < limit:
            th.append(0)
        else:
            mean = statistics.mean(avgtr[i - limit:i + 1])
            th.append(mean * coef)

    for t, _ in enumerate(avgtr):
        if avgtr[t] >= th[t]:
            atrl.append(1)
        else:
            atrl.append(0)
    return atrl

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

def kaufman_er(data, length):
    ker = []
    calc = []
    for i, _ in enumerate(data):
        if i < length:
            ker.append(0)
        else:
            change = abs(data[i] - data[i - length])
            calc.append(abs(data[i] - data[i - 1]))
            volat = sum(calc[-length:])
            ker.append(change / volat)
    return ker

def ebsw(data, hp_length, ssf_length):
    pi = 3.14159
    alpha1 = (1 - math.sin(2 * pi / hp_length)) / math.cos(2 * pi / hp_length)

    hpf = []

    for i, _ in enumerate(data):
        if i < hp_length:
            hpf.append(0)
        else:
            hpf.append((0.5 * (1 + alpha1) * (data[i] - data[i - 1])) + (alpha1 * hpf[i - 1]))

    ssf = super_smoother(hpf, ssf_length)

    wave = []
    for i, _ in enumerate(data):
        if i < ssf_length:
            wave.append(0)
        else:
            w = (ssf[i] + ssf[i - 1] + ssf[i - 2]) / 3
            p = (pow(ssf[i], 2) + pow(ssf[i - 1], 2) + pow(ssf[i - 2], 2)) / 3
            if p == 0:
                wave.append(0)
            else:
                wave.append(w / math.sqrt(p))

    return wave

def cube_transform(data):
    cube = []
    for i, _ in enumerate(data):
        c = data[i]**3
        cube.append(c)
    return cube

def simple_harmonic_oscillator(data, length):
    pi = 3.14159

    sho = []
    cy = []
    cby = []
    vt = []
    vy = []
    at = []
    a = []
    t = []
    ti = []
    vp = []
    tp = []

    for i, _ in enumerate(data):
        if i < length:
            sho.append(0)
            cy.append(0)
            cby.append(0)
            vt.append(0)
            vy.append(0)
            at.append(0)
            a.append(1)
            t.append(0)
            ti.append(0)
            vp.append(0)
            tp.append(0)
        else:
            cy.append(data[i - 1])
            cby.append(data[i - 2])
            vt.append(data[i] - cy[i])
            vy.append(cy[i] - cby[i])
            at.append(vt[i] - vy[i])
            e1 = ema(at, length)[-1]
            if e1 == 0:
                a.append(1)
            else:
                a.append(e1)
            t.append(2 * pi * (math.sqrt(abs(vt[i] / a[i]))))

            if data[i] > cy[i]:
                ti.append(t[i])
            else:
                ti.append(-t[i])

            vp.append(ema(ti, length)[-1])
            e2 = ema(t, length)[-1]
            if e2 == 0:
                tp.append(1)
            else:
                tp.append(e2)

            sho.append((vp[i] / tp[i]) * 100)
    return sho

def smoothed_simple_harmonic_oscillator(data, length):
    pi = 3.14159

    ssho = []
    cy = []
    cby = []
    vt = []
    vy = []
    at = []
    a = []
    t = []
    ti = []
    vp = []
    tp = []

    for i, _ in enumerate(data):
        if i < length:
            ssho.append(0)
            cy.append(0)
            cby.append(0)
            vt.append(0)
            vy.append(0)
            at.append(0)
            a.append(1)
            t.append(0)
            ti.append(0)
            vp.append(0)
            tp.append(0)
        else:
            cy.append(data[i - 1])
            cby.append(data[i - 2])
            vt.append(data[i] - cy[i])
            vy.append(cy[i] - cby[i])
            at.append(vt[i] - vy[i])
            e1 = super_smoother(at, length)[-1]
            if e1 == 0:
                a.append(1)
            else:
                a.append(e1)
            t.append(2 * pi * (math.sqrt(abs(vt[i] / a[i]))))

            if data[i] > cy[i]:
                ti.append(t[i])
            else:
                ti.append(-t[i])

            vp.append(super_smoother(ti, length)[-1])
            e2 = super_smoother(t, length)[-1]
            if e2 == 0:
                tp.append(1)
            else:
                tp.append(e2)

            ssho.append((vp[i] / tp[i]) * 100)
    return ssho

def kama(data, length):
    ama = []
    vnoise = []

    fastnd = 0.666
    slownd = 0.0645

    for i, _ in enumerate(data):
        if i < length:
            ama.append(0)
            vnoise.append(0)
        else:
            vnoise.append(abs(data[i] - data[i - 1]))
            signal = abs(data[i] - data[i - length])
            tvnoise = vnoise[-length:]
            noise = sum(tvnoise)
            efratio = 0
            if tvnoise != 0:
                efratio = signal / noise
            smooth = math.pow(efratio * (fastnd - slownd) + slownd, 2)
            ama.append(ama[i - 1] + smooth * (data[i] - ama[i - 1]))
    return ama

def double_decycler(data, length, delay):
    dec = decycler(data, length)
    ddec = []

    for i, _ in enumerate(dec):
        if i < delay:
            ddec.append(0)
        else:
            diff = dec[i] - dec[i - delay]
            ddec.append(diff)

    return ddec

def linreg_curve(data, length):
    x = range(0, length)
    
    lr = []
    for i, _ in enumerate(data):
        if i < length:
            lr.append(0)
        else:
            y = data[i - length:i]
            slope, intercept, _, _, _ = stats.linregress(x, y)
            lr.append(intercept + slope * (length - 1))
    return lr

def linreg_slope(data, length):
    lr = linreg_curve(data, length)

    data.pop()
    data.insert(0, 0)

    lr_prev = linreg_curve(data, length)

    slope = []
    for i, _ in enumerate(lr):
        slope.append((lr[i] - lr_prev[i]) / length)

    return slope

def trendflex(data, length):
    ssf = super_smoother(data, length / 2)

    tf = []
    ms = []
    sums = []
    for i, _ in enumerate(ssf):
        if i < length:
            tf.append(0)
            ms.append(0)
            sums.append(0)
        else:
            sum = 0
            for t in range(1, length + 1):
                sum = sum + ssf[i] - ssf[i - t]
            sum = sum / length
            sums.append(sum)

            ms.append(0.04 * sums[i] * sums[i] + 0.96 * ms[i - 1])
            if ms[i] != 0:
                tf.append(round(sums[i] / math.sqrt(ms[i]), 2))

    return tf

def custom_trendflex(data, length, s_length):
    ssf = super_smoother(data, s_length)

    tf = []
    ms = []
    sums = []
    for i, _ in enumerate(ssf):
        if i < length:
            tf.append(0)
            ms.append(0)
            sums.append(0)
        else:
            sum = 0
            for t in range(1, length + 1):
                sum = sum + ssf[i] - ssf[i - t]
            sum = sum / length
            sums.append(sum)

            ms.append(0.04 * sums[i] * sums[i] + 0.96 * ms[i - 1])
            if ms[i] != 0:
                tf.append(round(sums[i] / math.sqrt(ms[i]), 2))

    return tf

def agc(data):
    real = []
    peak = []
    for i, _ in enumerate(data):
        if i < 1:
            real.append(0)
            peak.append(.0000001)
        else:
            peak.append(0.991 * peak[i - 1])
            if abs(data[i]) > peak[i]:
                peak[i] = abs(data[i])

            if peak[i] != 0:
                real.append(data[i] / peak[i])

    return real

def smoothed_ssl(data, length):
    high = []
    low = []
    close = []

    for i, _ in enumerate(data):
        high.append(data[i][0])
        low.append(data[i][1])
        close.append(data[i][2])

    s_high = super_smoother(high, length)
    s_low = super_smoother(low, length)

    hlv = []
    ssl_up = []
    ssl_down = []

    for i, _ in enumerate(close):
        if i < 1:
            hlv.append(.00000001)
            ssl_up.append(0)
            ssl_down.append(0)
        else:
            if close[i] > s_high[i]:
                hlv.append(1)
            else:
                if close[i] < s_low[i]:
                    hlv.append(-1)
                else:
                    hlv.append(hlv[i - 1])

            if hlv[i] < 0:
                ssl_down.append(s_high[i])
                ssl_up.append(s_low[i])
            else:
                ssl_down.append(s_low[i])
                ssl_up.append(s_high[i])

    return ssl_up, ssl_down

def bollinger_bands_pb(data, length, stdd):
    dev = []
    upper = []
    lower = []
    bbr = []
    basis = sma(data, length)

    for i, _ in enumerate(data):
        if i < length:
            dev.append(0)
            upper.append(0)
            lower.append(0)
            bbr.append(0)
        else:
            tmp = data[i - length:i]
            dev.append(stdd * statistics.stdev(tmp))
            upper.append(basis[i] + dev[i])
            lower.append(basis[i] - dev[i])
            bbr.append((data[i] - lower[i]) / (upper[i] - lower[i]))
    return bbr

def noise_elemination_tech(data, length):
    net = []
    denom = []

    for i, _ in enumerate(data):
        denom.append(length * (length - 1) / 2)
        n = 0

        for i in range(1, length - 1):
            for k in range(0, i - 1):
                sign = 0
                if (data[i] - data[k]) == 0:
                    sign = 0
                elif (data[i] - data[k]) > 0:
                    sign = 1
                elif (data[i] - data[k]) < 0:
                    sign = -1

                n = n - sign

        net.append(n / denom[i])
    return net
