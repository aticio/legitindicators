# flake8: noqa
import math
import statistics
import numpy as np
from scipy import stats


def sma(data, length):
    """Simple Moving Average

    Parameters:
        data (list): List of price data
        length (int): Lookback period for ema

    Returns:
        res (list): SMA of given data
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

    Parameters:
        data (list): List of price data
        length (int): Lookback period for ema

    Returns:
        res (list): EMA of given data
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

    Parameters:
        data (list): List of ohlc data [open, high, low, close]
        length (int): Lookback period for atr indicator

    Returns:
        res (list): ATR of given ohlc data
    """
    trng = true_range(data)
    res = rma(trng, length)
    return res


def smoothed_atr(data, length):
    """Average True Range indicator smoothed with super smoother

    Parameters:
        data (list): List of ohlc data [open, high, low, close]
        length (int): Lookback period for atr indicator

    Returns:
        res (list): Smoothed ATR of given ohlc data
    """
    trng = true_range(data)
    res = super_smoother(trng, length)
    return res


def rma(data, length):
    """Rolled moving average

    Parameters:
        data (list): List of price data
        length (int): Lookback period for rma

    Returns:
        romoav (list): RMA of given data
    """
    alpha = 1 / length
    romoav = []
    for i, _ in enumerate(data):
        if i < 1:
            romoav.append(0)
        else:
            romoav.append(alpha * data[i] + (1 - alpha) * romoav[i - 1])
    return romoav


def atrlimit(data, length, ss_length):
    """Average True Range implementation with a limit for using as a volatility indicator

    Parameters:
        data (list): List of ohlc data [open, high, low, close]
        length (int): Lookback period for atr indicator
        ss_length (int): Super Smoother length to be used as threashold

    Returns:
        atrl (list): List of ones and zeros to be used as volatility indicator
    """
    atrl = []

    avgtr = atr(data, length)
    s_avgtr = super_smoother(avgtr, ss_length)

    for t, _ in enumerate(avgtr):
        if avgtr[t] >= s_avgtr[t]:
            atrl.append(1)
        else:
            atrl.append(0)
    return atrl


def smoothed_atrlimit(data, length, limit, coef):
    """Smoothed Average True Range implementation with a limit for using as a volatility indicator

    Parameters:
        data (list): List of ohlc data [open, high, low, close]
        length (int): Lookback period for atr indicator
        limit (int):Aaverage limit number to be used as threashold
        coef (float): Threshold coefficient

    Returns:
        atrl (list): List of ones and zeros to be used as volatility indicator
    """
    atrl = []
    th = []

    avgtr = smoothed_atr(data, length)
    for i, _ in enumerate(data):
        if i < limit:
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

    Parameters:
        data (list): List of price data
        hp_length (int): High Pass filter length
        ss_length (int): Period for super smoother

    Returns:
        ss (list): Roofing Filter applied data
    """
    hpf = []

    for i, _ in enumerate(data):
        if i < 2:
            hpf.append(0)
        else:
            alpha_arg = 2 * 3.14159 / (hp_length * 1.414)
            alpha1 = (math.cos(alpha_arg) + math.sin(alpha_arg) - 1) / math.cos(alpha_arg)
            hpf.append(math.pow(1.0-alpha1/2.0, 2)*(data[i]-2*data[i-1]+data[i-2]) + 2*(1-alpha1)*hpf[i-1] - math.pow(1-alpha1, 2)*hpf[i-2])
    ss = super_smoother(hpf, ss_length)
    return ss


def super_smoother(data, length):
    """Python implementation of the Super Smoother indicator created by John Ehlers

    Parameters:
        data (list): List of price data
        length (int): Period

    Returns:
        ssf (list): Super smoothed price data
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

    Parameters:
        data (list): List data consists of [high, low, close]
        length (int): Lookback period of adx
        treshold (int): Threshold line for adx

    Returns:
        ssf (list): List of low lag adx indicator data
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
            smoothed_directional_movement_plus.append(smoothed_directional_movement_plus[i-1] - (smoothed_directional_movement_plus[i - 1] / length) + directional_movement_plus)
            smoothed_directional_movement_minus.append(smoothed_directional_movement_minus[i-1] - (smoothed_directional_movement_minus[i - 1] / length) + directional_movement_minus)

            di_plus = smoothed_directional_movement_plus[i] / smoothed_true_range[i] * 100
            di_minus = smoothed_directional_movement_minus[i] / smoothed_true_range[i] * 100
            dxi.append(abs(di_plus - di_minus) / (di_plus + di_minus) * 100)

            szladxi.append(dxi[i] + (dxi[i] - dxi[i-round(lag)]))

    ssf = super_smoother(szladxi, 10)
    return ssf


def true_range(data):
    """True range

    Parameters:
        data (list): List of ohlc data [open, high, low, close]

    Returns:
        trng (list): True range of given data
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

    Parameters:
        data (list): List of price data
        hp_length (int): High Pass filter length

    Returns:
        dec (list): Decycler applied price data
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

    Parameters:
        data (list): List of price data
        hp_length (int): High pass length for first filter
        k_multiplier (float): Multiplier for first filter
        hp_length2 (int): High pass length for second filter
        k_multiplier2 (float): Multiplier for second filter

    Returns:
        decosc_final (list): Decycler Oscillator data
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

    Parameters:
        data (list): List of price data
        hp_length (int): High pass length
        multiplier (float): Multiplier

    Returns:
        hpf (list): High pass filter applied price data
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
        threshold (float): Threshold

    Returns:
        vol_m (list): List of damiani volatmeter data
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

    Parameters:
        data (list): List of price data
        period (int): Period
        predict (int): Predict
        bandwith (float): Bandwith

    Returns:
        vf (list): Voss indicator data
    """
    voss = []
    filt = []
    vf = []

    pi = 3.14159

    order = 3 * predict
    f1 = math.cos(2 * pi / period)
    g1 = math.cos(bandwith * 2 * pi / period)
    s1 = 1 / g1 - math.sqrt(1 / (g1 * g1) - 1)

    for i, _ in enumerate(data):
        if i <= period or i <= 5 or i <= order:
            filt.append(0)
        else:
            filt.append(0.5 * (1 - s1) * (data[i] - data[i - 2]) + f1 * (1 + s1) * filt[i - 1] - s1 * filt[i - 2])

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
    """Hurst coefficient

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        ss (list): Hurst coefficient value of given data
    """
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
    ss = super_smoother(hurst, 20)
    return ss


def kaufman_er(data, length):
    """Kaufman Efficiency Ratio

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        ker (list): Kaufman Efficiency Ratio for given data
    """
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
    """Python implementation of Even Better Sine Wave indicator created by John Ehlers

    Parameters:
        data (list): List of price data
        hp_length (int): Period
        ssf_length (int): Predict

    Returns:
        wave (list): Even Better Sine Wave indicator data
    """
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
    """Python implementation of Cube Transform created by John Ehlers

    Parameters:
        data (list): List of price data

    Returns:
        cube (list): Cube Transform data
    """
    cube = []
    for i, _ in enumerate(data):
        c = data[i]**3
        cube.append(c)
    return cube


def simple_harmonic_oscillator(data, length):
    """Simple Harmonic Oscillator

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        sho (list): Simple Harmonic Oscillator indicator data
    """
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
    """Smoothed Simple Harmonic Oscillator - Super Smoother applied

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        ssho (list): Smoothed Simple Harmonic Oscillator indicator data
    """
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

            vp.append(super_smoother(ti, length)[-1])
            e2 = super_smoother(t, length)[-1]
            if e2 == 0:
                tp.append(1)
            else:
                tp.append(e2)

            ssho.append((vp[i] / tp[i]) * 100)
    return ssho


def kama(data, length):
    """Kaufman Adaptive Moving Average

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        ama (list): Kaufman Adaptive Moving Average indicator data
    """
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


def linreg_curve(data, length):
    """Linear Regression Curve

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        lr (list): Linear Regression Curve indicator data
    """
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
    """Linear Regression Slope

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        slope (list): Linear Regression Slope indicator data
    """
    lr = linreg_curve(data, length)

    data.pop()
    data.insert(0, 0)

    lr_prev = linreg_curve(data, length)

    slope = []
    for i, _ in enumerate(lr):
        slope.append((lr[i] - lr_prev[i]) / length)

    return slope


def trendflex(data, length):
    """Python implementation of TrendFlex indicator created by John Ehlers

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        tf (list): TrendFlex indicator data
    """
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
    """Python implementation of TrendFlex indicator with customizable super smoother length, created by John Ehlers

    Parameters:
        data (list): List of price data
        length (int): Lookback period
        s_length (int): SuperSmoother lookback period

    Returns:
        tf (list): TrendFlex indicator data
    """
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
    """Python implementation of Automatic Gain Control created by John Ehlers

    Parameters:
        data (list): List of price data

    Returns:
        real (list): Automatic Gain Control data
    """
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
    """Super Smoother Applied SSL indicator

    Parameters:
        data (list): List of price data
        length (int): Lookback period

    Returns:
        ssl_up (list): SSL up data of indicator
        ssl_down (list): SSL down data of indicator
    """
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
    """Bollinger Bands %B indicator

    Parameters:
        data (list): List of price data
        length (int): Lookback period
        stdd (float): standart deviation multiplier

    Returns:
        bbr (list): Bollinger Bands %B indicator data
    """
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


def volume_heat(data, ma_length):
    """Volume Heat - A custom volume indicator

    Parameters:
        data (list): List of volume data
        ma_length (int): Moving average lookback period

    Returns:
        vh (list): Volume Heat indicator data
    """
    vh = []

    for i, _ in enumerate(data):
        if i < ma_length:
            vh.append(0)
        else:
            mean = statistics.mean(data[i - ma_length:i + 1])
            std = statistics.stdev(data[i - ma_length:i + 1])

            if (data[i] - mean) / std > 1:
                vh.append(1)
            else:
                vh.append(0)
    return vh


def double_super_smoother(data, ssf_length1, ssf_length2):
    """Two Super Smoother indicators for checking line crosses

    Parameters:
        data (list): List of price data
        ssf_length1 (int): Super smoother lookback period for first line
        ssf_length2 (int): Super smoother lookback period for second line

    Returns:
        dssf (list): Double super smoother indicator data
    """
    ssf1 = super_smoother(data, ssf_length1)
    ssf2 = super_smoother(data, ssf_length2)

    dssf = []

    for i, _ in enumerate(ssf1):
        dssf.append(ssf1[i] - ssf2[i])

    return dssf


def ema_trailing(data, ema_length, trailing_stop_percent):
    """A trailing stop implementation using Exponantial Moving Average

    Parameters:
        data (list): List of price data
        ema_length (int): Exponantial Moving Average lookback period
        trailing_stop_percent (float): A multiplier needed to determine the distance of trailing stop from price. 

    Returns:
        emavg (list): Exponantial Moving Average indicator data
        ts (list): Trailing stop
    """
    ts = []
    emavg = ema(data, ema_length)

    for i, e in enumerate(emavg):
        if i < 3:
            ts.append(0)
        else:
            if emavg[i] > ts[i - 1]:
                ts_temp = emavg[i] - (emavg[i] * trailing_stop_percent)
                if emavg[i - 1] < ts[i - 2]:
                    ts.append(ts_temp)
                else:
                    if ts_temp > ts[i - 1]:
                        ts.append(ts_temp)
                    else:
                        ts.append(ts[i - 1])
            else:
                ts_temp = emavg[i] + (emavg[i] * trailing_stop_percent)
                if emavg[i - 1] > ts[i - 2]:
                    ts.append(ts_temp)
                else:
                    if ts_temp < ts[i - 1]:
                        ts.append(ts_temp)
                    else:
                        ts.append(ts[i - 1])
    return (emavg, ts)


def momentum_normalized(data, length):
    """Normalized Momentum Indicator - Cube Transform applied

    Parameters:
        data (list): List of price data
        length (int): Momentum indicator lookback period

    Returns:
        cube_anm (list): Normalized Momentum indicator data
    """
    momentum = []
    norm_mom = []

    for i, _ in enumerate(data):
        if i < length:
            momentum.append(0)
        else:
            momentum.append(data[i] - data[i - length])

    smoothed_momentum = super_smoother(momentum, 10)

    for i, _ in enumerate(smoothed_momentum):
        if i > 1:
            norm_mom.append(smoothed_momentum[i] - smoothed_momentum[i - 1])
        else:
            norm_mom.append(0)

    agc_norm_mom = agc(norm_mom)
    cube_anm = cube_transform(agc_norm_mom)

    return cube_anm
