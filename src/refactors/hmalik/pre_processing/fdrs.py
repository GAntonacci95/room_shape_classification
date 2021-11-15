from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.signal import argrelmin, argrelmax
import framework.extension.math as me


def e_profile(signal, win_len: int):
    frame_start = 0
    E = []

    while frame_start + win_len <= len(signal):
        E.append(np.sum(np.square(signal[frame_start:frame_start+win_len])))
        frame_start += 1
    return np.array(E)


def butter_lowpass(cutoff, fs: int, order: int = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)
    return b, a


def butter_lowpass_filter(data, cutoff, fs: int, order: int = 5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def line(x, m, q): return m * x + q


def linear_segmentation(Ef, win_len: int):
    frame_start = 0
    Ef_lin = []

    while frame_start + win_len <= len(Ef):
        x = np.arange(frame_start, frame_start+win_len)
        mq, _ = opt.curve_fit(line, x, Ef[x])
        Ef_lin.append(line(x, *mq))
        frame_start += win_len

    ret = np.concatenate(Ef_lin)
    return ret


def first_left_der(signal):
    signal1 = np.zeros(len(signal))
    signal1[0:-1] = signal[1:]
    return signal1 - signal


def Te_filter(f, peaks_args, valleys_args, Tep = 0.05, Tev = 0.1):
    peaks_args2, valleys_args2 = -np.ones(peaks_args.shape, dtype=np.int), -np.ones(valleys_args.shape, dtype=np.int)
    np.putmask(peaks_args2, f[peaks_args] > Tep, peaks_args)
    np.putmask(valleys_args2, f[valleys_args] > Tev, valleys_args)
    peaks_args2, valleys_args2 = peaks_args2[peaks_args2 >= 0], valleys_args2[valleys_args2 >= 0]
    m = np.min([len(peaks_args2), len(valleys_args2)])
    return peaks_args2[0:m], valleys_args2[0:m]


def Td_filter(f, pv_pairs, f_s: int, Td: float = -1.25):
    xdeltas = (pv_pairs[1] - pv_pairs[0]) / f_s
    ydeltas = f[pv_pairs[1]] - f[pv_pairs[0]]
    ders = -np.log10(np.abs(ydeltas) / xdeltas)

    pv_pairs2 = -np.ones(pv_pairs.shape, dtype=np.int)
    mask = np.stack([ders < Td, ders < Td])
    np.putmask(pv_pairs2, mask, pv_pairs)
    pv_pairs2 = pv_pairs2[pv_pairs2 >= 0].reshape([2, -1])
    return pv_pairs2


# signal is a mic or band signal of a single mic
def entry_point(signal, f_s: int):
    raise Exception("TROPPO SIGNAL CORRELATED, IMPL INCOMPLETA")
    ret = []
    signal = me.normalize_signal(signal)
    E = me.normalize_signal(e_profile(signal, int(f_s / 40)))
    Ef = me.normalize_signal(butter_lowpass_filter(E, 50, f_s))

    dEf_dt = me.normalize_signal(first_left_der(Ef))
    # apart from the initial pv_pair, which raises a spurious peak
    peaks_args, valleys_args = np.array(argrelmax(dEf_dt), dtype=np.int)[:, 1:],\
                               np.array(argrelmin(dEf_dt), dtype=np.int)[:, 1:]
    peaks_args, valleys_args = Te_filter(E, peaks_args, valleys_args)
    pv_pairs = np.stack([peaks_args, valleys_args])
    pv_pairs = Td_filter(dEf_dt, pv_pairs, f_s)

    plt.figure()
    plt.plot(signal, color="yellow")
    plt.plot(E, color="green")
    plt.plot(Ef, color="red")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(dEf_dt, color="blue")
    plt.scatter(peaks_args, dEf_dt[peaks_args], marker='x', color="green")
    plt.scatter(valleys_args, dEf_dt[valleys_args], marker='x', color="red")
    plt.scatter(pv_pairs[0, :], dEf_dt[pv_pairs[0, :]], marker='o', color="yellow")
    plt.scatter(pv_pairs[1, :], dEf_dt[pv_pairs[1, :]], marker='o', color="green")
    plt.grid()
    plt.show()

    return ret
