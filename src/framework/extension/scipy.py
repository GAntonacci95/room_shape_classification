def butter_lp_coeffs(cutoff: float, fs: int, order: int = 3):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, "low")
    return b, a


def lp_filter(data, cutoff: float, fs: int, order: int = 6):
    if data.ndim != 1:
        raise Exception("lp: data.ndim = 1 expected")
    from scipy.signal import filtfilt
    b, a = butter_lp_coeffs(cutoff, fs, order=order // 2)
    y = filtfilt(b, a, data)
    return y


def butter_hp_coeffs(cutoff: float, fs: int, order: int = 3):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, "high")
    return b, a


def hp_filter(data, cutoff: float, fs: int, order: int = 6):
    if data.ndim != 1:
        raise Exception("hp: data.ndim = 1 expected")
    from scipy.signal import filtfilt
    b, a = butter_hp_coeffs(cutoff, fs, order=order // 2)
    y = filtfilt(b, a, data)
    return y


# 1 or 2D windowing applied row-wise
def windowing_along_w(signal, win_type: str, win_len: int, win_olap: int, padding: bool = False):
    import numpy as np
    import framework.extension.math as me
    from scipy.signal.windows import get_window

    signal = np.array(signal)
    assert signal.ndim <= 2
    if signal.ndim == 1:
        signal = signal.reshape([1, len(signal)])
    n_rows = signal.shape[0]

    win_type = "boxcar" if win_type == "rect" else win_type
    win_hop = win_len - win_olap
    window = get_window(win_type, win_len)\
        .reshape([1, win_len])\
        .repeat(signal.shape[0], axis=0)

    frame_start = 0
    frames = []
    signal = me.adapt_along_w(signal, int(np.ceil(signal.shape[1] / win_len) * win_len) + win_hop)
    while frame_start + win_len <= signal.shape[1]:
        frames.append(signal[:, frame_start:frame_start+win_len] * window)
        frame_start += win_hop
    if not padding:
        for i in reversed(range(len(frames))):
            if np.all(frames[i] == 0):
                frames.pop(i)
            elif np.any(frames[i] == 0):
                frames[i] = frames[i][frames[i] != 0].reshape([n_rows, -1])
    return frames
