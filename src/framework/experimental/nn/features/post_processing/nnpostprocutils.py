import numpy as np
import sys


# TODO: REFACTOR DI STI FILE DI UTILITA', C'E' ROBA MISCHIATA


def log_E(mapee: "np.array", win_len: int, hop_size: int) -> "np.array":
    if mapee.ndim != 2:
        raise Exception("log_E: mapee.ndim = 2 expected")
    frame_start = 0
    logee = []
    while frame_start + win_len <= mapee.shape[1]:
        map_frame = mapee[:, frame_start:frame_start + win_len]
        # squaring del frame -> apparent power = |complex power| -> cumulazione lungo tempo (riga) -> energy
        energy = np.sum(np.abs(np.square(map_frame)), axis=1).reshape([mapee.shape[0], 1])
        log_e = np.log10(energy)
        logee.append(log_e)
        frame_start += hop_size
    ret = np.concatenate(logee, axis=1)
    return ret


def min_max_scaling(x: "np.array") -> "np.array":
    m, M = x.min(), x.max()
    return (x - m) / (M - m)
