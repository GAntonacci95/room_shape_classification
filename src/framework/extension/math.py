import numpy as np


def is_multiple(multiple: float, of: float) -> bool:
    multiple, of = abs(multiple), abs(of)
    if of == 0:
        raise Exception("of can't be 0")
    elif 0 < of < 1:
        factor = int(1 / of)
        posdiff = (multiple * factor) - int(multiple * factor)  # either 0+ or 1-
        tolerance = 1E-10
        return posdiff < tolerance or 1 - posdiff < tolerance   # 0+ < tol or 1-1- < tol
    else:
        return (multiple / of) - int(multiple / of) == 0


def step_index(var: float, step: float) -> int:
    if not is_multiple(var, step):
        raise Exception("var must be multiple of step!")
    return int(var / step)


def first_le_step_value(var: float, step: float) -> float:  # ARRANGE
    return int(var / step) * step


def first_ge_step_value(var: float, step: float) -> float: # OK
    tmp = first_le_step_value(var, step)
    return tmp + step if var - tmp > 0 else tmp


def draw_point(lo: float, hi: float, step: float, dim: int) -> "np.array":
    if lo > hi:
        raise Exception("lo > hi unexpected")
    ret = []
    for i in range(0, dim):
        np.random.seed()    # ok with no args
        ret.append(
            np.random.randint(
                step_index(lo, step), step_index(hi, step) + 1
            ) * step
        )
        # let n \in [0, 1[ -> int((n * (hi - lo)) + lo)
        # ret.append(int((np.random.sample() * (hi - lo)) + lo) * step)
    return np.array(ret)


def rndv_wrt_ref(ref: "np.array", dist_min: float, dist_max: float) -> "np.array":
    if ref.ndim != 1:
        raise Exception("ref must be a vector!")
    rndv = np.random.random_sample(ref.shape[0])
    # random (mic_pos - src_pos) with magnitude within [dist_min, dist_max]m
    return ref + np.random.uniform(dist_min, dist_max) * (rndv / np.linalg.norm(rndv))


def distance(v1: "np.array", v2: "np.array") -> float:
    return np.linalg.norm(v1 - v2)


def normalize_signal(signal: "np.array", dB_lvl: int = 0) -> "np.array":
    lin_atten_lvl = 10 ** (dB_lvl / 20)
    norm_factor = np.max([signal.max(), np.abs(signal.min())])
    return signal * lin_atten_lvl / norm_factor  # dB_lvl max


def adapt_along_w(v: "np.array", n_cols: int) -> "np.array":
    ret = np.zeros([1 if v.ndim == 1 else v.shape[0], n_cols], dtype=v.dtype)
    for r in range(ret.shape[0]):
        if v.ndim == 1:
            m = np.min([len(v), n_cols])
            ret[r, 0:m] = v[0:m]
        else:
            m = np.min([v.shape[1], n_cols])
            ret[r, 0:m] = v[r, 0:m]
    return ret


def pad_along_h(v: "np.array", n_rows: int) -> "np.array":
    ret = np.zeros([n_rows, v.shape[-1]], dtype=v.dtype)
    for r in range(v.shape[0]):
        ret[r, :] = v[r][0]
    return ret
