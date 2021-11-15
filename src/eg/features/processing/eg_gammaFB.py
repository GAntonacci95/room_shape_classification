import pandas as p

from framework.data import fs_utils
from framework.data.thread_timing import ThreadTimer
import gammatone.filters as gf
import numpy as np

from framework.data.io_relations.dirs import one_to_one_or_many
from framework.experimental.nn.features.post_processing import nnpostprocutils


def lin_to_erb(f):
    return 21.4 * np.log10(0.00437 * f + 1)


def erb_to_lin(f):
    return (np.power(10, f / 21.4) - 1) / 0.00437


def erb_centers(f_lo: float, f_hi: float, n_bands: int):
    # centri uniformi in scala erb -> logaritmici in scala lineare
    f_erb_lo = lin_to_erb(f_lo)
    f_erb_hi = lin_to_erb(f_hi)
    return np.linspace(f_erb_lo, f_erb_hi, n_bands)


def gammatone(df: p.DataFrame, params: dict) -> None:
    f_s = params["f_s"]
    f_lo = params["f_lo"]
    f_hi = params["f_hi"]
    n_bands = params["n_bands"]
    win_len = params["win_len"]
    hop_size = params["hop_size"]

    coeffs = gf.make_erb_filters(f_s, erb_to_lin(erb_centers(f_lo, f_hi, n_bands)))
    print("Gammatones generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        signal = p.read_pickle(t.input_file_path).output.values[0][0]   # (1,l) -> (l,)
        file_gfb = gf.erb_filterbank(wave=signal, coefs=coeffs)

        datum = {
            "input_file_path": t.input_file_path,
            "output": nnpostprocutils.log_E(file_gfb, win_len, hop_size)
        }

        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()

    return


def compute(anypreproc_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        anypreproc_output_dir,
        {"handler": gammatone, "params": params},
        "./datasets/nn/features/processing"
    )
    return


def load(anypreproc_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        anypreproc_output_dir,
        {"handler": gammatone, "params": params},
        "./datasets/nn/features/processing"
    )


def load_shape(loaded_dir: str) -> any:
    df = p.read_pickle(fs_utils.get_dir_filepaths(loaded_dir)[0])
    return df.output.values[0].shape
