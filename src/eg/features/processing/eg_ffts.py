import pandas as p
from framework.data.thread_timing import ThreadTimer
from scipy.fft import fft as scifft
import numpy as np

from framework.data.io_relations.dirs import one_to_one_or_many
import framework.extension.math as me
from scipy.signal import decimate

# to store ffts and complex data, I must use a binary output format


def fft(df: p.DataFrame, params: dict) -> None:
    f_s = params["f_s"]
    n_bins = params["n_bins"]
    print("FFTs generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        signal = p.read_pickle(t.input_file_path).output.values[0][0]
        # automatic LP to 500Hz (16:1 factor)  -> anti-aliasing filtering -> downsampled signal
        signal = decimate(signal, int(f_s / 1000))
        if len(signal) - n_bins < 100:
            signal = me.adapt_along_w(signal, n_bins)[0]
        else:
            raise Exception("FFTs: warning! sure about win_len & hop_size?")
        file_fft = np.abs(me.adapt_along_w(scifft(signal, 2 * n_bins), n_bins))

        datum = {
            "input_file_path": t.input_file_path,
            "output": file_fft
        }

        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute_fft(anypreproc_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        anypreproc_output_dir,
        {"handler": fft, "params": params},
        "./datasets/nn/features/processing"
    )
    return


def load_fft(anypreproc_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        anypreproc_output_dir,
        {"handler": fft, "params": params},
        "./datasets/nn/features/processing"
    )


def msfft(df: p.DataFrame, params: dict) -> None:
    print("MagSortedFFTs generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        signal_fft = p.read_pickle(t.input_file_path).output.values[0]  # already magnitude

        file_msfft = np.sort(signal_fft)

        datum = {
            "input_file_path": t.input_file_path,
            "output": file_msfft
        }
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute_msfft(anypreproc_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        anypreproc_output_dir,
        {"handler": msfft, "params": params},
        "./datasets/nn/features/processing"
    )
    return


def load_msfft(anypreproc_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        anypreproc_output_dir,
        {"handler": msfft, "params": params},
        "./datasets/nn/features/processing"
    )
