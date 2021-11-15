import pandas as p
from framework.data.thread_timing import ThreadTimer
import acoustics.cepstrum as ac

from framework.data.io_relations.dirs import one_to_one_or_many
import framework.extension.math as me
from scipy.signal import decimate


def complex_cepstrum(df: p.DataFrame, params: dict) -> None:
    import matplotlib.pyplot as plt
    f_s = params["f_s"]
    n_bins = params["n_bins"]
    print("ComplexCepstums generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        signal = p.read_pickle(t.input_file_path).output.values[0][0]
        # automatic LP to 1KHz (16:1 factor) -> anti-aliasing filtering -> downsampled signal
        signal = decimate(signal, int(f_s / 1000))
        if len(signal) - n_bins < 100:
            signal = me.adapt_along_w(signal, n_bins)[0]
        else:
            raise Exception("ComplexCepstrums: warning! sure about win_len & hop_size?")
        ceps, _ = ac.complex_cepstrum(signal, n_bins)
        file_mfccs = me.adapt_along_w(ceps, n_bins)     # or .reshape((1, -1))

        datum = {
            "input_file_path": t.input_file_path,
            "output": file_mfccs
        }
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()

    return

# TODO: real or MFCepst?


def compute(anypreproc_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        anypreproc_output_dir,
        {"handler": complex_cepstrum, "params": params},
        "./datasets/nn/features/processing"
    )
    return


def load(anypreproc_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        anypreproc_output_dir,
        {"handler": complex_cepstrum, "params": params},
        "./datasets/nn/features/processing"
    )
