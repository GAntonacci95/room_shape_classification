import pandas as p
from framework.data.thread_timing import ThreadTimer
import numpy as np

from framework.data.io_relations.dirs import one_to_one_or_many
import framework.extension.math as me
import sys


def adapt(df: p.DataFrame, params: dict) -> None:
    l = params["len_max"]
    print("RirAdaptations generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        rir = p.read_pickle(t.input_file_path).rir.values[0]
        linirs = [smdim for mdim in rir for smdim in mdim]
        for signal in linirs:
            adapted = me.adapt_along_w(np.array(signal), l)
            # -120dB noisy superposition to prevent logE ~ -inf
            adapted = adapted + (10E-6 * np.random.randn(1, l))

            datum = {
                "input_file_path": t.input_file_path,
                "output": adapted
            }
            # il salvataggio binario preserva il tipo
            p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute(acqus_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        acqus_output_dir,
        {"handler": adapt, "params": params},
        "./datasets/nn/features/pre_processing"
    )
    return


def load(acqus_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        acqus_output_dir,
        {"handler": adapt, "params": params},
        "./datasets/nn/features/pre_processing"
    )
