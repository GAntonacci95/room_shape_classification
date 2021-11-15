import pandas as p
from framework.data.thread_timing import ThreadTimer
import numpy as np
import sys

from framework.data.io_relations.dirs import one_to_one_or_many
from framework.experimental.nn.features.post_processing import nnpostprocutils


def logE(df: p.DataFrame, params: dict) -> None:
    win_len = params["win_len"]
    hop_size = int(win_len * (1 - params["olap_perc"]))
    print("logEs generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        pre_loge = p.read_pickle(t.input_file_path).output.values[0]

        datum = {
            "input_file_path": t.input_file_path,
            "output": nnpostprocutils.log_E(pre_loge, 64, 32)
        }
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute(anypreproc_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        anypreproc_output_dir,
        {"handler": logE, "params": params},
        "./datasets/nn/features/pre_processing"
    )
    return


def load(anypreproc_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        anypreproc_output_dir,
        {"handler": logE, "params": params},
        "./datasets/nn/features/pre_processing"
    )
