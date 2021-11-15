import pandas as p

from framework.extension.scipy import lp_filter
from framework.data.thread_timing import ThreadTimer
import obspy.signal.filter as ob

from framework.data.io_relations.dirs import one_to_one_or_many
import framework.extension.math as me
from framework.experimental.nn.features.pre_processing import nnpreprocutils
from framework.experimental.nn.features.post_processing import nnpostprocutils


def envelope(df: p.DataFrame, params: dict) -> None:
    f_s = params["f_s"]
    win_len = params["win_len"]
    hop_size = params["hop_size"]
    print("ENVs generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        signal = p.read_pickle(t.input_file_path).output.values[0][0]
        signal = lp_filter(signal, 4000, f_s)
        file_env = ob.envelope(signal)

        datum = {
            "input_file_path": t.input_file_path,
            "output": nnpostprocutils.log_E(file_env.reshape((1, -1)), win_len, hop_size)
        }
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute(anypreproc_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        anypreproc_output_dir,
        {"handler": envelope, "params": params},
        "./datasets/nn/features/processing"
    )
    return


def load(anypreproc_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        anypreproc_output_dir,
        {"handler": envelope, "params": params},
        "./datasets/nn/features/processing"
    )
