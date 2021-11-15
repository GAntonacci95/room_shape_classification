import pandas as p
from framework.data.thread_timing import ThreadTimer
import numpy as np
from framework.data.io_relations.dirs import one_to_one_or_many


def autocut(df: p.DataFrame, params: dict) -> None:
    print("Autocuts generation")
    f_s = params["f_s"]
    duration_s = params["duration_s"]
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        micarray_signal = np.array(p.read_json(t.input_file_path).output_micarray_signal.values[0])

        file_cut = [[{
            "being_at": int(f_s * duration_s),
            "end_at": len(mic_signal) - 1,
            "signal_slice": mic_signal[int(f_s * duration_s):]
        }] for mic_signal in micarray_signal]

        datum = {
            "input_file_path": t.input_file_path,
            "output_regions": file_cut
        }
        # il salvataggio binario preserva il tipo
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute(egnoiseacqus_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        egnoiseacqus_output_dir,
        {"handler": autocut, "params": params},
        "./datasets/nn/features/pre_processing"
    )
    return


def load(egnoiseacqus_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        egnoiseacqus_output_dir,
        {"handler": autocut, "params": params},
        "./datasets/nn/features/pre_processing"
    )
