import pandas as p
import numpy as np
from scipy.io import wavfile
import framework.extension.math as me
from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many


def white_noises(df: p.DataFrame, params: dict) -> None:
    print("White noises generation")
    tt = ThreadTimer()
    f_s = params["f_s"]
    duration_s = params["duration_s"]
    for t in df.itertuples():
        tt.start()
        wnoise = np.random.randn(int(f_s * duration_s))
        wnoise = me.normalize_signal(wnoise, -3)
        wavfile.write("{}/{}.wav".format(t.output_dir, t.input_file_name), f_s, wnoise)
        tt.end_first(df.shape[0])

    tt.end_total()
    return

# TODO: QUI NON DOVREI AVERE I SETUP IN INPUT, MA SOLO IL NUMERO DI RUMORI
#  CHE VORREI FOSSERO GENERATI, VABE' CORREGGERO'...
def compute(egsetups_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        egsetups_output_dir,
        {
            "handler": white_noises,
            "params": params
        },
        "./datasets/rooms/setups"
    )
    return


def load(egsetups_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        egsetups_output_dir,
        {
            "handler": white_noises,
            "params": params
        },
        "./datasets/rooms/setups"
    )
