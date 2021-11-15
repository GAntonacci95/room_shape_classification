import pandas as p
from scipy.io import wavfile
import framework.extension.math as me
import numpy as np


import framework.experimental.simulation.room_utilities as ru
from pyroomacoustics import Room
from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many


def noise_acquisitions(df: p.DataFrame, params: dict):
    print("Noise acquisitions generation")
    tt = ThreadTimer()
    rir_files_paths = params["rir_files_paths"]
    # TODO: anche le rir andrebbero ripartite, ma dovrei modificare "one_to_one_or_many.py"
    #  per fargli prendere input_dirs: List[str] e gestire tutta l'associazione (1noise,1rir)

    for t in df.itertuples():
        tt.start()
        # trascuro che e segnale e rir dovrebbero venire dallo stesso setup,
        # sarebbe inutile ed inefficiente
        fs, signal = wavfile.read(t.input_file_path)
        assoc_rir_file = rir_files_paths[t.Index]

        rir_df: p.DataFrame = p.read_pickle(assoc_rir_file)
        setup_df = p.read_pickle(rir_df.input_file_path.values[0])
        if fs != setup_df.f_s.values[0] or signal.dtype != np.float:
            raise Exception("Different fs or wrong dtype!")

        # SETUP & COMPUTE_RIR
        praroom: Room = None
        try:
            praroom = ru.praroom_from_setup_df(setup_df, signal)
            praroom.rir = rir_df.rir.values[0]
            praroom.simulate()
            # DBG PURPOSES
            # tmp1 = praroom.mic_array.signals
            # praroom.simulate(recompute_rir=True)
            # tmp2 = praroom.mic_array.signals
            # OK = np.array_equal(tmp1, tmp2)
        except Exception as e:     # ho rispettato la notazione, se qualcosa non va debug!
            print("Debug break on me! WTH went wrong?")

        # NB normalizzo come imposto da "eg_rirs.py"
        for i in range(len(praroom.mic_array.signals)):
            mic_signal = praroom.mic_array.signals[i]
            datum = {
                "input_file_path": t.input_file_path,
                "rir_file_path": assoc_rir_file,
                "output": me.normalize_signal(mic_signal, -3)
            }
            # SALVATAGGIO, l'hashing qui lo trascuro seppur 1 rir, 1 signal -> N mic_signals
            p.DataFrame([datum]).to_pickle("{}/{}_{}.pkl".format(t.output_dir, t.input_file_name, i))

        tt.end_first(df.shape[0])

    tt.end_total()
    return


def compute(egnoises_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        egnoises_output_dir,
        {
            "handler": noise_acquisitions,
            "params": params
        },
        "./datasets/rooms/setups"
    )
    return


def load(egnoises_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        egnoises_output_dir,
        {
            "handler": noise_acquisitions,
            "params": params
        },
        "./datasets/rooms/setups"
    )
