import pandas as p
from scipy.io import wavfile
import framework.extension.math as me
import numpy as np


import framework.experimental.simulation.room_utilities as ru
from pyroomacoustics import Room
from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many


# NB: a contrario del caso noise in cui c'è un rumore per ogni rir,
# in questo caso non ho sufficienti segnali d'ingresso, quindi magheggio ^^
def voice_acquisitions(df: p.DataFrame, params: dict):
    print("Voice acquisitions generation")
    tt = ThreadTimer()
    # [s_i[dir_list]]
    for i in range(len(params["all_srcs_signals"])):
        np.random.shuffle(params["all_srcs_signals"][i])
    all_srcs_signals = params["all_srcs_signals"]

    for t in df.itertuples():
        tt.start()
        rir_df = p.read_pickle(t.input_file_path)
        setup_df = p.read_pickle(rir_df.input_file_path.values[0])

        sigs = [np.random.choice(all_srcs_signals[i], 1, replace=False)[0] for i in range(len(all_srcs_signals))]
        fss, signals = [], []
        for sig in sigs:
            fs, signal = wavfile.read(sig)
            fss.append(fs)
            signals.append(me.normalize_signal(signal, -3))
            if fs != fss[-1] or fs != setup_df.f_s.values[0]:
                raise Exception("VoiceAcqus: incoherent samp rates")

        # SETUP & COMPUTE_RIR
        praroom: Room = None
        try:
            praroom = ru.praroom_from_setup_df(setup_df)
            ru.praroom_set_srcs_signal(praroom, signals)
            praroom.rir = rir_df.rir.values[0]
            praroom.simulate()
            # DBG PURPOSES
            # tmp1 = praroom.mic_array.signals
            # praroom.simulate(recompute_rir=True)
            # tmp2 = praroom.mic_array.signals
            # OK = np.array_equal(tmp1, tmp2)
        except Exception as e:     # ho rispettato la notazione, se qualcosa non va debug!
            print("VoiceAcqus: Debug break on me! WTH went wrong?")

        # NB normalizzo come imposto da "eg_rirs.py"
        # per costruzione i microfoni sono separati da setup
        mic_signal = praroom.mic_array.signals[0]
        datum = {
            "input_file_path": sigs,
            "rir_file_path": t.input_file_path,
            "output": me.normalize_signal(mic_signal, -3)
        }
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])

    tt.end_total()
    return


def compute(egnoises_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        egnoises_output_dir,
        {
            "handler": voice_acquisitions,
            "params": params
        },
        "./datasets/rooms/setups"
    )
    return


def load(egnoises_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        egnoises_output_dir,
        {
            "handler": voice_acquisitions,
            "params": params
        },
        "./datasets/rooms/setups"
    )
