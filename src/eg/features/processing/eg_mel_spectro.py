import pandas as p
from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many
import librosa


def mel_spectrogram(df: p.DataFrame, params: dict) -> None:
    f_s = params["f_s"]
    win_len = params["win_len"]
    hop_size = int(win_len * (1 - params["olap_perc"]))
    print("MelScaled-Spectrogram generation")
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        region = p.read_pickle(t.input_file_path).output.values[0]

        mel_spectro = librosa.feature.melspectrogram(region[0], sr=int(f_s/2),
                                                     hop_length=hop_size, win_length=win_len)

        datum = {
            "input_file_path": t.input_file_path,
            # TODO: NB STANDARDIZZARE UNA VOLTA PER TUTTE STI NOMI
            "output_map": mel_spectro
        }
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))
        tt.end_first(df.shape[0])
    tt.end_total()

    return


def compute(anypreproc_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        anypreproc_output_dir,
        {"handler": mel_spectrogram, "params": params},
        "./datasets/nn/features/processing"
    )
    return


def load(anypreproc_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        anypreproc_output_dir,
        {"handler": mel_spectrogram, "params": params},
        "./datasets/nn/features/processing"
    )
