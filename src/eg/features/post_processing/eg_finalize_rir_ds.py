import pandas as p

from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many, file_room_df_chain
from framework.data.io_relations.files import parallelization as parall_old
from framework.experimental.simulation import room_utilities as ru
import framework.extension.math as me


def finalize(df: p.DataFrame, params) -> p.DataFrame:
    print("Finalization")
    tt = ThreadTimer()
    ret = p.DataFrame(columns=["input_file_path", 'X', "class_label", "volume_m3"])

    for t in df.itertuples():
        tt.start()
        room_df = file_room_df_chain.get_room_df_by_chaining(t.input_file_path)
        georoom = ru.georoom_from_room_df(room_df)
        rir_df = p.read_pickle(t.input_file_path)
        rir = rir_df.rir.values[0]
        linirs = [smdim for mdim in rir for smdim in mdim]
        t60 = rir_df.t60.values[0]
        lint60s = [smdim for mdim in t60 for smdim in mdim]
        for ir, t60 in zip(linirs, lint60s):
            nu_row = p.DataFrame([{
                "input_file_path": t.input_file_path,
                'X': ir,
                "class_label": room_df.class_name.values[0],
                "volume_m3": georoom.volume,
                "t60": t60
            }])

            ret = p.concat([ret, nu_row], ignore_index=True)
        tt.end_first(df.shape[0])

    tt.end_total()
    return ret


def compute(anypreproc_output_dir: str, ds_name: str) -> None:
    df = one_to_one_or_many.get_i_df(anypreproc_output_dir)
    df2 = finalize(df, {})
    df2.to_pickle("./datasets/nn/final_datasets/{}.pkl".format(ds_name))
    return


def load(ds_name: str) -> any:
    return p.read_pickle("./datasets/nn/final_datasets/{}.pkl".format(ds_name))
