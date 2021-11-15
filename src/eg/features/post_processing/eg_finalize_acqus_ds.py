import pandas as p

from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many, file_room_df_chain, file_rir_df_chain
from framework.data.io_relations.files import parallelization as parall_old
from framework.experimental.simulation import room_utilities as ru


def finalize(df: p.DataFrame, params) -> p.DataFrame:
    print("Finalization")
    tt = ThreadTimer()
    input_file_path = []
    X = []
    class_label = []
    volume_m3 = []
    t60 = []

    for t in df.itertuples():
        tt.start()
        input_file_path.append(t.input_file_path)
        rir_df = file_rir_df_chain.get_rir_df_by_chaining(t.input_file_path)
        room_df = file_room_df_chain.get_room_df_by_chaining(t.input_file_path)
        X.append(p.read_pickle(t.input_file_path).output.values[0])
        class_label.append(room_df.class_name.values[0])
        georoom = ru.georoom_from_room_df(room_df)
        volume_m3.append(georoom.volume)
        t60.append(rir_df.t60.values[0][0][0])
        tt.end_first(df.shape[0])

    tt.end_total()
    tmp = p.DataFrame([input_file_path, X, class_label, volume_m3, t60]).T
    tmp.columns = ["input_file_path", 'X', "class_label", "volume_m3", "t60"]
    return tmp


def compute(anypreproc_output_dir: str, ds_name: str) -> None:
    df = one_to_one_or_many.get_i_df(anypreproc_output_dir)
    df2 = parall_old.compute_parallel(df, finalize, {})
    df2.to_pickle("./datasets/nn/final_datasets/{}.pkl".format(ds_name))
    return


def load(ds_name: str) -> any:
    return p.read_pickle("./datasets/nn/final_datasets/{}.pkl".format(ds_name))
