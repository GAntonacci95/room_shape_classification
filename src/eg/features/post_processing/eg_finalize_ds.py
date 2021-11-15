import pandas as p

from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many, file_room_df_chain, file_rir_df_chain
from framework.data.io_relations.files import parallelization as parall_old
from framework.experimental.simulation import room_utilities as ru


def finalize(df: p.DataFrame, params) -> p.DataFrame:
    print("Finalization")
    ret = p.DataFrame(columns=["input_file_path", 'X', "class_label", "volume_m3", "t60"])
    tt = ThreadTimer()

    for t in df.itertuples():
        tt.start()
        rir_df = file_rir_df_chain.get_rir_df_by_chaining(t.input_file_path)
        room_df = file_room_df_chain.get_room_df_by_chaining(t.input_file_path)
        feat_map_df = p.read_pickle(t.input_file_path)
        georoom = ru.georoom_from_room_df(room_df)
        nu_row = p.DataFrame([{
            "input_file_path": t.input_file_path,
            'X': feat_map_df.output.values[0],
            "class_label": room_df.class_name.values[0],
            "volume_m3": georoom.volume,
            "t60": rir_df.t60.values[0][0][0]
        }])

        ret = p.concat([ret, nu_row], ignore_index=True)
        tt.end_first(df.shape[0])

    tt.end_total()
    return ret


def __save(df2: p.DataFrame, ds_name: str) -> None:
    df2.to_pickle("./datasets/nn/final_datasets/{}.pkl".format(ds_name))
    return


def compute(anypreproc_output_dir: str, ds_name: str) -> None:
    df = one_to_one_or_many.get_i_df(anypreproc_output_dir)
    # extreme memory consumption => serial computation
    df2 = finalize(df, {})  # parall_old.compute_parallel(df, finalize, {})
    __save(df2, ds_name)
    return


def load(ds_name: str) -> any:
    return p.read_pickle("./datasets/nn/final_datasets/{}.pkl".format(ds_name))


def feat_normalize(in_ds_name: str, out_ds_name: str, ngammabands: int = 20) -> None:
    import numpy as np
    df: p.DataFrame = load(in_ds_name)
    x_ds_m_values, x_ds_M_values = np.inf * np.ones((df.X.values[0].shape[0], 1), dtype=np.float64),\
                                   -np.inf * np.ones((df.X.values[0].shape[0], 1), dtype=np.float64)
    for i in range(df.shape[0]):
        fmapmins, fmapmaxs = df.X.values[i].min(axis=1), df.X.values[i].max(axis=1)
        x_ds_m_values[:, 0][x_ds_m_values[:, 0] > fmapmins] = fmapmins[x_ds_m_values[:, 0] > fmapmins]
        x_ds_M_values[:, 0][x_ds_M_values[:, 0] < fmapmaxs] = fmapmaxs[x_ds_M_values[:, 0] < fmapmaxs]

    # gamma coherent scaling
    x_ds_m_values[:ngammabands, 0], x_ds_M_values[:ngammabands, 0] = \
        x_ds_m_values[:ngammabands, 0].min(), x_ds_M_values[:ngammabands, 0].max()
    # miMa map extension to match X
    x_ds_m_values, x_ds_M_values = \
        np.repeat(x_ds_m_values, df.X.values[0].shape[1], axis=1), np.repeat(x_ds_M_values, df.X.values[0].shape[1], axis=1)

    for i in range(df.shape[0]):
        tmp = df.X.values[i]
        Xmima = (tmp - x_ds_m_values) / (x_ds_M_values - x_ds_m_values)
        df.X.values[i] = Xmima
    __save(df, out_ds_name)
    return
