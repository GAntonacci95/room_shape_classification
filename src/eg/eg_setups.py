import os
import pandas as p
from framework.data.thread_timing import ThreadTimer
from framework.data.hashing import Hasher
from framework.experimental.simulation import room_utilities as ru
from framework.model.sample.room.room_sample import RoomSample
import framework.extension.math as me

from framework.data.io_relations.dirs import one_to_one_or_many
import numpy as np


def setups(df: p.DataFrame, params: dict) -> None:
    print("Setups generation")
    import matplotlib.pyplot as plt
    tt = ThreadTimer()
    myh = Hasher()
    n_measures_per_room = params["n_measures_per_room"]
    n_srcs_per_room_measure = params["n_srcs_per_room_measure"]
    n_mics_per_room_measure = params["n_mics_per_room_measure"]

    for room_of_setups in df.itertuples():
        tt.start()
        room_tpl = p.read_pickle(room_of_setups.input_file_path)
        georoom: RoomSample = ru.georoom_from_room_df(room_tpl)
        for measure in range(n_measures_per_room):
            pos_mics = georoom.draw_n(0.1, n_mics_per_room_measure, 1)
            for pos_mic in pos_mics:
                # FEW FRMP, INEFFICIENT
                # WC cube 3x3x3 with central src given inner_margin = 1m
                # -> mic can be at most at sqrt(3)/2=0.86m distance on the diagonal
                # -> grid step of 0.1 allows to find a sole src point on the corner
                g = georoom.grid(0.1, 1)
                np.random.shuffle(g)
                pos_srcs = []
                for p1 in g:
                    # MIGHT CONTINUE TIL INFTY, NB INITIAL CONDITIONS
                    if len(pos_srcs) < n_srcs_per_room_measure and 0.8 <= me.distance(p1, pos_mic) <= 1.5:
                        pos_srcs.append(p1)
                        break
                if len(pos_srcs) != n_srcs_per_room_measure or len(pos_mics) != n_mics_per_room_measure:
                    raise Exception("Setup: Something went wrong!")
                datum = {
                    "input_file_path": room_of_setups.input_file_path,
                    "pos_srcs": pos_srcs,
                    "pos_mic": pos_mic
                }
                p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(df.output_dir.values[0], myh.compute_hash(datum)))
                # # PRINT IN CASE OF DBG PURPOSES
                # setupex = p.DataFrame([datum])
                # praroom = ru.praroom_from_setup(setupex)
                # praroom.compute_rir()
                # praroom.plot_rir()
                # plt.show()
                # plt.clf()
                # print("DBG")
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def compute(eg3d_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        eg3d_output_dir,
        {"handler": setups, "params": params},
        "./datasets/rooms"
    )
    return


def load(eg3d_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        eg3d_output_dir,
        {"handler": setups, "params": params},
        "./datasets/rooms"
    )

def duplicateWCs(setups_output_dir: str, classes: [str]) -> None:
    from framework.data import fs_utils
    from shutil import copy
    ret = "{}WCs".format(setups_output_dir)
    fs_utils.dir_exists_or_create(ret)
    wcs = {c: 1 for c in classes}
    setups = fs_utils.get_dir_filepaths(setups_output_dir)
    for sfile in setups:
        if any(list(wcs.values())):
            setupdf = p.read_pickle(sfile)
            georoom = ru.georoom_from_setup_df(setupdf)
            if georoom.volume > 1000 and wcs[georoom.__class__.__name__] == 1:
                # copiare il setup in ret
                copy(sfile, "{}/{}".format(ret, fs_utils.get_filepath_filename(sfile)))
                # mandare a 0 il wcs
                wcs[georoom.__class__.__name__] = 0
        else:
            break
    return

def load_duplicateWCs(setups_output_dir: str) -> str:
    from pathlib import Path
    ret = "{}WCs".format(setups_output_dir)
    if not Path(ret).is_dir():
        raise Exception("Directory does not exist")
    return ret
