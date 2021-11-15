import pickle as pkl

import pandas
import pyroomacoustics
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt

from eg import eg_3d, eg_setups
from framework.data import fs_utils
from framework.experimental.simulation import room_utilities
from framework.model.sample.room.room_sample import RoomSample


# selfish method of RoomSample?
def e_absorption(room: RoomSample, t60: float, c: float = pyroomacoustics.constants.get('c')) -> float:
    # sole 3D case
    sab_coef = 24
    return sab_coef * np.log(10) * room.volume / (c * t60 * room.surface)


# main application
def run():
    # Bayes
    classes = ["RectangleRoomSample", "LRoomSample", "HouseRoomSample"]
    room_vars_step = 0.2
    h_min, h_max = 3, 6
    v_min, v_max, v_bandwidth = 50, 1050, 50
    n_rooms_per_band = 63

    # <editor-fold desc="rooms Generation & Loading">
    room_classes_params = {
        "R": {
            "side_min": 3, "side_max": 20, "h_min": h_min, "h_max": h_max,
            "room_vars_step": room_vars_step,
            "v_min": v_min, "v_max": v_max, "v_bandwidth": v_bandwidth,
            "n_rooms_per_band": n_rooms_per_band
        },
        "L": {
            "side_min": 6, "side_max": 20, "side_min_min": 3, "h_min": h_min, "h_max": h_max,
            "room_vars_step": room_vars_step,
            "v_min": v_min, "v_max": v_max, "v_bandwidth": v_bandwidth,
            "n_rooms_per_band": n_rooms_per_band
        },
        "H": {
            "side_min": 3, "side_max": 20, "h_min": h_min, "h_max": h_max,
            "room_vars_step": room_vars_step,
            "v_min": v_min, "v_max": v_max, "v_bandwidth": v_bandwidth,
            "n_rooms_per_band": n_rooms_per_band
        }
    }
    # eg_3d.compute_rlh(room_classes_params)
    eg3d_output_dir = eg_3d.load_rlh(room_classes_params)
    # eg_reports_rooms.create_report(eg3d_output_dir, classes)
    N = len(fs_utils.get_dir_filenames(eg3d_output_dir))
    # </editor-fold>

    # each value can either be a material descriptor or an energy absorption scalar coeff,
    # BUT a single coefficient might be far more efficient - NO. DAMN.
    wall_absorptions = [
        # un materiale con assorbimento attorno 0.04
        pra.materials_absorption_table["brick_wall_rough"],
        # un materiale con assorbimento attorno 0.02
        pra.materials_absorption_table["marble_floor"]
    ]
    n_measures_per_room = 8
    n_srcs_per_room_measure = 1
    n_mics_per_room_measure = 1
    setups_params = {
        "wall_absorptions": wall_absorptions,
        "n_measures_per_room": n_measures_per_room,
        "n_srcs_per_room_measure": n_srcs_per_room_measure,
        "n_mics_per_room_measure": n_mics_per_room_measure
    }
    # N file stanza -> S = N * n_measures_per_room * n_mics_per_room_measure
    # <editor-fold desc="rooms/setups Generation & Loading">
    # eg_setups.compute(eg3d_output_dir, setups_params)
    egsetups_output_dir = eg_setups.load(eg3d_output_dir, setups_params)
    # eg_reports_setups.create_report(egsetups_output_dir, classes)
    # </editor-fold>

    t60_desired = 0.9
    t60s_measured = []
    rir_lens = []
    files = fs_utils.get_dir_filepaths(egsetups_output_dir)
    for i in range(len(files)):
        setupfile = files[i]
        setupdf = pandas.read_pickle(setupfile)
        georoom = room_utilities.georoom_from_setup_df(setupdf)
        # if georoom.__class__.__name__ == classes[0]:
        setupdf.material_descriptor.values[0] = e_absorption(georoom, t60_desired)
        praroom = room_utilities.praroom_from_setup_df(setupdf, 16000, 3, True)
        praroom.compute_rir()
        rir_lens.append((len(praroom.rir[0][0]) // 1000) * 1000)
        t60s_measured.append(np.round(praroom.measure_rt60()[0][0], 2))
        if i > 0 and i % 2000 == 0:
            t60s, cnts1 = np.unique(t60s_measured, return_counts=True)
            lens, cnts2 = np.unique(rir_lens, return_counts=True)
            plt.bar(t60s, cnts1)
            plt.show()
            plt.clf()
            plt.bar(lens, cnts2, width=2000)
            plt.show()
            plt.clf()
    return
