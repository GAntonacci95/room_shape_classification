import pyroomacoustics as pra

# NB: eg_nn shall be imported only on workstations with GPUs
from eg import eg_3d, \
    eg_setups, \
    eg_rirs, \
    eg_noises, \
    eg_acquisitions
from eg.reports import eg_reports_rooms, \
    eg_reports_setups, \
    eg_reports_noise_acquisitions, \
    eg_reports_feat_maps

from eg.features.post_processing import eg_finalize_acqus_ds

from framework.data import fs_utils
from framework.experimental.nn.features.pre_processing import nnpreprocutils
import os
import numpy as np


# main application
def run():

    # Bayes
    classes = ["RectangleRoomSample", "LRoomSample", "HouseRoomSample"]
    room_vars_step = 0.2
    v_min, v_max, v_bandwidth = 50, 1050, 50
    n_rooms_per_band = 63

    # <editor-fold desc="rooms Generation & Loading">
    room_classes_params = {
        "R": {
            "side_min": 3, "side_max": 15, "h_min": 3, "h_max": 10,
            "room_vars_step": room_vars_step,
            "v_min": v_min, "v_max": v_max, "v_bandwidth": v_bandwidth,
            "n_rooms_per_band": n_rooms_per_band
        },
        "L": {
            "side_min": 6, "side_max": 12, "side_min_min": 3, "h_min": 3, "h_max": 10,
            "room_vars_step": room_vars_step,
            "v_min": v_min, "v_max": v_max, "v_bandwidth": v_bandwidth,
            "n_rooms_per_band": n_rooms_per_band
        },
        "H": {
            "side_min": 3, "side_max": 10, "h_min": 3, "h_max": 10,
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

    wall_absorptions = [        # each value can either be a material descriptor or an energy absorption scalar coeff
        # un materiale con assorbimento attorno 0.2 su MHF
        pra.materials_absorption_table["smooth_brickwork_10mm_pointing"],
        # un materiale con assorbimento attorno 0.02
        pra.materials_absorption_table["marble_floor"]
    ]
    n_measures_per_room = 8
    n_srcs_per_room_measure = 1
    n_mics_per_room_measure = 1
    f_s = 16000
    max_ord = 10
    setups_params = {
        "wall_absorptions": wall_absorptions,
        "n_measures_per_room": n_measures_per_room,
        "n_srcs_per_room_measure": n_srcs_per_room_measure,
        "n_mics_per_room_measure": n_mics_per_room_measure,
        "f_s": f_s,
        "max_ord": max_ord
    }

    # N file stanza -> S = N * n_measures_per_room * n_mics_per_room_measure
    # <editor-fold desc="rooms/setups Generation & Loading">
    # eg_setups.compute(eg3d_output_dir, setups_params)
    egsetups_output_dir = eg_setups.load(eg3d_output_dir, setups_params)
    # eg_reports_setups.create_report(egsetups_output_dir, classes)
    # </editor-fold>

    # S file setup -> S file rir
    # <editor-fold desc="rooms/setups/rirs Generation & Loading">
    # eg_rirs.compute(egsetups_output_dir, {})
    egrirs_output_dir = eg_rirs.load(egsetups_output_dir, {})
    # eg_reports_rirs.create_report(egrirs_output_dir)
    # </editor-fold>

    # S file setup -> S file rumore
    # <editor-fold desc="rooms/setups/white_noises Generation & Loading">
    # eg_noises.compute(egsetups_output_dir, {
    #             "f_s": f_s,
    #             "duration_s": 0.3
    #         })
    egnoises_output_dir = eg_noises.load(egsetups_output_dir, {
        "f_s": f_s,
        "duration_s": 0.3
    })
    # </editor-fold>

    # S file rumore, S file rir  -> S file mono-segnale
    # <editor-fold desc="rooms/setups/noise_acqisitions Generation & Loading">
    # eg_acquisitions.compute(egrirs_output_dir,
    #                               {"all_srcs_signals": [fs_utils.get_dir_filepaths(egnoises_output_dir)]})
    egnoiseacqus_output_dir = eg_acquisitions.load(egrirs_output_dir,
                                  {"all_srcs_signals": [fs_utils.get_dir_filepaths(egnoises_output_dir)]})
    # eg_reports_noise_acquisitions.create_report(egnoiseacqus_output_dir)
    # </editor-fold>

    # creazione dataset "finale":
    # nonostante la previsione, ci mette meno
    # ricomporre X=(30K, 25, L3), class_id=(30K, 1), volume_log=(30K, 1) è fattibile ed
    # è ora caricabile in memoria senza eccessivi problemi (2GB ca)
    # in modo da poter sfruttare appieno la GPU evitando problematiche di InputPipeline

    # # VOLUME REGRESSION OVER R/L/RL
    dsname = "main_rlh_wn_frmp0_nofdr_for_leaf"
    # eg_finalize_acqus_ds.compute(egnoiseacqus_output_dir, dsname)
    # </editor-fold>

    return