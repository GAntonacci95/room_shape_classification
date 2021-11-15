from eg import eg_3d, eg_setups, eg_rirs, eg_noises, eg_acquisitions
from eg.features.post_processing import eg_finalize_rir_ds, eg_finalize_acqus_ds, eg_stack, eg_finalize_ds
from eg.features.pre_processing import eg_adapt
from eg.features.processing import eg_envelope, eg_tds, eg_ffts, eg_cepstrums, eg_gammaFB
from eg.reports import eg_reports_acquisitions, eg_reports_feat_maps, eg_report_ds
from framework.data import fs_utils
from framework.experimental.nn.features.pre_processing import nnpreprocutils
import numpy as np


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
    # </editor-fold>

    n_measures_per_room = 8
    n_srcs_per_room_measure = 1
    n_mics_per_room_measure = 1
    setups_params = {
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

    f_s = 16000
    max_ord = 10
    use_ray = True
    # # each value can either be a material descriptor or an energy absorption scalar coeff
    # wall_absorptions = [
    #     # un materiale con assorbimento attorno 0.04
    #     pra.materials_absorption_table["brick_wall_rough"],
    #     # un materiale con assorbimento attorno 0.02
    #     pra.materials_absorption_table["marble_floor"]
    # ]
    rirs_params = {
        "v_range": {"min": v_min, "max": v_max},
        "t60_range": {"min": 0.5, "max": 2.5},
        "f_s": f_s,
        "max_ord": max_ord,
        "use_ray": use_ray
    }
    # S file setup -> S file rir
    # <editor-fold desc="rooms/setups/rirs Generation & Loading">
    # eg_rirs.compute(egsetups_output_dir, rirs_params)
    egrirs_output_dir = eg_rirs.load(egsetups_output_dir, rirs_params)
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
                                                   {"all_srcs_signals": [
                                                       fs_utils.get_dir_filepaths(egnoises_output_dir)]})
    # eg_reports_acquisitions.create_report(egnoiseacqus_output_dir, "sensed_o10_ray_wn")
    # </editor-fold>

    # remember L3(cepst_nbins(len(decimation(signal))))
    win_len, hop_size = 32, 16

    # contemplo la len(rir_longest)
    Lnegligible = 53649
    L = 122448  # nnpreprocutils.get_len_max_of_signals_in_dir(egnoiseacqus_output_dir)
    L -= Lnegligible
    Lhop = int(np.ceil(L / hop_size) * hop_size)

    # <editor-fold desc="NN features pre_processing">
    # eg_adapt.compute(egnoiseacqus_output_dir, {"len_max": Lhop})
    adapted_out_dir = eg_adapt.load(egnoiseacqus_output_dir, {"len_max": Lhop})
    feature_in_dir = adapted_out_dir
    # </editor-fold>

    # <editor-fold desc="NN features processing">

    # <editor-fold desc="γ-toneFB">
    gamma_params = {"f_s": f_s, "f_lo": 20, "f_hi": 2000, "n_bands": 30, "win_len": win_len, "hop_size": hop_size}
    # eg_gammaFB.compute(feature_in_dir, gamma_params)
    eggammatones_output_dir = eg_gammaFB.load(feature_in_dir, gamma_params)
    _, L3 = eg_gammaFB.load_shape(eggammatones_output_dir)
    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc="NN features post_processing">

    # SHAPES of the data in each file of the dir below: (30, L3)
    feat_maps_dir = eggammatones_output_dir
    # eg_reports_feat_maps.create_report(feat_maps_dir,
    #                                    [
    #                                        *[r"$\gamma_{"+str(i)+"}$" for i in range(30)]
    #                                    ], win_len, hop_size)

    dsname = "main_rlh_ismray_wn_nofdr_featAGAlt"
    dsnamenormal = "{}_normal".format(dsname)
    # eg_finalize_ds.compute(feat_maps_dir, dsname)
    # eg_finalize_ds.feat_normalize(dsname, dsnamenormal, 30)
    # eg_report_ds.create_report(dsnamenormal,
    #                                    [
    #                                        *[r"$\gamma_{"+str(i)+"}$" for i in range(30)]
    #                                    ], win_len, hop_size, "wn_featAGAlt_normal")

    # </editor-fold>

    return
