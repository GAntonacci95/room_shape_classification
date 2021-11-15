from eg import eg_3d, eg_setups, eg_rirs
from eg.features.post_processing import eg_stack, eg_finalize_ds
from eg.features.pre_processing import eg_rirs_adapt
from eg.features.processing import eg_gammaFB, eg_cepstrums, eg_ffts, eg_tds, eg_envelope
from eg.reports import eg_reports_rirs
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

    # remember L3(cepst_nbins(len(decimation(signal))))
    win_len, hop_size = 32, 16

    # contemplo la len(rir_longest) ~ 4*16Ksamp
    L = 117649  # nnpreprocutils.get_len_max_of_rirs_in_dir(egrirs_output_dir)
    Lpreserved = 4 * f_s  # we neglect part of the rt tail preserving the more energetic content (T60_max ~ 2.5s)
    # Lnegligible = L - Lpreserved  # 53649
    L = Lpreserved
    Lhop = int(np.ceil(L / hop_size) * hop_size)

    # <editor-fold desc="NN features pre_processing">
    # eg_rirs_adapt.compute(egrirs_output_dir, {"len_max": Lhop})
    adapted_out_dir = eg_rirs_adapt.load(egrirs_output_dir, {"len_max": Lhop})
    feature_in_dir = adapted_out_dir
    # </editor-fold>

    # <editor-fold desc="NN features processing">

    # <editor-fold desc="γ-toneFB">
    # eg_gammaFB.compute(feature_in_dir, {"f_s": f_s, "f_lo": 50, "f_hi": 2000, "n_bands": 20,
    #                                     "win_len": win_len, "hop_size": hop_size})
    eggammatones_output_dir = eg_gammaFB.load(feature_in_dir, {"f_s": f_s, "f_lo": 50, "f_hi": 2000, "n_bands": 20,
                                        "win_len": win_len, "hop_size": hop_size})
    _, L3 = eg_gammaFB.load_shape(eggammatones_output_dir)
    # </editor-fold>

    # <editor-fold desc="complex-cepstrum">
    # eg_cepstrums.compute(feature_in_dir, {"f_s": f_s, "n_bins": L3})
    egcomplexcepstrums_output_dir = eg_cepstrums.load(feature_in_dir, {"f_s": f_s, "n_bins": L3})
    # </editor-fold>

    # <editor-fold desc="fft">
    # eg_ffts.compute_fft(feature_in_dir, {"f_s": f_s, "n_bins": L3})
    egffts_output_dir = eg_ffts.load_fft(feature_in_dir, {"f_s": f_s, "n_bins": L3})
    # </editor-fold>

    # <editor-fold desc="msfft">
    # eg_ffts.compute_msfft(egffts_output_dir, {})
    egmsffts_output_dir = eg_ffts.load_msfft(egffts_output_dir, {})
    # </editor-fold>

    # <editor-fold desc="time-domain">
    # eg_tds.compute(feature_in_dir, {"f_s": f_s, "win_len": win_len, "hop_size": hop_size})
    egtds_output_dir = eg_tds.load(feature_in_dir, {"f_s": f_s, "win_len": win_len, "hop_size": hop_size})
    # </editor-fold>

    # <editor-fold desc="envelope">
    # eg_envelope.compute(feature_in_dir, {"f_s": f_s, "win_len": win_len, "hop_size": hop_size})
    egenvelopes_output_dir = eg_envelope.load(feature_in_dir, {"f_s": f_s, "win_len": win_len, "hop_size": hop_size})
    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc="NN features post_processing">

    # SHAPES of the data in each file of the dirs below
    postproc_in_dirs = [
        eggammatones_output_dir,  # (20, L3)
        egffts_output_dir,  # (1, L3)
        egmsffts_output_dir,  # (1, L3)
        egcomplexcepstrums_output_dir,  # (1, L3)
        egenvelopes_output_dir,  # (1, L3)
        egtds_output_dir  # (1, L3)
    ]

    # eg_stack.compute(postproc_in_dirs, {})
    # SHAPES of the data in each file of the dir below: (25, L3)
    feat_maps_dir = eg_stack.load(postproc_in_dirs, {})
    # eg_reports_feat_maps.create_report(feat_maps_dir,
    #                                    [
    #                                        *[r"$\gamma_{"+str(i)+"}$" for i in range(20)],
    #                                        r"$\phi$",
    #                                        r"$\phi_{ms}$",
    #                                        r"$\zeta$",
    #                                        r"$\epsilon$",
    #                                        r"$\chi$"
    #                                    ], win_len, hop_size)

    dsname = "main_rlh_ismray_rir_nocut_featAG"
    # eg_finalize_ds.compute(feat_maps_dir, dsname)

    # </editor-fold>

    return
