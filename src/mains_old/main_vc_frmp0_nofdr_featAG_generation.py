import pyroomacoustics as pra

# NB: eg_nn shall be imported only on workstations with GPUs
from eg import eg_3d, \
    eg_setups, \
    eg_rirs, \
    eg_voice_acquisitions
from eg.reports import eg_reports_rooms, \
    eg_reports_setups, \
    eg_reports_voice_acquisitions, \
    eg_reports_feat_maps, eg_reports_fdrs, eg_reports_rirs

from eg.features.pre_processing import eg_adapt, eg_fdrs
from eg.features.processing import eg_gammaFB, eg_cepstrums, eg_envelope, eg_tds, eg_ffts
from eg.features.post_processing import eg_stack, eg_finalize_ds

from framework.data import fs_utils
from framework.experimental.nn.features.pre_processing import nnpreprocutils
import os
import numpy as np
import warnings


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

    # ?<S file voce
    # <editor-fold desc="rooms/setups/anechoic_voices Loading">
    egvoices_output_dir = "./datasets/CMU_Voices/no_grouping"
    # </editor-fold>

    # S file voce, S file rir  -> S file mono-segnale
    # <editor-fold desc="rooms/setups/voice_acquisitions Generation & Loading">
    # NB, GIOCO COL FUNZIONAMENTO DELLE ACQUISIZIONI PER GESTIRE IL FATTO CHE
    # I SEGNALI ANECOICI SONO MENO DELLE RIR
    # eg_voice_acquisitions.compute(egrirs_output_dir,
    #                               {"all_srcs_signals": [fs_utils.get_dir_filepaths(egvoices_output_dir)]})
    egvoiceacqus_output_dir = eg_voice_acquisitions.load(egrirs_output_dir,
                                  {"all_srcs_signals": [fs_utils.get_dir_filepaths(egvoices_output_dir)]})
    # eg_reports_voice_acquisitions.create_report(egvoiceacqus_output_dir)
    # </editor-fold>

    # ADAPTATION FOR THE VOICE CASE
    win_len, hop_size = 32, 16

    # <editor-fold desc="NN features pre_processing">
    # in questa regione si adattano i formati:
    # da quelli delle varie regioni precedenti ->
    # al formato accettato dalle features

    # adaptation: whichever data format to feature function data format
    L = 64000
    Lhop = L

    # eg_adapt.compute(egvoiceacqus_output_dir, {"len_max": Lhop})
    adapted_out_dir = eg_adapt.load(egvoiceacqus_output_dir, {"len_max": Lhop})
    feature_in_dir = adapted_out_dir
    # </editor-fold>

    # <editor-fold desc="NN features processing">

    # <editor-fold desc="γ-toneFB">
    # eg_gammaFB.compute(feature_in_dir, {"f_s": f_s, "f_lo": 50, "f_hi": 2000, "n_bands": 20,
    #                                     "win_len": win_len, "hop_size": hop_size})
    eggammatones_output_dir = eg_gammaFB.load(feature_in_dir,
                                              {"f_s": f_s, "f_lo": 50, "f_hi": 2000, "n_bands": 20,
                                        "win_len": win_len, "hop_size": hop_size})
    _, L3 = eg_gammaFB.load_shape(eggammatones_output_dir)
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

    # <editor-fold desc="complex-cepstrum">
    # eg_cepstrums.compute(feature_in_dir, {"f_s": f_s, "n_bins": L3})
    egcomplexcepstrums_output_dir = eg_cepstrums.load(feature_in_dir, {"f_s": f_s, "n_bins": L3})
    # </editor-fold>

    # </editor-fold>

    # SHAPES of the data in each file of the dirs below
    postproc_in_dirs = [
        eggammatones_output_dir,  # (20, L3)
        egffts_output_dir,  # (1, L3)
        egmsffts_output_dir,  # (1, L3)
        egcomplexcepstrums_output_dir,  # (1, L3)
        egenvelopes_output_dir,  # (1, L3)
        egtds_output_dir  # (1, L3)
    ]

    # <editor-fold desc="NN features post_processing">
    # in questa regione si adattano i formati:
    # da quelli delle varie features ->
    # al formato accettato dalla rete

    # nonostante la previsione, ci mette meno
    # eg_stack.compute(postproc_in_dirs, {})
    # SHAPES of the data in each file of the dir below: (25, L3)
    feat_maps_dir = eg_stack.load(postproc_in_dirs, {})
    map_shape = eg_stack.load_shape(feat_maps_dir)
    # eg_reports_feat_maps.create_report(feat_maps_dir,
    #                                    [
    #                                        *[r"$\gamma_{"+str(i)+"}$" for i in range(20)],
    #                                        r"$\phi$",
    #                                        r"$\phi_{ms}$",
    #                                        r"$\zeta$",
    #                                        r"$\epsilon$",
    #                                        r"$\chi$"
    #                                    ], win_len, hop_size)

    # creazione dataset "finale":
    # nonostante la previsione, ci mette meno
    # ricomporre X=(30K, 25, L3), class_id=(30K, 1), volume_log=(30K, 1) è fattibile ed
    # è ora caricabile in memoria senza eccessivi problemi (25GB ca)
    # in modo da poter sfruttare appieno la GPU evitando problematiche di InputPipeline

    # # VOLUME REGRESSION OVER R/L/RL
    dsname = "main_vc_frmp0_nofdr_featAG"
    # eg_finalize_ds.compute(feat_maps_dir, dsname)
    # </editor-fold>

    return
