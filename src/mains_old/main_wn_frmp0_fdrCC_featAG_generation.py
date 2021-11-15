import pyroomacoustics as pra

# NB: eg_nn shall be imported only on workstations with GPUs
from eg import eg_3d, \
    eg_setups, \
    eg_rirs, \
    eg_noises, \
    eg_noise_acquisitions
from eg.reports import eg_reports_rooms, \
    eg_reports_setups, \
    eg_reports_noise_acquisitions, \
    eg_reports_fdrs, \
    eg_reports_feat_maps

from eg.features.pre_processing import eg_adapt, eg_fdrs
from eg.features.processing import eg_gammaFB, eg_cepstrums, eg_envelope, eg_tds, eg_ffts
from eg.features.post_processing import eg_stack, eg_finalize_ds

from framework.data import fs_utils
from framework.experimental.nn.features.pre_processing import nnpreprocutils
import os
import numpy as np


# main application
def run():
    # Bayes
    classes = ["RectangleRoomSample", "LRoomSample"]

    # <editor-fold desc="rooms Generation & Loading">
    # eg_3d.compute({})
    eg3d_output_dir = eg_3d.load({})
    # eg_reports_rooms.create_report(eg3d_output_dir, classes)
    # </editor-fold>

    f_s = 16000
    max_ord = 10
    num_tot_irs = 10000
    frmp = 0
    N = len(fs_utils.get_dir_filenames(eg3d_output_dir))
    # FIXED MATERIALS
    mds = [
        # un materiale con assorbimento oltre 15% su MHF
        pra.materials_absorption_table["felt_5mm"],
        # un materiale con assorbimento attorno 20% in LF
        pra.materials_absorption_table["wooden_lining"]
    ]

    # N file stanza -> S = N * n_materials * n_rnd_acquisitons_per_room file setup
    # <editor-fold desc="rooms/setups Generation & Loading">
    # eg_setups.compute(eg3d_output_dir, {
    #     "materials_descriptors": mds,
    #     "n_rnd_acquisitons_per_room": int(num_tot_irs / (N * len(mds))),
    #     "f_s": f_s,
    #     "max_ord": max_ord
    # })
    egsetups_output_dir = eg_setups.load(eg3d_output_dir, {
        "materials_descriptors": mds,
        "n_rnd_acquisitons_per_room": int(num_tot_irs / (N * len(mds))),
        "f_s": f_s,
        "max_ord": max_ord
    })
    # eg_reports_setups.create_report(egsetups_output_dir, classes)
    # </editor-fold>

    # S file setup -> S file rir
    # <editor-fold desc="rooms/setups/rirs Generation & Loading">
    # eg_rirs.compute(egsetups_output_dir, {})
    egrirs_output_dir = eg_rirs.load(egsetups_output_dir, {})
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
    # eg_noise_acquisitions.compute(egnoises_output_dir,
    #                               {"rir_files_paths": fs_utils.get_dir_filepaths(egrirs_output_dir)})
    egnoiseacqus_output_dir = eg_noise_acquisitions.load(egnoises_output_dir,
                                                         {"rir_files_paths": fs_utils.get_dir_filepaths(
                                                             egrirs_output_dir)})
    # eg_reports_noise_acquisitions.create_report(egnoiseacqus_output_dir)
    # </editor-fold>

    # CI VUOLE MENO DEL PREVISTO ^^
    # <editor-fold desc="fdrsCC Generation & Loading">
    # eg_fdrs.computeCC(egnoiseacqus_output_dir, {"f_s": f_s})
    egnoisefdrs_output_dir = eg_fdrs.loadCC(egnoiseacqus_output_dir, {"f_s": f_s})
    # eg_reports_fdrs.create_report(egnoisefdrs_output_dir, "fdrsCC")
    # </editor-fold>

    # ADAPTATION FOR THE WHITE NOISE CASE
    win_len, hop_size = 8, 4

    # <editor-fold desc="NN features pre_processing">
    # in questa regione si adattano i formati:
    # da quelli delle varie regioni precedenti ->
    # al formato accettato dalle features

    # adaptation: whichever data format to feature function data format
    # sbatta salvare e ricaricare il seguente: eseguo una volta e cablo il risultato
    L = 9612    # nnpreprocutils.get_len_max_of_signals_in_dir(egnoiseacqus_output_dir)
    Lhop = int(np.ceil(L / hop_size) * hop_size)

    # eg_adapt.compute(egnoiseacqus_output_dir, {"len_max": Lhop})
    adapted_out_dir = eg_adapt.load(egnoiseacqus_output_dir, {"len_max": Lhop})
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
    # ricomporre X=(8K, 25, L3), class_id=(8K, 1), volume_log=(8K, 1) è fattibile ed
    # è ora caricabile in memoria senza eccessivi problemi (4GB ca)
    # in modo da poter sfruttare appieno la GPU evitando problematiche di InputPipeline

    # # VOLUME REGRESSION OVER R/L/RL
    dsname = "main_wn_frmp0_nofdr_featAG"
    # eg_finalize_ds.compute(feat_maps_dir, dsname)
    # </editor-fold>

    return
