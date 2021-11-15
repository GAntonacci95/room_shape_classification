from eg import eg_3d, eg_setups, eg_rirs, eg_acquisitions
from eg.features.post_processing import eg_finalize_rir_ds, eg_finalize_acqus_ds, eg_stack, eg_finalize_ds
from eg.features.pre_processing import eg_adapt
from eg.features.processing import eg_gammaFB, eg_cepstrums, eg_ffts, eg_tds, eg_envelope
from eg.reports import eg_reports_acquisitions, eg_reports_feat_maps, eg_report_ds
from framework.data import fs_utils
import numpy as np



def copy_v_trick(in_dir: str, v_start: float, v_end: float, out_dir: str):
    import os.path
    import pathlib
    from framework.data.io_relations.dirs import file_room_df_chain
    from shutil import copyfile
    from framework.experimental.simulation import room_utilities

    if pathlib.Path(out_dir).is_dir() and len(fs_utils.get_dir_filenames(out_dir)) > 0:
        return out_dir
    else:
        os.makedirs(out_dir, mode=0o777, exist_ok=True)

    fnames = fs_utils.get_dir_filepaths(in_dir)
    for f in fnames:
        fname = fs_utils.get_filepath_filename(f)
        room_df = file_room_df_chain.get_room_df_by_chaining(f)
        georoom = room_utilities.georoom_from_room_df(room_df)
        if v_start <= georoom.volume <= v_end:
            copyfile(f, os.path.join(out_dir, fname))
    return out_dir


def copy_t_trick(in_dir: str, t_start: float, t_end: float, out_dir: str):
    import os.path
    import pathlib
    from framework.data.io_relations.dirs import file_rir_df_chain
    from shutil import copyfile

    if pathlib.Path(out_dir).is_dir() and len(fs_utils.get_dir_filenames(out_dir)) > 0:
        return out_dir
    else:
        os.makedirs(out_dir, mode=0o777, exist_ok=True)

    fnames = fs_utils.get_dir_filepaths(in_dir)
    for f in fnames:
        fname = fs_utils.get_filepath_filename(f)
        rir_df = file_rir_df_chain.get_rir_df_by_chaining(f)
        if t_start <= rir_df.t60.values[0] <= t_end:
            copyfile(f, os.path.join(out_dir, fname))
    return out_dir


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
    egrirs_output_dir, dsname, dsnamenormal = \
        eg_rirs.load(egsetups_output_dir, rirs_params), \
        "main_rlh_ismray_vc_nofdr_featAGAlt", \
        "main_rlh_ismray_vc_nofdr_featAGAlt_normal"
    # eg_reports_rirs.create_report(egrirs_output_dir)

    # NB questo file contiene qualche porcata

    # egrirs_output_dir, dsname, dsnamenormal, L = \
    #     copy_v_trick(egrirs_output_dir, 50, 250, "{}/{}".format(egrirs_output_dir, "vb0")), \
    #     "tmp_vc_vb0", \
    #     "tmp_vc_vb0_normal", 124944
    egrirs_output_dir, dsname, dsnamenormal, L = \
        copy_t_trick(egrirs_output_dir, 0.5, 0.9, "{}/{}".format(egrirs_output_dir, "tb0")), \
        "tmp_vc_tb0", \
        "tmp_vc_tb0_normal", 128464
    # </editor-fold>

    # ?<S file voce
    # <editor-fold desc="rooms/setups/anechoic_voices Loading">
    egvoices_output_dir = "./datasets/CMU_Voices/no_grouping"
    # </editor-fold>

    # S file rumore, S file rir  -> S file mono-segnale
    # <editor-fold desc="rooms/setups/noise_acqisitions Generation & Loading">
    # eg_acquisitions.compute(egrirs_output_dir,
    #                               {"all_srcs_signals": [fs_utils.get_dir_filepaths(egvoices_output_dir)]})
    egvoiceacqus_output_dir = eg_acquisitions.load(egrirs_output_dir,
                                  {"all_srcs_signals": [fs_utils.get_dir_filepaths(egvoices_output_dir)]})
    # eg_reports_acquisitions.create_report(egvoiceacqus_output_dir, "sensed_o10_ray_vc")
    # </editor-fold>

    # remember L3(cepst_nbins(len(decimation(signal))))
    # win_len, hop_size = 32, 16
    # changed to, considering 4s of audio from the end
    win_len, hop_size = 8, 4
    # maybe fdrs again? otherwise mem explodes...

    # contemplo len(rir_longest)
    from framework.experimental.nn.features.pre_processing import nnpreprocutils
    # Lnegligible = 44049     # nnpreprocutils.get_len_max_of_rirs_in_dir(egrirs_output_dir)
    # L = 124944  # nnpreprocutils.get_len_max_of_signals_in_dir(egvoiceacqus_output_dir) fixed for vb0
    # L = 128464  # fixed for tb0
    # L -= Lnegligible
    Lhop = int(np.ceil(L / hop_size) * hop_size)

    # <editor-fold desc="NN features pre_processing">
    adapt_params = {"l_start": Lhop - 4 * 16000, "l_end": Lhop}
    # eg_adapt.compute(egvoiceacqus_output_dir, adapt_params)
    adapted_out_dir = eg_adapt.load(egvoiceacqus_output_dir, adapt_params)
    feature_in_dir = adapted_out_dir
    # </editor-fold>

    # <editor-fold desc="NN features processing">

    # <editor-fold desc="γ-toneFB">
    n_gamma_bands = 30
    # original, new win size
    # gamma_params = {"f_s": f_s, "f_lo": 50, "f_hi": 2000, "n_bands": 20, "win_len": win_len, "hop_size": hop_size}
    gamma_params = {"f_s": f_s, "f_lo": 50, "f_hi": 2000, "n_bands": 30, "win_len": win_len, "hop_size": hop_size}
    # trials leading to no improvement at all... wl, hs = 32, 16
    # gamma_params = {"f_s": f_s, "f_lo": 50, "f_hi": 4000, "n_bands": 30, "win_len": win_len, "hop_size": hop_size}
    # gamma_params = {"f_s": f_s, "f_lo": 20, "f_hi": 2000, "n_bands": 30, "win_len": win_len, "hop_size": hop_size}
    # what about this way?
    # gamma_params = {"f_s": f_s, "f_lo": 20, "f_hi": 2000, "n_bands": 15, "win_len": win_len, "hop_size": hop_size}
    # eg_gammaFB.compute(feature_in_dir, gamma_params)
    eggammatones_output_dir = eg_gammaFB.load(feature_in_dir, gamma_params)
    _, L3 = eg_gammaFB.load_shape(eggammatones_output_dir)
    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc="NN features post_processing">

    # SHAPES of the data in each file of the dir below: (n_gamma_bands, L3)
    feat_maps_dir = eggammatones_output_dir
    # eg_reports_feat_maps.create_report(feat_maps_dir,
    #                                    [
    #                                        *[r"$\gamma_{"+str(i)+"}$" for i in range(n_gamma_bands)]
    #                                    ], win_len, hop_size)

    # eg_finalize_ds.compute(feat_maps_dir, dsname)
    # eg_finalize_ds.feat_normalize(dsname, dsnamenormal, n_gamma_bands)

    # # NB qui sotto evitare, scoppia la RAM
    # eg_report_ds.create_report(dsnamenormal,
    #                                    [
    #                                        *[r"$\gamma_{"+str(i)+"}$" for i in range(n_gamma_bands)]
    #                                    ], win_len, hop_size, "vc_featAGAlt_normal")

    # </editor-fold>
    return
