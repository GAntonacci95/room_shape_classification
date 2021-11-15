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
    eg_reports_feat_maps

from eg.features.pre_processing import eg_adapt
from eg.features.processing import eg_gammaFB, eg_cepstrums, eg_envelope, eg_tds, eg_ffts
from eg.features.post_processing import eg_stack, eg_finalize_rir_ds

from framework.data import fs_utils
from framework.experimental.nn.features.pre_processing import nnpreprocutils
import os

# NB: nn_architectures shall be imported only on workstations with GPUs
from framework.experimental.nn.usage import nn_architectures


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

    L = 4812    # nnpreprocutils.get_len_max_of_rirs_in_dir(egrirs_output_dir)


    # creazione dataset "finale":
    # nonostante la previsione, ci mette meno
    # ricomporre X=(8K, 1, L), class_id=(8K, 1), volume_log=(8K, 1) è fattibile ed
    # è ora caricabile in memoria senza eccessivi problemi (300MB ca)
    # in modo da poter sfruttare appieno la GPU evitando problematiche di InputPipeline

    # # SHAPE CLASSIFICATION
    dsname = "main_rir_nocut"
    # eg_finalize_rir_ds.compute(egrirs_output_dir, L, dsname)
    # </editor-fold>

    return
