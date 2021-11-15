from eg import eg_nn
from eg.features.post_processing import eg_finalize_ds
import framework.experimental.nn.dataset_utilities as dsu
import framework.experimental.nn.usage.batch_generation2 as bg2
from eg.reports import eg_reports_nn
from framework.experimental.nn.usage import nn_architectures
import tensorflow as tf


def field_bands(x_start: float, x_end: float, x_num_output_ranges: int):
    import numpy as np
    n_splits = 2 * x_num_output_ranges - 1
    # x_start, x_end = df[split_field].min(), df[split_field].max()
    x_band_width = (x_end-x_start) / n_splits

    ret = [
      [bandstart, bandend]
      for bandstart, bandend in zip(np.arange(x_start, x_end, x_band_width),
                                       np.arange(x_start + x_band_width, x_end + x_band_width, x_band_width)
                                       )]
    return [ret[i*2] for i in range(x_num_output_ranges)]


def run():
    # from Bayes to GPU Machine
    classes = [
        "RectangleRoomSample",
        "LRoomSample",
        "HouseRoomSample"
    ]
    selection_field = "volume_m3"
    bands = field_bands(50, 1050, 3)[0]
    archis = [
        nn_architectures.gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0,      # bad results, diggin'deeper \w featAGAlt...
        nn_architectures.gantonacci_rlhismrayvcnofdrfeatAG_normal_vb2,
        nn_architectures.gantonacci_rlhismrayvcnofdrfeatAG_normal_vb4
    ]

    dsname = "main_rlh_ismray_vc_nofdr_featAG_normal"
    df = eg_finalize_ds.load(dsname)
    df = dsu.id_recode_class_field(df, "class_label", classes, "class_id")

    for band, archi in zip(bands, archis):
        tf.keras.backend.clear_session()
        bandstart, bandend = band
        # <editor-fold desc="NN data preparation - NHWC">
        dfi = dsu.select_by_scalar_field_range(df, selection_field, bandstart, bandend)
        dftrain, dfval, dftest = dsu.split_tvt(dfi)
        map_shape = dftrain.X.values[0].shape
        only_t_feats_dbg = True     # added
        if only_t_feats_dbg:
            a, b = map_shape
            map_shape = (a-3, b)
        training_gen, val_gen, test_gen = bg2.get_batch_generators2(dftrain, dfval, dftest, map_shape, len(classes),
                                                                b_norm=False, only_t_feats_dbg=only_t_feats_dbg)   # changed
        model_params = {"map_shape": map_shape, "n_classes": len(classes), "lr_init": 1E-1}
        train_params = {"red_factor": 0.1, "red_patience": 30, "delta": 0.01, "early_patience": 93}
        # </editor-fold>

        # <editor-fold desc="NN training">
        base_model_dir, latest_model_dir, latest_epoch, latest_model =\
            eg_nn.create_or_load_model(archi, model_params)
        # the training callbacks do automatically save new model checkpoint directories within model_dir
        eg_nn.train_model(latest_model, training_gen, val_gen, base_model_dir, latest_epoch, train_params)
        # </editor-fold>

        # <editor-fold desc="NN test">
        base_model_dir, k_model_dir, k_model = eg_nn.load_best_test_model(archi)
        eg_reports_nn.create_classif_report(base_model_dir, k_model, test_gen, classes)
        eg_reports_nn.create_gradcam_reports(base_model_dir, test_gen, k_model, "conv2d_5", ["average_pooling2d_5",
                                                                                             "dropout",
                                                                                             "batch_normalization_1",
                                                                                             "flatten",
                                                                                             "dense"],
                                             [
                                               *[r"$\gamma_{"+str(i)+"}$" for i in range(20)],
                                               # r"$\phi$",
                                               # r"$\phi_{ms}$",
                                               # r"$\zeta$",
                                               r"$\epsilon$",
                                               r"$\chi$"
                                           ])
        # </editor-fold>
    return
