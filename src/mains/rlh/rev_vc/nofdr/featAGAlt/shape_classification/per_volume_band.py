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
    band = field_bands(50, 1050, 3)[0]
    archis = [
        nn_architectures.gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0_c0tf,
        nn_architectures.gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0_c1tf,
        nn_architectures.gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0_c2tf
    ]
    n_gamma_bands = 30

    # temprary ds, result of featAGAlt
    dsname = "tmp_vc_vb0_normal"     # "main_rlh_ismray_vc_nofdr_featAGAlt_normal"
    df = eg_finalize_ds.load(dsname)
    initial_fields = df.columns
    df = dsu.curr_cls_vs_oth(df, "class_label", classes)

    for i in range(len(archis)):
        archi = archis[i]
        cur_cls = "class_label_{}".format(i)
        dfi = df[[*initial_fields, cur_cls]]
        classes = dfi[cur_cls].unique()
        dfi = dsu.id_recode_class_field(dfi, cur_cls, classes, "class_id")
        tf.keras.backend.clear_session()
        bandstart, bandend = band
        # <editor-fold desc="NN data preparation - NHWC">
        dfi = dsu.select_by_scalar_field_range(dfi, selection_field, bandstart, bandend)
        dftrain, dfval, dftest = dsu.split_tvt(dfi)
        map_shape = dftrain.X.values[0].shape
        training_gen, val_gen, test_gen = bg2.get_batch_generators2(dftrain, dfval, dftest, map_shape, len(classes),
                                                                b_norm=False)
        model_params = {"map_shape": map_shape, "n_classes": len(classes), "lr_init": 1*1E-2}
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
                                               *[r"$\gamma_{"+str(i)+"}$" for i in range(n_gamma_bands)]
                                           ])
        # </editor-fold>
    return
