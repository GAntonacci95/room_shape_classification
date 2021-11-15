import tensorflow.keras.backend

import framework.experimental.nn.usage.batch_generation3 as bg3
from eg import eg_nn
from eg.features.post_processing import eg_finalize_ds
from eg.reports import eg_reports_nn
from framework.experimental.nn import dataset_utilities as dsu
from framework.experimental.nn.usage import nn_architectures


def run():
    # from Bayes to GPU Machine

    classes = [
        "RectangleRoomSample",
        "LRoomSample",
        "HouseRoomSample"
    ]

    match = [
        nn_architectures.agenovese_ismraywnnofdrfeatAG_R,
        nn_architectures.agenovese_ismraywnnofdrfeatAG_L,
        nn_architectures.agenovese_ismraywnnofdrfeatAG_H
    ]

    dsname = "main_rlh_ismray_wn_nofdr_featAG"
    df = eg_finalize_ds.load(dsname)
    map_shape = df.X.values[0].shape


    for c, archi in zip(classes, match):
        tensorflow.keras.backend.clear_session()
        dfi = dsu.retain_class_field_labels(df, "class_label", [c])
        dfi = dsu.retain_field_as_y(dfi, "volume_m3")
        # dsu.show_field_distro(dfi, 'y')
        dfitrain, dfival, dfitest = dsu.split_tvt(dfi)
        # dsu.show_field_distro(dfitrain, 'y')
        # dsu.show_field_distro(dfival, 'y')
        # dsu.show_field_distro(dfitest, 'y')

        # <editor-fold desc="NN data preparation - NHWC">
        training_gen, val_gen, test_gen = bg3.get_batch_generators3(dfitrain, dfival, dfitest, map_shape)
        # </editor-fold>

        # <editor-fold desc="NN training">
        base_model_dir, latest_model_dir, latest_epoch, latest_model =\
            eg_nn.create_or_load_model(archi, {"map_shape": map_shape})
        # the training callbacks do automatically save new model checkpoint directories within model_dir
        eg_nn.train_model(latest_model, training_gen, val_gen, base_model_dir, latest_epoch, {
            "red_factor": 0.2, "red_patience": 10, "delta": 100, "early_patience": 20
        })
        # </editor-fold>

        # <editor-fold desc="NN test">
        base_model_dir, k_model_dir, k_model = eg_nn.load_best_test_model(archi)
        eg_reports_nn.create_regress_report(base_model_dir, k_model, test_gen)
        eg_reports_nn.create_regress_var_reports(base_model_dir, k_model, [
            {"split_name": "training", "generator": training_gen},
            {"split_name": "validation", "generator": val_gen},
            {"split_name": "test", "generator": test_gen}
        ])
        # </editor-fold>

    del df
    return
