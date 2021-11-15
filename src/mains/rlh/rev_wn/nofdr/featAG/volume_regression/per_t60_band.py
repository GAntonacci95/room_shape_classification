import tensorflow.keras.backend

import framework.experimental.nn.usage.batch_generation3 as bg3
from eg import eg_nn
from eg.features.post_processing import eg_finalize_ds
from eg.reports import eg_reports_nn
from framework.experimental.nn import dataset_utilities as dsu
from framework.experimental.nn.usage import nn_architectures


def field_bands(x_start: float, x_end: float, x_num_output_ranges: int):
    import numpy as np
    n_splits = 2 * x_num_output_ranges - 1
    # x_start, x_end = df[split_field].min(), df[split_field].max()
    x_band_width = (x_end - x_start) / n_splits

    ret = [
        [bandstart, bandend]
        for bandstart, bandend in zip(np.arange(x_start, x_end, x_band_width),
                                      np.arange(x_start + x_band_width, x_end + x_band_width, x_band_width)
                                      )]
    return [ret[i * 2] for i in range(x_num_output_ranges)]


def run():
    # from Bayes to GPU Machine

    classes = [
        "RectangleRoomSample",
        "LRoomSample",
        "HouseRoomSample"
    ]

    field_name = "t60"
    x_bands = field_bands(0.5, 2.5, 3)

    match = [
        nn_architectures.agenovese_ismraywnnofdrfeatAG_tb0,
        nn_architectures.agenovese_ismraywnnofdrfeatAG_tb2,
        nn_architectures.agenovese_ismraywnnofdrfeatAG_tb4
    ]

    dsname = "main_rlh_ismray_wn_nofdr_featAG"
    df = eg_finalize_ds.load(dsname)
    map_shape = df.X.values[0].shape

    for x_band, archi in zip(x_bands, match):
        tensorflow.keras.backend.clear_session()
        dfi = dsu.select_by_scalar_field_range(df, field_name, *x_band)
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
        base_model_dir, latest_model_dir, latest_epoch, latest_model = \
            eg_nn.create_or_load_model(archi, {"map_shape": map_shape})
        # the training callbacks do automatically save new model checkpoint directories within model_dir
        eg_nn.train_model(latest_model, training_gen, val_gen, base_model_dir, latest_epoch, {
            "red_factor": 0.2, "red_patience": 10, "delta": 100, "early_patience": 33
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
