# NB: eg_nn shall be imported only on workstations with GPUs
import framework.experimental.nn.dataset_utilities as dsu
from eg import eg_nn
from eg.reports import eg_reports_nn

from eg.features.post_processing import eg_finalize_ds

import framework.experimental.nn.usage.batch_generation2 as bg2

# NB: nn_architectures shall be imported only on workstations with GPUs
from framework.experimental.nn.usage import nn_architectures


# main application
def run():
    # from Bayes to GPU Machine

    classes = ["RectangleRoomSample", "LRoomSample", "HouseRoomSample"]

    # # SHAPE CLASSIFICATION
    dsname = "main_vc_frmp0_nofdr_featAG"
    df = eg_finalize_ds.load(dsname)

    # <editor-fold desc="NN data preparation - NHWC">
    df = dsu.id_recode_class_field(df, "class_label", classes, "class_id")
    dftrain, dfval, dftest = dsu.split_tvt(df)
    del df
    map_shape = dftrain.X.values[0].shape
    training_gen, val_gen, test_gen = bg2.get_batch_generators2(dftrain, dfval, dftest, map_shape, len(classes))
    model_params = {"map_shape": map_shape, "n_classes": len(classes)}
    train_params = {"red_factor": 0.2, "red_patience": 5, "delta": 0.01, "early_patience": 12}
    # </editor-fold>

    # # <editor-fold desc="NN training">
    # base_model_dir, latest_model_dir, latest_epoch, latest_model =\
    #     eg_nn.create_or_load_model(nn_architectures.gantonacci_vcfrmp0nofdrfeatAG, model_params)
    # # the training callbacks do automatically save new model checkpoint directories within model_dir
    # eg_nn.train_model(latest_model, training_gen, val_gen, base_model_dir, latest_epoch, train_params)
    # # </editor-fold>

    # <editor-fold desc="NN test">
    base_model_dir, k_model_dir, k_model = eg_nn.load_best_test_model(nn_architectures.gantonacci_vcfrmp0nofdrfeatAG)
    eg_reports_nn.create_classif_report(base_model_dir, k_model, test_gen, classes)
    eg_reports_nn.create_gradcam_reports(base_model_dir, test_gen, k_model, "conv2d_5", ["average_pooling2d_5",
                                                                                         "dropout",
                                                                                         "flatten",
                                                                                         "dense"],
                                         [
                                           *[r"$\gamma_{"+str(i)+"}$" for i in range(20)],
                                           r"$\phi$",
                                           r"$\phi_{ms}$",
                                           r"$\zeta$",
                                           r"$\epsilon$",
                                           r"$\chi$"
                                       ])
    # </editor-fold>

    del dftrain, dfval, dftest
    return
