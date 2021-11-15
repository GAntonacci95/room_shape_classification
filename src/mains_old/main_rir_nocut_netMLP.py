# NB: eg_nn shall be imported only on workstations with GPUs
import framework.experimental.nn.dataset_utilities as dsu
from eg import eg_nn
from eg.reports import eg_reports_nn

from eg.features.post_processing import eg_finalize_rir_ds

import framework.experimental.nn.usage.batch_generation4 as bg4

# NB: nn_architectures shall be imported only on workstations with GPUs
from framework.experimental.nn.usage import nn_architectures


# main application
def run():
    # from Bayes to GPU Machine

    classes = ["RectangleRoomSample", "LRoomSample"]

    # # SHAPE CLASSIFICATION
    dsname = "main_rir_nocut"
    df = eg_finalize_rir_ds.load(dsname)

    # <editor-fold desc="NN data preparation">
    df = dsu.id_recode_class_field(df, "class_label", classes, "class_id")
    # dsu.show_field_distro(df, "class_id")
    df = dsu.to_uniform_class_field_distro(df, "class_id")
    # dsu.show_field_distro(df, "class_id")
    df = dsu.retain_field_as_y(df, "class_id")
    dftrain, dfval, dftest = dsu.split_tvt(df)
    # dsu.show_field_distro(dftrain, 'y')
    # dsu.show_field_distro(dfval, 'y')
    # dsu.show_field_distro(dftest, 'y')
    del df
    input_shape = dftrain.X.values[0].shape
    training_gen, val_gen, test_gen = bg4.get_batch_generators4(dftrain, dfval, dftest, input_shape, len(classes))
    # </editor-fold>

    # <editor-fold desc="NN training">
    base_model_dir, latest_model_dir, latest_epoch, latest_model =\
        eg_nn.create_or_load_model(nn_architectures.gantonacci_rirnocutMLP, {"input_shape": input_shape,
                                                                            "n_classes": len(classes)})
    # the training callbacks do automatically save new model checkpoint directories within model_dir
    eg_nn.train_model(latest_model, training_gen, val_gen, base_model_dir, latest_epoch)
    # </editor-fold>

    # <editor-fold desc="NN test">
    base_model_dir, k_model_dir, k_model = eg_nn.load_best_test_model(nn_architectures.gantonacci_rirnocutMLP)
    eg_reports_nn.create_classif_report(base_model_dir, k_model, test_gen, classes)
    # </editor-fold>

    del dftrain, dfval, dftest
    return
