# NB: eg_nn shall be imported only on workstations with GPUs
import framework.experimental.nn.dataset_utilities as dsu
from eg import eg_nn
from eg.reports import eg_reports_nn

from eg.features.post_processing import eg_finalize_rir_ds

# NB: nn_architectures shall be imported only on workstations with GPUs
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


# main application
def run():
    # from Bayes to GPU Machine

    classes = ["RectangleRoomSample", "LRoomSample"]

    # # SHAPE CLASSIFICATION
    dsname = "main_rir_cut"
    df = eg_finalize_rir_ds.load(dsname)

    # <editor-fold desc="data preparation">
    df = dsu.id_recode_class_field(df, "class_label", classes, "class_id")
    # dsu.show_field_distro(df, "class_id")
    df = dsu.to_uniform_class_field_distro(df, "class_id")
    # dsu.show_field_distro(df, "class_id")
    df = dsu.retain_field_as_y(df, "class_id")
    dftrain, dfval, dftest = dsu.split_tvt(df)
    # dsu.show_field_distro(dftrain, 'y')
    # dsu.show_field_distro(dfval, 'y')
    # dsu.show_field_distro(dftest, 'y')
    del df, dftest
    input_len = dftrain.X.values[0].shape[1]

    X_train, y_train = np.empty((len(dftrain), input_len), dtype=np.float), \
                       np.empty((len(dftrain),), dtype=np.int)
    X_test, y_test = np.empty((len(dfval), input_len), dtype=np.float), \
                     np.empty((len(dfval),), dtype=np.int)
    for i in range(len(dftrain)):
        X_train[i, :], y_train[i] = dftrain.X.values[i].reshape((-1,)), dftrain.y.values[i]
    for i in range(len(dfval)):
        X_test[i, :], y_test[i] = dfval.X.values[i].reshape((-1,)), dfval.y.values[i]
    del dftrain, dfval
    # </editor-fold>

    # <editor-fold desc="training">
    rfc = RandomForestClassifier(n_estimators=128, n_jobs=4, verbose=3)
    rfc.n_classes_ = 2
    trained_forest = rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print("########## ########## ##########")
    print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
    print("Balanced accuracy: {:.3f}".format(balanced_accuracy_score(y_test, y_pred)))
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred)))
    print("########## ########## ##########")
    # </editor-fold>

    return
