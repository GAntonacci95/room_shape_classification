import os
from typing import Optional

import gin
import pandas
import tensorflow.keras.backend

from leaf_audio import models
from example import data
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
import matplotlib.pyplot as plt
import numpy as np


def confusion_matrix(ygt, ypred):
    return tf.math.confusion_matrix(ygt, ypred)


def normalize_confusion_matrix(mat, num_classes):
    return mat / np.repeat(np.sum(mat, axis=1).reshape((num_classes, 1)), num_classes, axis=1)


def __create_confusion_report_sub(mat, classes: [str]):
    fig, ax = plt.subplots()
    lin = np.arange(len(classes))

    ax.set_xlabel(r"$y_{PRED}$")
    ax.set_ylabel(r"$y_{GT}$")
    ax.yaxis.set_label_position("right")
    ax.set_xticks(lin)
    ax.set_yticks(lin)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.matshow(mat, cmap=plt.get_cmap("summer"))

    for i in lin:
        for j in lin:
            # ax.text(c, r, ... mat[r, c] ...)
            ax.text(j, i, "{:.2f}".format(mat[i, j]), va='center', ha='center', size="x-large")
    plt.tight_layout()
    return fig


def __create_confusion_report(ygt, ypred, classes: [str],
                              report_dir: str,
                              varname: str, varfrom: float, varto: float):
    # for 2 classes: [[TN, FP], [FN, TP]], r is GT (N, P), c is PRED (N, P)
    # reference: https://stackoverflow.com/questions/52852742/how-to-read-tensorflow-confusion-matrix-rows-and-columns
    mat = confusion_matrix(ygt, ypred)
    # row-wise normalization
    mat = normalize_confusion_matrix(mat, len(classes))

    fig = __create_confusion_report_sub(mat, classes)
    plt.savefig("{}/confusion_{}_{:.2f}_{:.2f}.png".format(report_dir, varname, varfrom, varto), dpi=300)
    plt.close(fig)
    return


def get_prediction_df(datasets, testdata, model):
    preds = np.argmax(model.predict(testdata["test"]), axis=1)
    ret = pandas.DataFrame(columns=["volume_m3", "t60", "y_gt", "y_pred"])
    for e1, ypred in zip(list(datasets["test"].as_numpy_iterator()), preds):
        audiopcm, ygt, vol, t60 = e1["audio"], e1["label"], e1["volume_m3"], e1["t60"]
        nu_row = pandas.DataFrame([{
            "volume_m3": vol,
            "t60": t60,
            "y_gt": ygt,
            "y_pred": ypred
        }])
        ret = pandas.concat([ret, nu_row])
    return ret


def sub_report_field_ranged_confusions(d: dict, classes: [str], split_field: str, workdir: str):
    varfrom, varto = d["bandstart"], d["bandend"]
    ygt, ypred = np.asarray(d["data"].y_gt.values).astype(np.int), np.asarray(d["data"].y_pred.values).astype(np.int)
    __create_confusion_report(ygt, ypred, classes, workdir, split_field, varfrom, varto)
    return


def report_field_confusions(df: pandas.DataFrame, classes: [str],
                                   split_field: str, x_band_start: float, x_band_end: float,
                                   workdir: str):
    # here wowdf is already filtered
    tmp = {"bandstart": x_band_start, "bandend": x_band_end, "data": df}
    sub_report_field_ranged_confusions(tmp, classes, split_field, workdir)
    return


def report_field_ranged_confusions(df: pandas.DataFrame, classes: [str],
                                   split_field: str, x_start: float, x_end: float, x_num_output_ranges: int,
                                   workdir: str):
    # here wowdf must be filtered
    tmp = [{"bandstart": bandstart, "bandend": bandend,
            "data": df.loc[(df[split_field] >= bandstart) & (df[split_field] <= bandend)]}
           for bandstart, bandend in data.field_bands(x_start, x_end, x_num_output_ranges)]
    for x_range in tmp:
        sub_report_field_ranged_confusions(x_range, classes, split_field, workdir)
    return


def __go_test(datasets, info, workdir, learning_rate, batch_size, **kwargs):
    testdata = data.prepare_test(datasets, batch_size=batch_size)
    num_classes = info.features['label'].num_classes
    testmodel = models.AudioClassifier(num_outputs=num_classes, **kwargs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = 'sparse_categorical_accuracy'
    testmodel.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=[metric])

    ckpt_path = os.path.join(workdir, 'checkpoint')
    testmodel.load_weights(ckpt_path)
    wowdf = get_prediction_df(datasets, testdata, testmodel)
    return wowdf


def __go_test_vol_reg(datasets, workdir, learning_rate, batch_size, **kwargs):
    raise Exception("IMPLEMENTAMIII")
    testdata = data.prepare_test_vol_reg(datasets, batch_size=batch_size)
    num_classes = info.features['label'].num_classes
    testmodel = models.AudioClassifier(num_outputs=num_classes, **kwargs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = 'sparse_categorical_accuracy'
    testmodel.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=[metric])

    ckpt_path = os.path.join(workdir, 'checkpoint')
    testmodel.load_weights(ckpt_path)
    wowdf = get_prediction_df(datasets, testdata, testmodel)
    return wowdf


@gin.configurable
def test(workdir: str = '/tmp/',
          dataset: str = 'speech_commands',
          learning_rate: float = 1e-4,
          batch_size: int = 64,
          **kwargs):
    # here the as_supervised is neglected because further fields are needed for testing
    datasets, info = tfds.load(dataset, with_info=True)
    wowdf = __go_test(datasets, info, workdir, learning_rate, batch_size, **kwargs)
    report_field_ranged_confusions(wowdf, info.features["label"].names, "volume_m3", 50, 1050, 3, workdir)
    # la distro dei t60 non è proprio soddisfacentemente uniforme... introduciamo v_start, v_end come parametri ^^
    report_field_ranged_confusions(wowdf, info.features["label"].names, "t60", 0.7, 1.7, 3, workdir)
    return


@gin.configurable
def test_x_bands(workdir: str = '/tmp/',
          dataset: str = 'speech_commands',
          learning_rate: float = 1e-4,
          batch_size: int = 64,
          vol_start: int = 50, vol_end: float = 1050, vol_num_output_ranges: int = 3,
          t60_start: int = 0.7, t60_end: float = 1.7, t60_num_output_ranges: int = 3,
          **kwargs):
    datasets, info = tfds.load(dataset, with_info=True)
    xs = [
        {"x_name": "volume_m3", "x_bands": data.field_bands(vol_start, vol_end, vol_num_output_ranges)},
        {"x_name": "t60", "x_bands": data.field_bands(t60_start, t60_end, t60_num_output_ranges)}
    ]

    for x in xs:
        x_name = x["x_name"]
        x_bands = x["x_bands"]
        for x_band in x_bands:
            tensorflow.keras.backend.clear_session()
            x_start, x_end = x_band
            wd = os.path.join(workdir, "{}/{:.2f}_{:.2f}".format(x_name, x_start, x_end))
            dss = data.filter_by_scalar_field_range(datasets, x_name, x_start, x_end)
            wowdf = __go_test(dss, info, wd, learning_rate, batch_size, **kwargs)
            report_field_confusions(wowdf, info.features["label"].names, x_name, x_start, x_end, wd)
    return


@gin.configurable
def test_vol_reg_per_class(workdir: str = '/tmp/',
          dataset: str = 'speech_commands',
          learning_rate: float = 1e-4,
          batch_size: int = 64,
          **kwargs):
    # here the as_supervised is neglected because further fields are needed for testing
    datasets, info = tfds.load(dataset, with_info=True)
    workdir = os.path.join(workdir, "vol_reg")
    wowdf = __go_test_vol_reg(datasets, workdir, learning_rate, batch_size, **kwargs)
    raise Exception("other things...")
    return


@gin.configurable
def test_vol_reg_x_bands(workdir: str = '/tmp/',
          dataset: str = 'speech_commands',
          learning_rate: float = 1e-4,
          batch_size: int = 64,
          vol_start: int = 50, vol_end: float = 1050, vol_num_output_ranges: int = 3,
          t60_start: int = 0.7, t60_end: float = 1.7, t60_num_output_ranges: int = 3,
          **kwargs):
    datasets, info = tfds.load(dataset, with_info=True)
    workdir = os.path.join(workdir, "vol_reg")
    xs = [
        {"x_name": "volume_m3", "x_bands": data.field_bands(vol_start, vol_end, vol_num_output_ranges)},
        {"x_name": "t60", "x_bands": data.field_bands(t60_start, t60_end, t60_num_output_ranges)}
    ]

    for x in xs:
        x_name = x["x_name"]
        x_bands = x["x_bands"]
        for x_band in x_bands:
            tensorflow.keras.backend.clear_session()
            x_start, x_end = x_band
            wd = os.path.join(workdir, "{}/{:.2f}_{:.2f}".format(x_name, x_start, x_end))
            dss = data.filter_by_scalar_field_range(datasets, x_name, x_start, x_end)
            wowdf = __go_test_vol_reg(dss, wd, learning_rate, batch_size, **kwargs)
            raise Exception("other things...")
    return
