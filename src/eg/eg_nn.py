from typing import List

import framework.experimental.nn.dataset_utilities as dsu
from framework.data.io_relations.dirs import one_to_one_or_many
from framework.experimental.nn.features.post_processing import nnpostprocutils
from framework.experimental.nn.usage import nn_utils, nn_metrics
# import tensorflow as tf
import keras
from keras.models import Model
from keras.utils import Sequence
from pathlib import Path
import os
# # # CNN CLASSIFICATION
# import framework.experimental.nn.usage.batch_generation2 as bg2
# # # CNN REGRESSION
# import framework.experimental.nn.usage.batch_generation3 as bg3
# # # MLP CLASSIFICATION
# import framework.experimental.nn.usage.batch_generation4 as bg4
import pandas as p
import numpy as np


def get_base_model_dir(handler: any) -> str:
    basedir = "./datasets/nn/checkpoints"
    if not Path(basedir).is_dir():
        os.makedirs(basedir)
    return "{}/{}".format(basedir, handler.__name__)


def get_initial_model_dir(base_model_dir: str) -> str:
    return "{}/epoca-{:4d}".format(base_model_dir, 0)


def get_latest_model_dir(base_model_dir: str) -> str:
    checkpoint_files = ["{}/{}".format(base_model_dir, point_name) for point_name in os.listdir(base_model_dir)
                        if "epoca-" in point_name]
    if not Path(base_model_dir).is_dir() or not checkpoint_files:
        raise Exception("Directory not found or empty!")
    latest = max(checkpoint_files, key=os.path.getctime)
    return latest


def default_load(model_dir: str):
    ret = keras.models.load_model(model_dir, custom_objects={
        "pearsons_coeff": nn_metrics.pearsons_coeff,
        "mean_mult": nn_metrics.mean_mult
    })
    ret.summary()
    return ret


def create_or_load_model(handler: any, archi_params: any):
    base_model_dir = get_base_model_dir(handler)

    if not Path(base_model_dir).is_dir():
        initial_model_dir = get_initial_model_dir(base_model_dir)
        # makes base, base/epoca-0
        os.makedirs(initial_model_dir, mode=0o777)
        model = handler(archi_params)
        keras.models.save_model(model, initial_model_dir)      # ora esiste di certo

    latest_model_dir = get_latest_model_dir(base_model_dir)     # epoca-0 o più recente
    latest_epoch = int(latest_model_dir.split('-')[-1])

    # sfrutto i dispositivi preparati, tramite la strategia
    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of synched replicas: {}".format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    latest_model = default_load(latest_model_dir)

    return base_model_dir, latest_model_dir, latest_epoch, latest_model


def train_model(archi: Model, training_gen: Sequence, val_gen: Sequence,
                base_model_dir: str, last_epoch: int, train_params: {} = None):
    h = nn_utils.train(archi, training_gen, val_gen, base_model_dir, last_epoch, train_params)
    p.DataFrame(h.history).to_json("{}/history.json".format(base_model_dir))
    return


def probs_to_ids(y):
    return np.argmax(y, axis=-1).reshape((-1))


# TODO: SAREBBE PIU' COMODO USARE LO SPLIT INVECE DEL GENERATORE - ORA NON PIU'
def predict_classif_model_prob(test_model: Model, test_gen: Sequence):
    tmp = []
    # for each batch append categorical_to_num( (batch=test_gen[i]).(y=[1]) )
    for i in range(len(test_gen)):
        tmp.append(np.argmax(test_gen[i][1], axis=-1))
    ygt = np.concatenate(tmp)    # unpack [[0,1,...], ..., [...,1,0]] in [0,1,...,1,0]
    ypred = test_model.predict(x=test_gen, workers=8)
    return ygt.reshape((-1)), ypred


def predict_classif_model(test_model: Model, test_gen: Sequence):
    ygt, ypred = predict_classif_model_prob(test_model, test_gen)
    return ygt, probs_to_ids(ypred)


def predict_regress_model(test_model: Model, test_gen: Sequence):
    tmp = []
    # for each batch append categorical_to_num( (batch=test_gen[i]).(y=[1]) )
    for i in range(len(test_gen)):
        tmp.append(test_gen[i][1])
    ygt = np.concatenate(tmp)    # unpack
    ypred = test_model.predict(x=test_gen, workers=8)
    return ygt.reshape((-1)), ypred.reshape((-1))


def load_test_model(handler: any, epoca: int):
    base_model_dir = get_base_model_dir(handler)
    k_model_dir = "{}/epoca-{:4d}".format(base_model_dir, epoca)
    if not Path(k_model_dir).is_dir():
        raise Exception("Model not found!")

    k_model = default_load(k_model_dir)
    return base_model_dir, k_model_dir, k_model


# Il chk epocale permette di creare l'history anche in caso di interruzione del training
def load_model_history(base_model_dir: str) -> p.DataFrame:
    return p.read_json("{}/history.json".format(base_model_dir))


def load_best_test_model(handler: any):
    h: p.DataFrame = load_model_history(get_base_model_dir(handler))

    loss = h.val_loss.values
    # metric goes from epoch 1 to n with 0-based argument
    # the dir associated to the best model is 1-based
    return load_test_model(handler, int(np.argmin(loss)) + 1)
