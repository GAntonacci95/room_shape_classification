# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop using the LEAF frontend."""

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


def __go_train(datasets, info, workdir, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs):
    datasets = data.prepare(datasets, batch_size=batch_size)
    num_classes = info.features['label'].num_classes
    model = models.AudioClassifier(num_outputs=num_classes, **kwargs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = 'sparse_categorical_accuracy'
    model.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=[metric])

    ckpt_path = os.path.join(workdir, 'checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        monitor=f'val_{metric}',
        mode='max',
        save_best_only=True)
    log_dir = os.path.join(workdir, "logs/fit/{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    early_callback = tf.keras.callbacks.EarlyStopping(monitor=f'val_{metric}', min_delta=0.01, patience=30, mode="max")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    h = model.fit(datasets['train'],
            validation_data=datasets['eval'],
            batch_size=None,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[model_checkpoint_callback, early_callback, tensorboard_callback])
    pandas.DataFrame(h.history).to_json("{}/history.json".format(log_dir))
    return


def __go_train_vol_reg(datasets, workdir, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs):
    datasets = data.prepare_vol_reg(datasets, batch_size=batch_size)
    # one dense output layer for the regression
    model = models.AudioClassifier(num_outputs=1, **kwargs)
    # TODO poteva essere effettivamente meglio specificare ReLU invece di lasciar libera l'attivazione.

    model.compile(loss="mean_squared_error",
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=["mean_absolute_error"])

    ckpt_path = os.path.join(workdir, 'checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        monitor=f'val_loss',
        mode='min',
        save_best_only=True)
    log_dir = os.path.join(workdir, "logs/fit/{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    red_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, mode="min",
                                                        min_delta=10, min_lr=1E-10)
    early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=10, patience=30, mode="min")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    h = model.fit(datasets['train'],
            validation_data=datasets['eval'],
            batch_size=None,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[model_checkpoint_callback, red_callback, early_callback, tensorboard_callback])
    pandas.DataFrame(h.history).to_json("{}/history.json".format(log_dir))
    return


def __go_train_t60_reg(datasets, workdir, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs):
    datasets = data.prepare_t60_reg(datasets, batch_size=batch_size)
    # one dense output layer for the regression
    model = models.AudioClassifier(num_outputs=1, **kwargs)
    model._head.activation("relu")

    model.compile(loss="mean_squared_error",
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=["mean_absolute_error"])

    ckpt_path = os.path.join(workdir, 'checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        monitor=f'val_loss',
        mode='min',
        save_best_only=True)
    log_dir = os.path.join(workdir, "logs/fit/{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    red_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, mode="min",
                                                        min_delta=0.01, min_lr=1E-20)
    early_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.01, patience=20, mode="min")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    h = model.fit(datasets['train'],
            validation_data=datasets['eval'],
            batch_size=None,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[model_checkpoint_callback, red_callback, early_callback, tensorboard_callback])
    pandas.DataFrame(h.history).to_json("{}/history.json".format(log_dir))
    return


@gin.configurable
def train(workdir: str = '/tmp/',
            dataset: str = 'speech_commands',
            num_epochs: int = 10,
            steps_per_epoch: Optional[int] = None,
            learning_rate: float = 1e-4,
            batch_size: int = 64,
            **kwargs):
    """Trains a model on a dataset.

    Args:
    workdir: where to store the checkpoints and metrics.
    dataset: name of a tensorflow_datasets audio datasset.
    num_epochs: number of epochs to training the model for.
    steps_per_epoch: number of steps that define an epoch. If None, an epoch is
      a pass over the entire training set.
    learning_rate: Adam's learning rate.
    batch_size: size of the mini-batches.
    **kwargs: arguments to the models.AudioClassifier class, namely the encoder
      and the frontend models (tf.keras.Model).
    """
    datasets, info = tfds.load(dataset, with_info=True)
    __go_train(datasets, info, workdir, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs)
    return


@gin.configurable
def train_x_bands(workdir: str = '/tmp/',
          dataset: str = 'speech_commands',
          num_epochs: int = 10,
          steps_per_epoch: Optional[int] = None,
          learning_rate: float = 1e-4,
          batch_size: int = 32,
          vol_start: int = 50, vol_end: float = 1050, vol_num_output_ranges: int = 3,
          t60_start: int = 0.5, t60_end: float = 2.5, t60_num_output_ranges: int = 3,
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
            dsfilt = data.filter_by_scalar_field_range(datasets, x_name, x_start, x_end)
            __go_train(dsfilt, info, wd, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs)
    return


@gin.configurable
def train_vol_reg_per_class(workdir: str = '/tmp/',
            dataset: str = 'speech_commands',
            num_epochs: int = 10,
            steps_per_epoch: Optional[int] = None,
            learning_rate: float = 1e-4,
            batch_size: int = 64,
            **kwargs):
    """Trains a model on a dataset.

    Args:
    workdir: where to store the checkpoints and metrics.
    dataset: name of a tensorflow_datasets audio datasset.
    num_epochs: number of epochs to training the model for.
    steps_per_epoch: number of steps that define an epoch. If None, an epoch is
      a pass over the entire training set.
    learning_rate: Adam's learning rate.
    batch_size: size of the mini-batches.
    **kwargs: arguments to the models.AudioClassifier class, namely the encoder
      and the frontend models (tf.keras.Model).
    """
    datasets, info = tfds.load(dataset, with_info=True)
    workdir = os.path.join(workdir, "vol_reg")
    for c in info.features["label"].names:
        wd = os.path.join(workdir, "_{}".format(c[0]))
        dsfilt = data.filter_by_class(datasets, info.features["label"].names.index(c))
        __go_train_vol_reg(dsfilt, wd, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs)
    return


@gin.configurable
def train_vol_reg_x_bands(workdir: str = '/tmp/',
          dataset: str = 'speech_commands',
          num_epochs: int = 10,
          steps_per_epoch: Optional[int] = None,
          learning_rate: float = 1e-4,
          batch_size: int = 32,
          vol_start: int = 50, vol_end: float = 1050, vol_num_output_ranges: int = 3,
          t60_start: int = 0.5, t60_end: float = 2.5, t60_num_output_ranges: int = 3,
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
            dsfilt = data.filter_by_scalar_field_range(datasets, x_name, x_start, x_end)
            __go_train_vol_reg(dsfilt, wd, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs)
    return


@gin.configurable
def train_t60_reg_per_class(workdir: str = '/tmp/',
            dataset: str = 'speech_commands',
            num_epochs: int = 10,
            steps_per_epoch: Optional[int] = None,
            learning_rate: float = 1e-4,
            batch_size: int = 64,
            **kwargs):
    """Trains a model on a dataset.

    Args:
    workdir: where to store the checkpoints and metrics.
    dataset: name of a tensorflow_datasets audio datasset.
    num_epochs: number of epochs to training the model for.
    steps_per_epoch: number of steps that define an epoch. If None, an epoch is
      a pass over the entire training set.
    learning_rate: Adam's learning rate.
    batch_size: size of the mini-batches.
    **kwargs: arguments to the models.AudioClassifier class, namely the encoder
      and the frontend models (tf.keras.Model).
    """
    datasets, info = tfds.load(dataset, with_info=True)
    workdir = os.path.join(workdir, "t60_reg")
    for c in info.features["label"].names:
        wd = os.path.join(workdir, "_{}".format(c[0]))
        dsfilt = data.filter_by_class(datasets, info.features["label"].names.index(c))
        __go_train_t60_reg(dsfilt, wd, num_epochs, steps_per_epoch, learning_rate, batch_size, **kwargs)
    return
