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

"""Preprocess the input data."""

import functools
from typing import Dict, Mapping

import gin
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE


def db_to_linear(samples):
    return 10.0 ** (samples / 20.0)


@gin.configurable
def loudness_normalization(samples: tf.Tensor,
                           target_db: float = 15.0,
                           max_gain_db: float = 30.0):
    """Normalizes the loudness of the input signal."""
    std = tf.math.reduce_std(samples) + 1e-9
    gain = tf.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
    return gain * samples


@gin.configurable
def align(samples: tf.Tensor, seq_len: int = 16000):
    pad_length = tf.maximum(seq_len - tf.size(samples), 0)
    return tf.image.random_crop(tf.pad(samples, [[0, pad_length]]), [seq_len])


def preprocess(inputs: Mapping[str, tf.Tensor],
               transform_fns=(align, loudness_normalization)):
    """Sequentially applies the transformations to the waveform."""
    audio = tf.cast(inputs['audio'], tf.float32) / tf.int16.max
    for transform_fn in transform_fns:
        audio = transform_fn(audio)
    return audio, inputs['label']


def preprocess_vol_reg(inputs: Mapping[str, tf.Tensor],
               transform_fns=(align, loudness_normalization)):
    """Sequentially applies the transformations to the waveform."""
    audio = tf.cast(inputs['audio'], tf.float32) / tf.int16.max
    for transform_fn in transform_fns:
        audio = transform_fn(audio)
    return audio, inputs["volume_m3"]


def preprocess_t60_reg(inputs: Mapping[str, tf.Tensor],
               transform_fns=(align, loudness_normalization)):
    """Sequentially applies the transformations to the waveform."""
    audio = tf.cast(inputs['audio'], tf.float32) / tf.int16.max
    for transform_fn in transform_fns:
        audio = transform_fn(audio)
    return audio, inputs["t60"]


@gin.configurable
def prepare(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
    """Prepares the datasets for training and evaluation."""
    valid = 'validation' if 'validation' in datasets else 'test'
    result = {}
    for split, key in ('train', 'train'), (valid, 'eval'):
        ds = datasets[split]
        ds = ds.map(functools.partial(preprocess, transform_fns=transform_fns),
                    num_parallel_calls=AUTOTUNE)
        result[key] = ds.batch(batch_size).prefetch(AUTOTUNE)
    return result


@gin.configurable   # customization: TODO extend the configuration files
def prepare_test(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
    """Prepares the datasets for test."""
    result = {}
    ds = datasets["test"]
    ds = ds.map(functools.partial(preprocess, transform_fns=transform_fns),
              num_parallel_calls=AUTOTUNE)
    result["test"] = ds.batch(batch_size).prefetch(AUTOTUNE)
    return result


@gin.configurable   # customization: TODO extend the configuration files
def prepare_vol_reg(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
    """Prepares the datasets for training and evaluation."""
    valid = 'validation' if 'validation' in datasets else 'test'
    result = {}
    for split, key in ('train', 'train'), (valid, 'eval'):
        ds = datasets[split]
        ds = ds.map(functools.partial(preprocess_vol_reg, transform_fns=transform_fns),     #
                    num_parallel_calls=AUTOTUNE)
        result[key] = ds.batch(batch_size).prefetch(AUTOTUNE)
    return result


@gin.configurable   # customization: TODO extend the configuration files
def prepare_t60_reg(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
    """Prepares the datasets for training and evaluation."""
    valid = 'validation' if 'validation' in datasets else 'test'
    result = {}
    for split, key in ('train', 'train'), (valid, 'eval'):
        ds = datasets[split]
        ds = ds.map(functools.partial(preprocess_t60_reg, transform_fns=transform_fns),     #
                    num_parallel_calls=AUTOTUNE)
        result[key] = ds.batch(batch_size).prefetch(AUTOTUNE)
    return result


@gin.configurable   # customization: TODO extend the configuration files
def prepare_test_vol_reg(datasets: Mapping[str, tf.data.Dataset],
            transform_fns=(align, loudness_normalization),
            batch_size: int = 64) -> Dict[str, tf.data.Dataset]:
    """Prepares the datasets for test."""
    result = {}
    ds = datasets["test"]
    ds = ds.map(functools.partial(preprocess_vol_reg, transform_fns=transform_fns),     #
              num_parallel_calls=AUTOTUNE)
    result["test"] = ds.batch(batch_size).prefetch(AUTOTUNE)
    return result


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


def __range_filtering_predicate(ds_tpl_dict: dict, x_name: str, x_start: float, x_end: float):
    x_val = ds_tpl_dict[x_name]
    return x_val >= x_start and x_val <= x_end


def filter_by_scalar_field_range(datasets: tf.data.Dataset, x_name: str, x_start: float, x_end: float):
    ret = {}
    for split in ["train", "validation", "test"]:
        ret[split] = datasets[split]\
            .filter(functools.partial(__range_filtering_predicate, x_name=x_name, x_start=x_start, x_end=x_end))
    return ret


def __class_filtering_predicate(ds_tpl_dict: dict, c: int):
    x_val = ds_tpl_dict["label"]
    return x_val == c


def filter_by_class(datasets: tf.data.Dataset, c: int):
    ret = {}
    for split in ["train", "validation", "test"]:
        ret[split] = datasets[split]\
            .filter(functools.partial(__class_filtering_predicate, c=c))
    return ret
