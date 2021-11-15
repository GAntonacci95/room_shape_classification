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

"""The main script to run the training loop."""

from typing import Sequence

from absl import app
from absl import flags
import gin
from example import train, mytest
import os
# refs wrt leaf-audio project
# https://www.tensorflow.org/datasets/add_dataset
# invocation:
# ssh gantonacci@dirac or ssh gantonacci@hack
# cd /nas/home/gantonacci/Thesis/Project/pythonProject/src/refactors/leaf-audio/new_dataset_interface
# tfds new (ds_name)
# TODO: dataset registration
# import new_dataset_interface.rev_voices_leaf_trial
# import new_dataset_interface.rev_voices_fdrscc_leaf_trial
# import new_dataset_interface.rev_noises_leaf_trial
# import new_dataset_interface.rev_voices_sensed_o3_ray_leaf_trial
# import new_dataset_interface.rev_noises_sensed_o3_ray_leaf_trial
import new_dataset_interface.rev_voices_sensed_o10_ray_leaf_trial
import new_dataset_interface.rev_noises_sensed_o10_ray_leaf_trial
import new_dataset_interface.rev_rirs_sensed_o10_ray_leaf_trial
# dataset cache clean-up:
# rm -r ~/tensorflow_datasets/(ds_name)

# invocation:
# ssh gantonacci@gpu_machine
# cd /nas/home/gantonacci/Thesis/Project/pythonProject/src/refactors/leaf-audio/
# a) python -m example.main --gin_config=example/rev_voice_configs/leaf.gin
# b) python -m example.main --gin_config=example/rev_voice_fdrsCC_configs/leaf.gin
# c) python -m example.main --gin_config=example/rev_noise_configs/leaf.gin
# sensed?
# d) python -m example.main --gin_config=example/rev_voice_sensed_configs/leaf.gin
# e) python -m example.main --gin_config=example/rev_noise_sensed_configs/leaf.gin
# ACTUALLY IT'S BETTER TO CONFIGURE A FURTHER DBG ENVIRONMENT (interpreter + running configuration)
# \w running configuration as
# Script path: /nas/home/gantonacci/Thesis/Project/pythonProject/src/refactors/leaf-audio/example/main.py
# Parameters: --gin_config=/nas/home/gantonacci/Thesis/Project/pythonProject/src/refactors/leaf-audio/example/rev_voice_sensed_configs/leaf.gin
# Parameters: --gin_config=/nas/home/gantonacci/Thesis/Project/pythonProject/src/refactors/leaf-audio/example/.../leaf.gin
# Parameters: --gin_config=/nas/home/gantonacci/Thesis/Project/pythonProject/src/refactors/leaf-audio/example/.../leaf.gin


flags.DEFINE_multi_string(
    'gin_config', [], 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Newline separated list of Gin parameter bindings.')
FLAGS = flags.FLAGS


def gpus_prepare_memory() -> None:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print("{} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # memory growth must be set before GPUs have been initialized
            print(e)
    return


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # if on euler '0' or '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    gpus_prepare_memory()

    gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)

    # SHAPE CLASSIFICATION
    # train.train()
    # test: where workdir and architecture are the same of the training and the weights have to be loaded from workdir
    # to create the model before proceeding with the prediction and the confusion_matrix test report
    # TODO run train and test separately to prevent a double dataset loading
    # mytest.test()

    # train.train_x_bands()
    # mytest.test_x_bands()

    # VOLUME REGRESSION
    # train.train_vol_reg_per_class()
    # mytest.test_vol_reg_per_class()
    # train.train_vol_reg_x_bands()
    # mytest.test_vol_reg_x_bands()
    # T60 REGRESSION - VA DA SCHIFO, YAHOO.
    # train.train_t60_reg_per_class()
    return


if __name__ == '__main__':
    app.run(main)
