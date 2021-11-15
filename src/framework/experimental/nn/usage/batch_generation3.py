import keras
import pandas as p

import numpy as np
from typing import List
import keras as k


# da Castelnuovo e
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class BatchGenerator3(keras.utils.Sequence):
    def __init__(self, subset: p.DataFrame, dim,
                 batch_size=64, n_channels=1, shuffle=True):
        self.__subset = subset
        self.__dim = dim
        self.__batch_size = batch_size
        self.__n_channels = n_channels
        self.__shuffle = shuffle
        # # lo shuffle iniziale è omesso per conservare il riferimento al file riverberante
        # # per il plot dell'heatmap di test
        # self.on_epoch_end()
        return

    def on_epoch_end(self):
        if self.__shuffle:
            self.__subset = self.__subset.sample(frac=1)

    def __len__(self):
        return int(np.floor(self.__subset.shape[0] / self.__batch_size))

    def __getitem__(self, i):   # i = batch_index
        # i < __len__ -> some samples are discarded to prevent the indexing exception
        subset_batch = self.__subset.iloc[i * self.__batch_size:(i + 1) * self.__batch_size, :]
        return self.__batch_data_generation(subset_batch)

    def __batch_data_generation(self, subset_batch: p.DataFrame):
        X = np.empty((self.__batch_size, *self.__dim, self.__n_channels), dtype=np.float)
        y = np.empty((self.__batch_size, 1), dtype=np.float)

        for i in range(X.shape[0]):
            X[i, :, :, 0] = subset_batch.X.values[i]
            y[i, 0] = np.round(subset_batch.y.values[i], 4)

        return X, y


def get_batch_generators3(training_set, val_set, test_set, map_shape, batch_size = 64) -> any:
    training_gen = BatchGenerator3(training_set, map_shape, batch_size=batch_size)
    val_gen = BatchGenerator3(val_set, map_shape, batch_size=batch_size)
    test_gen = BatchGenerator3(test_set, map_shape, batch_size=batch_size)
    return training_gen, val_gen, test_gen
