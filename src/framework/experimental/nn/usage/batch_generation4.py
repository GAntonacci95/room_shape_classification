import keras
import pandas as p

import numpy as np
from typing import List
import keras as k
from framework.experimental.nn.features.post_processing.nnpostprocutils import min_max_scaling


# da Castelnuovo e
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class BatchGenerator4(keras.utils.Sequence):
    def __init__(self, subset: p.DataFrame, dim, n_classes: int,
                 batch_size=64, shuffle=True):
        self.__subset = subset
        self.__n_classes = n_classes
        self.__dim = dim
        self.__batch_size = batch_size
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

    def subset_batch(self, i):
        # i < __len__ -> some samples are discarded to prevent the indexing exception
        return self.__subset.iloc[i * self.__batch_size:(i + 1) * self.__batch_size, :]

    def __batch_data_generation(self, subset_batch: p.DataFrame):
        # MLP INPUT TENSOR: batch, time_bins, n_features
        X = np.empty((self.__batch_size, *self.__dim), dtype=np.float)
        y = np.empty((self.__batch_size, 1), dtype=np.int)

        for i in range(X.shape[0]):
            X[i, :, :] = subset_batch.X.values[i]
            y[i, 0] = subset_batch.class_id.values[i]
        # fisso (h=rir_axis) normalizzo vettori (, W) lungo n
        for h in range(X.shape[1]):
            X[:, h, :] = min_max_scaling(X[:, h, :])

        return X, k.utils.to_categorical(y, self.__n_classes)

    def __getitem__(self, i):   # i = batch_index
        return self.__batch_data_generation(self.subset_batch(i))


def get_batch_generators4(training_set, val_set, test_set, input_shape, n_classes: int) -> any:
    training_gen = BatchGenerator4(training_set, input_shape, n_classes)
    val_gen = BatchGenerator4(val_set, input_shape, n_classes)
    test_gen = BatchGenerator4(test_set, input_shape, n_classes)
    return training_gen, val_gen, test_gen
