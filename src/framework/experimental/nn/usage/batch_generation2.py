import keras
import pandas as p
import numpy as np
import keras as k
# from scipy.stats import zscore  # zscore(data(n,w), axis=1, ddof=1)
from framework.experimental.nn.features.post_processing.nnpostprocutils import min_max_scaling


# da Castelnuovo e
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class BatchGenerator2(keras.utils.Sequence):
    def __init__(self, subset: p.DataFrame, dim, n_classes: int, b_norm: bool = True, only_t_feats_dbg: bool = False,
                 batch_size=64, n_channels=1, shuffle=True):
        self.__subset = subset
        self.__n_classes = n_classes
        self.__dim = dim
        self.__b_norm = b_norm
        self.__only_t_feats_dbg = only_t_feats_dbg
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

    def subset_batch(self, i):  # contains more info wrt generator[i] indexing
        # i < __len__ -> some samples are discarded to prevent the indexing exception
        return self.__subset.iloc[i * self.__batch_size:(i + 1) * self.__batch_size, :]

    def __batch_data_generation(self, subset_batch: p.DataFrame):
        X = np.empty((self.__batch_size, *self.__dim, self.__n_channels), dtype=np.float)
        y = np.empty((self.__batch_size, 1), dtype=np.int)

        for n in range(X.shape[0]):
            if not self.__only_t_feats_dbg:
                X[n, :, :, 0] = subset_batch.X.values[n]
            else:
                X[n, :, :, 0] = np.delete(subset_batch.X.values[n], [20, 21, 22], 0)
            y[n, 0] = subset_batch.class_id.values[n]
        if self.__b_norm:
            # fisso (h=feature, c) normalizzo vettori (1, W) lungo n
            # confido sia accettabile farlo sul batch e non sul dataset intero
            for h in range(X.shape[1]):
                X[:, h, :, 0] = min_max_scaling(X[:, h, :, 0])

        return X, k.utils.to_categorical(y, self.__n_classes)

    def __getitem__(self, i):   # i = batch_index
        return self.__batch_data_generation(self.subset_batch(i))


def get_batch_generators2(training_set, val_set, test_set, map_shape, n_classes: int,
                          b_norm: bool = True, only_t_feats_dbg: bool = False) -> any:
    training_gen = BatchGenerator2(training_set, map_shape, n_classes, b_norm, only_t_feats_dbg)
    val_gen = BatchGenerator2(val_set, map_shape, n_classes, b_norm, only_t_feats_dbg)
    test_gen = BatchGenerator2(test_set, map_shape, n_classes, b_norm, only_t_feats_dbg)
    return training_gen, val_gen, test_gen
