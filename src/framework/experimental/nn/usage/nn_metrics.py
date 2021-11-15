# TODO: riferire a https://keras.io/guides/training_with_built_in_methods/#custom-metrics

import keras.backend as K
import sys


# THE FOLLOWING ARE REGRESSION METRICS
def pearsons_coeff(y_true, y_pred):
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    variance_true = K.mean(K.square(y_true - mean_true))
    variance_pred = K.mean(K.square(y_pred - mean_pred))
    covariance = K.mean((y_true - mean_true) * (y_pred - mean_pred))
    pearson_coeff = covariance / (K.sqrt(variance_true * variance_pred) + sys.float_info.epsilon)
    return pearson_coeff


# TODO: don't use in case of regression on log(V)
def mean_mult(y_true, y_pred):
    absolute_values = K.abs(K.log(y_pred / y_true))
    mean = K.mean(absolute_values)
    return K.exp(mean)
