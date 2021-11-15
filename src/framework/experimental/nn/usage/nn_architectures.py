from keras.models import Sequential
from keras.layers import Input, Conv2D, Dropout, AveragePooling2D, MaxPool2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam, SGD

import keras
import keras.layers as layers


# functions returning the initial compiled model
# based upon A. F. Genovese Paper
def __agenovese(params) -> Sequential:
    map_shape = params["map_shape"]
    conv_filter_sizes = [ 30,      20,      10,      10,      5,      5]
    conv_ker_sizes =    [(1, 10), (1, 10), (1, 10), (1, 10), (3, 9), (3, 9)]
    conv_ker_stride =    (1, 1)
    avgpool_sizes =     [(1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (2, 2)]
    avgpool_strides =   avgpool_sizes

    architecture = Sequential()
    architecture.add(Input(shape=(*map_shape, 1)))
    for i in range(len(conv_filter_sizes)):
        architecture.add(Conv2D(filters=conv_filter_sizes[i], kernel_size=conv_ker_sizes[i], strides=conv_ker_stride,
                                activation="relu", padding="same"))
        architecture.add(AveragePooling2D(pool_size=avgpool_sizes[i], strides=avgpool_strides[i]))

    architecture.add(Dropout(0.5))
    architecture.add(Flatten())
    # TODO poteva essere effettivamente meglio specificare ReLU invece di lasciar libera l'attivazione.
    architecture.add(Dense(1))

    lr_init = 1*1E-3
    if "lr_init" in params.keys():
        lr_init = params["lr_init"]
    architecture.compile(optimizer=Adam(learning_rate=lr_init),
                         loss="mean_squared_error",
                         metrics=["mean_absolute_error", "pearsons_coeff", "mean_mult"])
    return architecture


def agenovese_wnfrmp0nofdrfeatAG_R(params) -> Sequential:
    return __agenovese(params)


def agenovese_wnfrmp0nofdrfeatAG_L(params) -> Sequential:
    return __agenovese(params)


def agenovese_wnfrmp0nofdrfeatAG_H(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_R(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_L(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_H(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_t60whole(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_R(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_L(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_H(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_vb0(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_vb2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_vb4(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_tb0(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_tb2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayrirnocutfeatAG_tb4(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_t60whole(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_vb0(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_vb2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_vb4(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_tb0(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_tb2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismraywnnofdrfeatAG_tb4(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_vb0(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_vb2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_vb4(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_vb0_try2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_vb2_try2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_vb4_try2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_tb0(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_tb2(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_tb4(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_R(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_t60whole(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_L(params) -> Sequential:
    return __agenovese(params)


def agenovese_ismrayvcnofdrfeatAG_H(params) -> Sequential:
    return __agenovese(params)


def agenovese_wnfrmp0nofdrfeatAGvolRestr_R(params) -> Sequential:
    return __agenovese(params)


def agenovese_wnfrmp0nofdrfeatAGvolRestr_L(params) -> Sequential:
    return __agenovese(params)


def __gantonacci1(params: dict) -> Sequential:
    map_shape, n_classes = params["map_shape"], params["n_classes"]
    conv_filter_sizes = [ 30,      20,      10,      10,      5,      5]
    conv_ker_sizes =    [(1, 10), (1, 10), (1, 10), (1, 10), (3, 9), (3, 9)]
    conv_ker_stride =    (1, 1)
    avgpool_sizes =     [(1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (2, 2)]
    avgpool_strides =   avgpool_sizes

    architecture = Sequential()
    architecture.add(Input(shape=(*map_shape, 1)))
    for i in range(len(conv_filter_sizes)):
        architecture.add(Conv2D(filters=conv_filter_sizes[i], kernel_size=conv_ker_sizes[i], strides=conv_ker_stride,
                                activation="relu", padding="same"))
        architecture.add(AveragePooling2D(pool_size=avgpool_sizes[i], strides=avgpool_strides[i]))

    architecture.add(Dropout(rate=0.5))
    architecture.add(Flatten())
    # 1 out neuron  with sigmoid for binary-classification
    # 2 out neurons with softmax for multi -classification
    architecture.add(Dense(n_classes, activation="softmax"))

    architecture.compile(optimizer=Adam(learning_rate=5*1E-4),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
    return architecture


def __gantonacci3(params: dict) -> Sequential:
    map_shape, n_classes = params["map_shape"], params["n_classes"]
    conv_filter_sizes = [ 30,      20,      10,      10,      5,      5]
    conv_ker_sizes =    [(1, 10), (1, 10), (1, 10), (1, 10), (3, 9), (3, 9)]
    conv_ker_stride =    (1, 1)
    avgpool_sizes =     [(1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (2, 2)]
    avgpool_strides =   avgpool_sizes

    architecture = Sequential()
    architecture.add(Input(shape=(*map_shape, 1)))
    architecture.add(BatchNormalization())  # just introduced
    for i in range(len(conv_filter_sizes)):
        architecture.add(Conv2D(filters=conv_filter_sizes[i], kernel_size=conv_ker_sizes[i], strides=conv_ker_stride,
                                activation="relu", padding="same"))
        architecture.add(AveragePooling2D(pool_size=avgpool_sizes[i], strides=avgpool_strides[i]))

    architecture.add(Dropout(rate=0.3))     # 0.5
    # https://stackoverflow.com/questions/47538391/keras-batchnormalization-axis-clarification
    architecture.add(BatchNormalization())  # just introduced
    architecture.add(Flatten())
    # 1 out neuron  with sigmoid for binary-classification
    # 2 out neurons with softmax for multi -classification
    architecture.add(Dense(n_classes, activation="softmax"))

    lr_init = 1*1E-2
    if "lr_init" in params.keys():
        lr_init = params["lr_init"]
    architecture.compile(optimizer=Adam(learning_rate=lr_init),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
    return architecture


# wrapper for data separation...
def gantonacci_wnfrmp0nofdrfeatAG(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismraywnnofdrfeatAG(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayrirnocutfeatAG(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_wnfrmp0nofdrfeatAGvolRestr(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_vcfrmp0nofdrfeatAG(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrfeatAG(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0_c0tf(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0_c1tf(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_vb0_c2tf(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_vb2(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_vb4(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_tb0(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_tb2(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_normal_tb4(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismrayvcnofdrfeatAG_SGD(params) -> Sequential:
    map_shape, n_classes = params["map_shape"], params["n_classes"]
    conv_filter_sizes = [ 30,      20,      10,      10,      5,      5]
    conv_ker_sizes =    [(1, 10), (1, 10), (1, 10), (1, 10), (3, 9), (3, 9)]
    conv_ker_stride =    (1, 1)
    avgpool_sizes =     [(1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (1, 2 ), (2, 2)]
    avgpool_strides =   avgpool_sizes

    architecture = Sequential()
    architecture.add(Input(shape=(*map_shape, 1)))
    for i in range(len(conv_filter_sizes)):
        architecture.add(Conv2D(filters=conv_filter_sizes[i], kernel_size=conv_ker_sizes[i], strides=conv_ker_stride,
                                activation="relu", padding="same"))
        architecture.add(AveragePooling2D(pool_size=avgpool_sizes[i], strides=avgpool_strides[i]))

    architecture.add(Dropout(rate=0.5))
    architecture.add(Flatten())
    # 1 out neuron  with sigmoid for binary-classification
    # 2 out neurons with softmax for multi -classification
    architecture.add(Dense(n_classes, activation="softmax"))

    architecture.compile(optimizer=SGD(learning_rate=5*1E-4),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
    return architecture


def gantonacci_vcfrmp0fdrCCfeatAG(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhwnfrmp0nofdrfeatAG(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayrirnocutnetAG_vb0(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayrirnocutnetAG_vb2(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayrirnocutnetAG_vb4(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayrirnocutnetAG_tb0(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayrirnocutnetAG_tb2(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayrirnocutnetAG_tb4(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismraywnnofdrnetAG_vb0(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismraywnnofdrnetAG_vb2(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismraywnnofdrnetAG_vb4(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismraywnnofdrnetAG_tb0(params) -> Sequential:
    return __gantonacci3(params)


def gantonacci_rlhismraywnnofdrnetAG_tb2(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismraywnnofdrnetAG_tb4(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrnetAG_vb0(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrnetAG_vb2(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrnetAG_vb4(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrnetAG_tb0(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrnetAG_tb2(params) -> Sequential:
    return __gantonacci1(params)


def gantonacci_rlhismrayvcnofdrnetAG_tb4(params) -> Sequential:
    return __gantonacci1(params)


def __gantonacci2(params) -> Sequential:
    input_shape, n_classes = params["input_shape"], params["n_classes"]
    architecture = Sequential()
    architecture.add(Input(shape=input_shape))
    architecture.add(Flatten())

    architecture.add(Dense(8192, activation="relu"))
    architecture.add(Dense(1024, activation="relu"))
    architecture.add(Dense(128, activation="relu"))
    architecture.add(Dense(n_classes, activation="softmax"))

    architecture.compile(optimizer="adam",  #Adam(learning_rate=7*1E-5),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
    return architecture


def gantonacci_rirnocutMLP(params) -> Sequential:
    return __gantonacci2(params)


def gantonacci_rircutMLP(params) -> Sequential:
    return __gantonacci2(params)


def gantonacci_rlhrirnocutMLP(params) -> Sequential:
    return __gantonacci2(params)


# keras Xception net example
def __xception(params: dict) -> keras.Model:
    (map_shape, n_classes) = (params["map_shape"], params["n_classes"])
    inputs = keras.Input(shape=(*map_shape, 1))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    architecture = keras.Model(inputs, outputs)

    architecture.compile(optimizer=Adam(learning_rate=1E-5),
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
    return architecture


def xception_wnfrmp0wbnofdrfeatAG(params) -> keras.Model:
    return __xception(params)


def xception_wnfrmp0wbnofdrfeatAGvolRestr(params) -> keras.Model:
    return __xception(params)


def xception_vcfrmp0fdrCCfeatAG(params) -> keras.Model:
    return __xception(params)


def __basic(params: dict) -> keras.Model:
    (map_shape, n_classes) = (params["map_shape"], params["n_classes"])

    model = Sequential()
    model.add(Input(shape=(*map_shape, 1)))
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(optimizer="adam",
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
    return model


def basic_wnfrmp0wbnofdrfeatAG(params) -> keras.Model:
    return __basic(params)
