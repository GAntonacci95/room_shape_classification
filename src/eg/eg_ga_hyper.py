from refactors.ga_hyper.generation import Generation
from tensorflow import keras
import wandb
import numpy as np
import tensorflow as tf
import pandas as p


def data_transformation(df_train: p.DataFrame, df_val: p.DataFrame, input_shape, classes):
    np.random.seed(666)
    tf.random.set_seed(666)

    X_train = np.empty([df_train.shape[0], *input_shape])
    y_train = np.empty([df_train.shape[0], 1])
    for i in range(X_train.shape[0]):
        X_train[i, :, :] = df_train.X.values[i]
        y_train[i, :] = classes.index(df_train.y.values[i])
    X_test = np.empty([df_val.shape[0], *input_shape])
    y_test = np.empty([df_val.shape[0], 1])
    for i in range(X_test.shape[0]):
        X_test[i, :, :] = df_val.X.values[i]
        y_test[i, :] = classes.index(df_val.y.values[i])

    y_train = tf.reshape(tf.one_hot(y_train, len(classes)), shape=(-1, len(classes)))
    y_test = tf.reshape(tf.one_hot(y_test, len(classes)), shape=(-1, len(classes)))

    # Create TensorFlow dataset
    BATCH_SIZE = 64
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1024).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, test_ds


def entry_point(df_train, df_val, input_shape, classes):
    population_size = 10
    number_generation = 3

    fitSurvivalRate = 0.5
    unfitSurvivalProb = 0.2
    mutationRate = 0.1
    number_of_phases = 5

    prevBestOrganism = None
    train_ds, test_ds = data_transformation(df_train, df_val, input_shape, classes)
    # TODO: REMOVE API-KEY
    wandb.login(anonymous="allow", key="d48bdd1d3f43843f6f83f2db4443b70a73fc4a13")

    for phase in range(number_of_phases):
        print("PHASE {}".format(phase))
        generation = Generation(fitSurvivalRate=fitSurvivalRate,
                                unfitSurvivalProb=unfitSurvivalProb,
                                mutationRate=mutationRate,
                                population_size=population_size,
                                phase=phase,
                                prevBestOrganism=prevBestOrganism,
                                train_ds=train_ds, test_ds=test_ds, input_shape=input_shape, n_classes=len(classes))
        while generation.generation_number < number_generation:
            print("GENERATION {}".format(generation.generation_number))
            generation.generate(train_ds, test_ds, input_shape, len(classes))
            if generation.generation_number == number_generation:
                # Last generation is the phase
                # print('I AM THE BEST IN THE PHASE')
                prevBestOrganism = generation.evaluate(last=True)
                keras.utils.plot_model(prevBestOrganism.model, to_file='./datasets/tmp/best.png', show_shapes=True)
                # caricamento del file sulla board
                wandb.log({"best_model": [wandb.Image('./datasets/tmp/best.png', caption="Best Model")]})
            else:
                generation.evaluate()
