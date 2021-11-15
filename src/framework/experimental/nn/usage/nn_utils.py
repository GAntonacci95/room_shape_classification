from keras import Model
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# def gpus_phys_count() -> int:
#     return len(tf.config.experimental.list_physical_devices("GPU"))
#

# TODO: riferire a https://keras.io/guides/distributed_training/#singlehost-multidevice-synchronous-training
#  per la parallelizzazione sulle GPU
# metodo di preparazione per il controllo della crescita della memoria VRAM
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


# def gpus_prepare_memory2() -> None:
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         # Create 2 virtual GPUs with 1GB memory each
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_virtual_device_configuration(gpu,
#                     # per ogni disp fisico preparo due disp virtuali da 4GB
#                     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096) for i in range(0, 2)])
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print("{} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
#         except RuntimeError as e:
#             # Virtual devices must be set before GPUs have been initialized
#             print(e)
#     return


# TODO: riferire a:
#  https://keras.io/guides/training_with_built_in_methods/#checkpointing-models ed a
#  https://keras.io/guides/training_with_built_in_methods/#using-the-tensorboard-callback
#  https://keras.io/guides/distributed_training/#using-callbacks-to-ensure-fault-tolerance
#  per migliorare la gestione dei checkpoints
#  secondo https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/keras/Model#fit
#  x accetta un generatore, a quel punto y diviene facoltativo
def train(architecture: Model, training_gen: Sequence, val_gen: Sequence,
          base_model_dir: str, last_epoch: int, train_params: {} = None) -> any:
    red_factor, red_patience, delta = 0.2, 10, 1E-3
    early_patience = 20
    if train_params:
        red_factor, red_patience, delta = \
            train_params["red_factor"], train_params["red_patience"], train_params["delta"]
        early_patience = train_params["early_patience"]

    # returns the training history with loss and metrics data
    return architecture.fit(
        x=training_gen,
        validation_data=val_gen,
        epochs=1000,    # EarlyStopping after LR-Reduction
        # use_multiprocessing=True,     # False by default to avoid deadlocks
        workers=8,
        # la callback crea nuovi file di checkpoint nella cartella base_model_dir/epoca-...
        # (di default ad ogni epoca)
        callbacks=[ModelCheckpoint(base_model_dir + "/epoca-{epoch:4d}"),
                   ReduceLROnPlateau(monitor="val_loss",
                                     factor=red_factor, patience=red_patience,   min_delta=delta, min_lr=1E-10),
                   EarlyStopping(    monitor="val_loss",patience=early_patience, min_delta=delta)
                   ],
        initial_epoch=last_epoch
    )
