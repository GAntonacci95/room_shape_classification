from tensorflow.keras.layers import MaxPool2D, ReLU, ELU, LeakyReLU, AveragePooling2D
import numpy as np


options_phase0 = {
    'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'a_include_BN': [True, False],
    'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'activation_type': [ReLU, ELU, LeakyReLU],
    'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'b_include_BN': [True, False],
    'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'include_pool': [True, False],
    'pool_type': [MaxPool2D, AveragePooling2D],
    'include_skip': [True, False]
}


options = {
    'include_layer': [True, False],
    'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'a_include_BN': [True, False],
    'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'b_include_BN': [True, False],
    'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'include_pool': [True, False],
    'pool_type': [MaxPool2D, AveragePooling2D],
    'include_skip': [True, False]
}


def random_hyper(phase):
    if phase == 0:
        return {
        'a_filter_size': options_phase0['a_filter_size'][np.random.randint(len(options_phase0['a_filter_size']))],
        'a_include_BN': options_phase0['a_include_BN'][np.random.randint(len(options_phase0['a_include_BN']))],
        'a_output_channels': options_phase0['a_output_channels'][np.random.randint(len(options_phase0['a_output_channels']))],
        'activation_type': options_phase0['activation_type'][np.random.randint(len(options_phase0['activation_type']))],
        'b_filter_size': options_phase0['b_filter_size'][np.random.randint(len(options_phase0['b_filter_size']))],
        'b_include_BN': options_phase0['b_include_BN'][np.random.randint(len(options_phase0['b_include_BN']))],
        'b_output_channels': options_phase0['b_output_channels'][np.random.randint(len(options_phase0['b_output_channels']))],
        'include_pool': options_phase0['include_pool'][np.random.randint(len(options_phase0['include_pool']))],
        'pool_type': options_phase0['pool_type'][np.random.randint(len(options_phase0['pool_type']))],
        'include_skip': options_phase0['include_skip'][np.random.randint(len(options_phase0['include_skip']))]
        }
    else:
        return {
        'a_filter_size': options['a_filter_size'][np.random.randint(len(options['a_filter_size']))],
        'a_include_BN': options['a_include_BN'][np.random.randint(len(options['a_include_BN']))],
        'a_output_channels': options['a_output_channels'][np.random.randint(len(options['a_output_channels']))],
        'b_filter_size': options['b_filter_size'][np.random.randint(len(options['b_filter_size']))],
        'b_include_BN': options['b_include_BN'][np.random.randint(len(options['b_include_BN']))],
        'b_output_channels': options['b_output_channels'][np.random.randint(len(options['b_output_channels']))],
        'include_pool': options['include_pool'][np.random.randint(len(options['include_pool']))],
        'pool_type': options['pool_type'][np.random.randint(len(options['pool_type']))],
        'include_layer': options['include_layer'][np.random.randint(len(options['include_layer']))],
        'include_skip': options['include_skip'][np.random.randint(len(options['include_skip']))]
        }
