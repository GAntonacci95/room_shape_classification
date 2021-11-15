from refactors.ga_hyper.options import options, options_phase0
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Add, GlobalAveragePooling2D
import pprint

pp = pprint.PrettyPrinter(indent=4)
import numpy as np
import wandb
from wandb.keras import WandbCallback


class Organism:
    def __init__(self,
                 chromosome={},
                 phase=0,
                 prevBestOrganism=None):
        '''
        chromosome is a dictionary of genes
        phase is the phase that the individual belongs to
        prevBestOrganism is the best organism of the previous phase
        '''
        self.phase = phase
        self.chromosome = chromosome
        self.prevBestOrganism = prevBestOrganism
        if phase != 0:
            # In a later stage, the model is made by
            # attaching new layers to the prev best model
            self.last_model = prevBestOrganism.model

    def build_model(self, input_shape, n_classes):
        '''
        This is the function to build the keras model
        '''
        keras.backend.clear_session()
        inputs = Input(shape=(*input_shape, 1))
        inter_inputs = None
        if self.phase != 0:
            # Slice the prev best model
            # Use the model as a layer
            # Attach new layer to the sliced model
            intermediate_model = Model(inputs=self.last_model.input,
                                       outputs=self.last_model.layers[-3].output)
            for layer in intermediate_model.layers:
                # To make the iteration efficient
                layer.trainable = False
            inter_inputs = intermediate_model(inputs)
            x = Conv2D(filters=self.chromosome['a_output_channels'],
                       padding='same',
                       kernel_size=self.chromosome['a_filter_size'],
                       use_bias=self.chromosome['a_include_BN'])(inter_inputs)
            # This is to ensure that we do not randomly chose anothere activation
            self.chromosome['activation_type'] = self.prevBestOrganism.chromosome['activation_type']
        else:
            # For PHASE 0 only
            # input layer
            x = Conv2D(filters=self.chromosome['a_output_channels'],
                       padding='same',
                       kernel_size=self.chromosome['a_filter_size'],
                       use_bias=self.chromosome['a_include_BN'])(inputs)
        if self.chromosome['a_include_BN']:
            x = BatchNormalization()(x)
        x = self.chromosome['activation_type']()(x)
        if self.chromosome['include_pool']:
            x = self.chromosome['pool_type'](strides=(1, 1),
                                             padding='same')(x)
        if self.phase != 0 and self.chromosome['include_layer'] == False:
            # Except for PHASE0, there is a choice for
            # the number of layers that the model wants
            if self.chromosome['include_skip']:
                y = Conv2D(filters=self.chromosome['a_output_channels'],
                           kernel_size=(1, 1),
                           padding='same')(inter_inputs)
                x = Add()([y, x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(n_classes, activation='softmax')(x)
        else:
            # PHASE0 or no skip
            # in the tail
            x = Conv2D(filters=self.chromosome['b_output_channels'],
                       padding='same',
                       kernel_size=self.chromosome['b_filter_size'],
                       use_bias=self.chromosome['b_include_BN'])(x)
            if self.chromosome['b_include_BN']:
                x = BatchNormalization()(x)
            x = self.chromosome['activation_type']()(x)
            if self.chromosome['include_skip']:
                y = Conv2D(filters=self.chromosome['b_output_channels'],
                           padding='same',
                           kernel_size=(1, 1))(inputs)
                x = Add()([y, x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(n_classes, activation='softmax')(x)
        self.model = Model(inputs=[inputs], outputs=[x])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fitnessFunction(self,
                        train_ds,
                        test_ds,
                        generation_number):
        '''
        This function is used to calculate the
        fitness of an individual.
        '''
        # TODO: REMOVE UNAME
        wandb.init(entity="pestenera95",
                   project="vlga",
                   group='KAGp{}'.format(self.phase),
                   job_type='g{}'.format(generation_number))
        self.model.fit(train_ds,
                       epochs=3,
                       callbacks=[WandbCallback()],
                       verbose=0)
        _, self.fitness = self.model.evaluate(test_ds,
                                              verbose=0)

    def crossover(self,
                  partner,
                  generation_number,
                  train_ds, test_ds, input_shape, n_classes):
        '''
        This function helps in making children from two
        parent individuals.
        '''
        child_chromosome = {}
        endpoint = np.random.randint(low=0, high=len(self.chromosome))
        for idx, key in enumerate(self.chromosome):
            if idx <= endpoint:
                child_chromosome[key] = self.chromosome[key]
            else:
                child_chromosome[key] = partner.chromosome[key]
        child = Organism(chromosome=child_chromosome, phase=self.phase, prevBestOrganism=self.prevBestOrganism)
        child.build_model(input_shape, n_classes)
        child.fitnessFunction(train_ds,
                              test_ds,
                              generation_number=generation_number)
        return child

    def mutation(self, generation_number,
                 train_ds, test_ds, input_shape, n_classes):
        '''
        One of the gene is to be mutated.
        '''
        index = np.random.randint(0, len(self.chromosome))
        key = list(self.chromosome.keys())[index]
        if self.phase != 0:
            self.chromosome[key] = options[key][np.random.randint(len(options[key]))]
        else:
            self.chromosome[key] = options_phase0[key][np.random.randint(len(options_phase0[key]))]
        self.build_model(input_shape, n_classes)
        self.fitnessFunction(train_ds,
                             test_ds,
                             generation_number=generation_number)

    def show(self):
        '''
        Util function to show the individual's properties.
        '''
        pp.pprint(self.chromosome)
