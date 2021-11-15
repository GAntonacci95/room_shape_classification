from refactors.ga_hyper.options import random_hyper
from refactors.ga_hyper.organism import Organism
import numpy as np
import wandb


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Generation:
    def __init__(self,
                 fitSurvivalRate,
                 unfitSurvivalProb,
                 mutationRate,
                 phase,
                 population_size,
                 prevBestOrganism,
                 train_ds, test_ds, input_shape, n_classes):
        self.population_size = population_size
        self.population = []
        self.generation_number = 0
        self.mutationRate = mutationRate
        self.fitSurvivalRate = fitSurvivalRate
        self.unfitSurvivalProb = unfitSurvivalProb
        self.prevBestOrganism = prevBestOrganism
        self.phase = phase
        # creating the first population: GENERATION_0
        # can be thought of as the setup function
        for idx in range(self.population_size):
            org = Organism(chromosome=random_hyper(self.phase), phase=self.phase,
                           prevBestOrganism=self.prevBestOrganism)
            org.build_model(input_shape, n_classes)
            org.fitnessFunction(train_ds,
                                test_ds,
                                generation_number=self.generation_number)
            self.population.append(org)

        # sorts the population according to fitness (high to low)
        self.sortModel()
        self.generation_number += 1

    def sortModel(self):
        '''
        sort the models according to the
        fitness in descending order.
        '''
        fitness = [ind.fitness for ind in self.population]
        sort_index = np.argsort(fitness)[::-1]
        self.population = [self.population[index] for index in sort_index]

    def generate(self, train_ds, test_ds, input_shape, n_classes):
        '''
        Generate a new generation in the same phase
        '''
        number_of_fit = int(self.population_size * self.fitSurvivalRate)
        new_pop = self.population[:number_of_fit]
        for individual in self.population[number_of_fit:]:
            if np.random.rand() <= self.unfitSurvivalProb:
                new_pop.append(individual)
        for index, individual in enumerate(new_pop):
            if np.random.rand() <= self.mutationRate:
                new_pop[index].mutation(generation_number=self.generation_number,
                                        train_ds=train_ds, test_ds=test_ds,
                                        input_shape=input_shape, n_classes=n_classes)
        fitness = np.array([ind.fitness for ind in new_pop])
        children = []
        for idx in range(self.population_size - len(new_pop)):
            parents = np.random.choice(new_pop, replace=False, size=(2,), p=softmax(fitness))
            A = parents[0]
            B = parents[1]
            child = A.crossover(B, generation_number=self.generation_number,
                                train_ds=train_ds, test_ds=test_ds,
                                input_shape=input_shape, n_classes=n_classes)
            children.append(child)
        self.population = new_pop + children
        self.sortModel()
        self.generation_number += 1

    def evaluate(self, last=False):
        '''
        Evaluate the generation
        '''
        fitness = [ind.fitness for ind in self.population]
        wandb.log({'Best fitness': fitness[0]})
        wandb.log({'Average fitness': sum(fitness) / len(fitness)})
        self.population[0].show()
        if last:
            return self.population[0]
