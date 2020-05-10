#!/usr/bin/env python
# coding: utf-8

import random
import sklearn                                                # for neural networks and performance metrics
import numpy as np                                            # for math and data operations
import string                                                 # for converting the genomes to meaningful information
from pyeasyga import pyeasyga                                 # the genetic algorithms library
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle


warnings.filterwarnings("ignore")


def create_individual(data):                               # Each genome is a half byte representing the size of a hidden layer,
    individual = [random.randint(0, 1) for _ in range(20)]   # with each neural network having a maximum of 5 hidden layers
    for i in range(len(individual)):
        if not((i+1) % 4):
            individual[i] = int(random.randint(0, int(i/4)) != 0) if not(individual[i]) else individual[i]
    return individual
# create_individual function overwrites the default to minimize the odds that bits starting genomes of hidden layers will be 0,
# thereby reducing the odds of missing hidden layers in chromosomes of the initial population with minimal increase of bias to
# deeper networks. Odds of a genome-starting bit being replaced with a 1 are highest for layers closest to the start of the
# network, as the rest of the network construction throws out all layers right of a missing hidden layer.


def evaluate_genome(genome):     # Iterates through the list of bits of a genome and evaluates them as a binary number
    value = "0B"
    for bit in genome:
        value += str(bit)
    return int(value, 2)


def fitness(individual, data): #builds a neural network according to the genomes representing the architecture and tests
    fitness = 0                # the neural network
    genomes = [evaluate_genome(individual[0:4]),
               evaluate_genome(individual[4:8]),
               evaluate_genome(individual[8:12]),
               evaluate_genome(individual[12:16]),
               evaluate_genome(individual[16:20])]
    
    if genomes[0]:    # a series of checks to confirm that the shape represented by the genome is valid
        if genomes[1]: # will only construct a neural network using valid genomes from the left end of the chromosome
            if genomes[2]:
                if genomes[3]:
                    if genomes[4]:
                        neuralNetwork = MLPClassifier(hidden_layer_sizes=(genomes[0], genomes[1], genomes[2], genomes[3], genomes[4]), activation='tanh', solver='lbfgs', max_iter=100)
                        propagationComplexity = genomes[0] * genomes[1] * genomes[2] * genomes[3] * genomes[4]
                    else:
                        neuralNetwork = MLPClassifier(hidden_layer_sizes=(genomes[0], genomes[1], genomes[2], genomes[3]), activation='tanh', solver='lbfgs', max_iter=100)
                        propagationComplexity = genomes[0] * genomes[1] * genomes[2] * genomes[3]
                else:
                    neuralNetwork = MLPClassifier(hidden_layer_sizes=(genomes[0], genomes[1], genomes[2]), activation='tanh', solver='lbfgs', max_iter=100)
                    propagationComplexity = genomes[0] * genomes[1] * genomes[2]
            else:
                neuralNetwork = MLPClassifier(hidden_layer_sizes=(genomes[0], genomes[1]), activation='tanh', solver='lbfgs', max_iter=100)
                propagationComplexity = genomes[0] * genomes[1]
        else:
            neuralNetwork = MLPClassifier(hidden_layer_sizes=(genomes[0]), activation='tanh', solver='lbfgs', max_iter=100)
            propagationComplexity = genomes[0]
    else:
        return fitness
    X_train, X_test, Y_train, Y_test = train_test_split(data[0], data[1], test_size=0.4)
    neuralNetwork.fit(X_train, np.ravel(Y_train))
    accuracy = neuralNetwork.score(X_test, np.ravel(Y_test))
    
    accuracy = accuracy + (1 / propagationComplexity) if accuracy == 1.0 else accuracy
    return accuracy**2


def startGenetics(X, Y, initial_population=100, generations=200):
    genetics = pyeasyga.GeneticAlgorithm([X, Y],  # It is the user's responsibility to provide clean and already-formatted
                               population_size=initial_population,
                               generations=generations,
                               crossover_probability=0.00, # As the chromosomes are prioritized left to right, crossver along
                               mutation_probability=0.5,  # this dimension does not provide significant use. An elitists
                               elitism=True,               # approach with high initial population and mutation is used instead.
                               maximise_fitness=True)
    
                               
    genetics.create_individual = create_individual  # data. The genetic algorithm will not modify the training data in
    genetics.fitness_function = fitness             # any way.
    print("--PARAMETERS FOR EVOLUTION INITIALIZED | STARTING EVOLUTION--")
    genetics.run()
    solution = []
    phenotype = [evaluate_genome(genetics.best_individual()[1][0:4]),
            evaluate_genome(genetics.best_individual()[1][4:8]),
            evaluate_genome(genetics.best_individual()[1][8:12]),       # generates the phenotype
            evaluate_genome(genetics.best_individual()[1][12:16]),
            evaluate_genome(genetics.best_individual()[1][16:20])]    
    
    for i in range(4):
        if not(phenotype[i]):
            break
        solution.append(phenotype[i])           # adjusts the phenotype to a valid shape for an MLPClassifier object
    ann_file = open("evolved_network", "wb+")
    print("--EVOLUTION COMPLETE | FITTING NETWORK OF OPTIMAL COMPLEXITY TO DATASET--")
    evolved_network = MLPClassifier(hidden_layer_sizes=(tuple(solution)), activation='tanh', solver='lbfgs', max_iter=1000)
    evolved_network.fit(X, Y)
    print("--CONVERGENCE OF OPTIMAL NETWORK COMPLETE | PICKLING NETWORK TO \"evolved_network\"--")
    pickle.dump(evolved_network, ann_file)
    print("--MODEL PRESERVATION COMPLETE | CLOSING ANN FILE--")
    ann_file.close()
    return tuple(solution)

