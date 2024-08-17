#!/usr/bin/env python
# coding: utf-8


# Genetic Algorithm Level 1 Sample


import numpy as np
import matplotlib.pyplot as plt


DNA_SIZE = 10
POP_SIZE = 100
CROSS_RATE = 0.8
MUTATION_RATE = 0.003
N_GENERATIONS = 200
X_BOUND = [0, 5]

pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0)

def f(x): return np.sin(10*x)*x + np.cos(2*x)*x


def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)


def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, DNA_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool8)
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child




plt.ion()
x = np.linspace(*X_BOUND, 200)
plt.plot(x, f(x))

for _ in range(N_GENERATIONS):
    
    F_values = f(translateDNA(pop))
    
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.2)
    
    fitness = get_fitness(F_values)
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child
        
    
plt.ioff();plt.show()

