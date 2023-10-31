from observer import observer
from random import random, uniform
import numpy as np
import controlTheory2 as ct


class individual(observer):
    def __init__(self, ID):
        self.ID = ID

    def init(self, config, best_lin_obs=None):
        self.C = config["C"]
        self.C1 = config["C1"]
        # A, poles, G, invAC
        self.__initA(config)
        self.__initPoles(config)
        self.G = ct.placeObserverPoles(self.A, self.C, self.poles)
        foo = self.A - np.matmul(self.G, self.C)
        while not ct.isHurwitzMatrix(foo):  # TODO: can last forever
            self.__initA(config)
            self.__initPoles(config)
            self.G = ct.placeObserverPoles(self.A, self.C, self.poles)
            foo = self.A - np.matmul(self.G, self.C)

        AC = self.A - np.matmul(self.G, self.C)
        self.invAC = np.linalg.inv(AC)

        # draw p1 p2 n1 n2
        self.n1 = uniform(config['min_n'], config['max_n'])
        self.n2 = uniform(config['min_n'], config['max_n'])
        self.p1 = uniform(0, 0.001)  # TODO magic numbers
        self.p2 = uniform(0, 0.001)

        self.hmN = config["hmN"]

        # create observer
        super().__init__(config)

    def __initA(self, config):
        """ matrix A is a matrix of random values from the range defined in config """
        n = config["sysRank"]
        self.A = np.random.uniform(low=config["minA"], high=config["maxA"], size=(n, n))
        # matrix A must be Hurwizowska, keep drawing until you find one!
        # TODO: with an inappropriate choice of range in config can last forever
        while not ct.isHurwitzMatrix(self.A) or not ct.kalmanObservability(self.A,
                                                                           config["C"]):
            self.A = np.random.uniform(low=config["minA"], high=config["maxA"], size=(n, n))

    def __initPoles(self, config):
        """ poles are limited by the assumption that Im(s) == 0 """
        n = config["sysRank"]
        self.poles = np.random.uniform(low=config["minPoleRe"],
                                       high=config["maxPoleRe"], size=(1,n))
        self.poles.sort()

    def crossover(self, parent1, parent2, config):
        self.C = config["C"]
        self.C1 = config["C1"]

        # cross matrix A
        self.A = self.__crossFloats(parent1.A, parent2.A)
        # cross poles
        self.poles = self.__crossFloats(parent1.poles, parent2.poles)
        self.G = ct.placeObserverPoles(self.A, self.C, self.poles)

        AC = self.A - np.matmul(self.G, self.C)
        self.invAC = np.linalg.inv(AC)

        # cross p1 p2 n1 n2 hmN
        self.n1 = self.__crossFloats(parent1.n1, parent2.n1)
        self.n2 = self.__crossFloats(parent1.n2, parent2.n2)
        self.p1 = self.__crossFloats(parent1.p1, parent2.p1)
        self.p2 = self.__crossFloats(parent1.p2, parent2.p2)

        self.hmN = config['hmN']

        # create observer
        super().__init__(config)

        # cross weights
        self.net.V = self.__crossFloats(parent1.net.V, parent2.net.V)
        self.net.W = self.__crossFloats(parent1.net.W, parent2.net.W)

    def __crossFloats(self, val1, val2):
        try:
            r = np.random.rand(val1.shape[0], val1.shape[1])
        except (AttributeError, IndexError):
            r = random()
        return r*val1 + (1-r)*val2

    def mutate(self, config):
        """ Mutates individual.
        If one mutation mode is enabled, one mutation is bound to happen.
        If not then it may not happen"""
        if config["singleMutation"]:
            r = uniform(0, config["mutationAProb"] + config["mutationPolesProb"] +
                        config["mutationObserverParamProb"] +
                        config["mutationWeights"])
            if r < config["mutationAProb"]:
                self.__mutateA(config)
            elif r < config["mutationAProb"] + config["mutationPolesProb"]:
                self.__mutatePoles(config)
            elif r < config["mutationAProb"] + config["mutationPolesProb"] +\
                    config["mutationWeights"]:
                self.__mutateWeights(config)
            else:
                self.__mutateObserverParam(config)

        else:  # if single mutation mode is not enabled
            if random() < config["mutationAProb"]:
                self.__mutateA(config)

            if random() < config["mutationWeights"]:
                self.__mutateWeights(config)

            if random() < config["mutationPolesProb"]:
                self.__mutatePoles(config)

            if random() < config["mutationObserverParamProb"]:
                self.__mutateObserverParam(config)

    def __mutateFloat(self, val, percentageChange):
        try:
            r = np.random.uniform(1-percentageChange, 1+percentageChange,
                                  (val.shape[0], val.shape[1]))
        except (AttributeError, IndexError):
            r = np.random.uniform(1-percentageChange, 1+percentageChange)
        return val*r

    def __mutateWeights(self, config):
        self.net.W = self.__mutateFloat(self.net.W, config["maxMutationFloatRange"])
        self.net.V = self.__mutateFloat(self.net.V, config["maxMutationFloatRange"])

    def __mutateA(self, config):
        self.A = self.__mutateFloat(self.A, config["maxMutationFloatRange"])

    def __mutatePoles(self, config):
        self.poles = self.__mutateFloat(self.poles, config["maxMutationFloatRange"])
        self.poles.sort()

    def __mutateObserverParam(self, config):
        self.n1 = self.__mutateFloat(self.n1, config["maxMutationFloatRange"])
        # self.n2 = self.n1
        self.n2 = self.__mutateFloat(self.n2, config["maxMutationFloatRange"])

        self.p1 = self.__mutateFloat(self.p1, config["maxMutationFloatRange"])
        # self.p2 = self.p1
        self.p2 = self.__mutateFloat(self.p2, config["maxMutationFloatRange"])
