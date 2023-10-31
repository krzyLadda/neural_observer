import individual as indLib
import random
import numpy as np
import time
import itertools
import reporting
import copy


class population():
    def __init__(self, config, pop=None, best=None, rep=None):
        np.random.seed()
        random.seed()
        self.config = config

        if pop is not None:
            self.fitness_function = None
            self.inds = pop
            self.best = best
            self.generation = len(rep.genBest)
            # remove last report from self.reporter because last population is accutal
            rep.genBest.pop()
            self.reporter = rep
            # find the max id of an individual in all generations
            maxID = 0
            for gen in rep.genInds:
                maxID = int(max([maxID] + list(gen.keys())))
            self.__indsIndexer = itertools.count(maxID)

        else:  # create a new population
            # individual counter
            self.__indsIndexer = itertools.count(0)
            self.generation = 0
            # create an appropriate number of individuals
            self.inds = {}
            for i in range(self.config["popSize"]):
                # create ind
                ind = indLib.individual(next(self.__indsIndexer))
                # initialise the characteristics of an individual
                ind.init(self.config)
                # save it to a collection of individuals
                self.inds[ind.ID] = ind

            self.best = None
            self.genBest = []
            self.genFitnesses = []

    def crossover(self):
        childs = {}
        # draw a set of parents
        parents = [ID for ID in self.inds.keys() if random.random() < self.config['pCross']]
        # if there are no parents then return empty dict
        if len(parents) < 2:
            return {}
        random.shuffle(parents)
        # if there is an odd number, remove one
        if len(parents) % 2 == 1:
            parents.pop()
        while parents:
            p1 = self.inds[parents.pop()]
            p2 = self.inds[parents.pop()]
            # create offspring
            new = indLib.individual(next(self.__indsIndexer))
            # initialise the characteristics of an individual
            new.crossover(p1, p2, self.config)
            childs[new.ID] = new
        return childs

    def mutate(self):
        mutants = {}
        for ind in self.inds.values():
            if random.random() < self.config['pMutation']:
                new = copy.deepcopy(ind)
                new.ID = next(self.__indsIndexer)
                new.mutate(self.config)
                mutants[new.ID] = new
        return mutants

    def selection(self):
        # lambda + gamma
        newPop = {}
        inds = {**self.inds, **self.newInds}
        # take fitness and id
        fits = np.array([[ID, ind.fitness, 0] for (ID, ind) in inds.items()])
        fits = fits[fits[:, 1].argsort()]

        # elitism:
        if self.config['elitism']:
            elite = fits[-self.config['elitism']:, :]
            fits = fits[:-self.config['elitism'], :]
            for ID in elite[:, 0]:
                newPop[ID] = inds[ID]

        # rulet with repetitions:
        hm_inds = self.config["popSize"]-len(newPop)
        fits[:, 1] = fits[:, 1] + abs(min(fits[:, 1]))
        try:
            fits[:, 1] = fits[:, 1] / np.sum(fits[:, 1])
        # if all individuals have fit=0
        except (ZeroDivisionError, FloatingPointError):
            # create new individuals
            print(f"{hm_inds} new individuals are created in the selection.")
            created_inds = {}
            for i in range(hm_inds):
                new = indLib.individual(next(self.__indsIndexer))
                new.init(self.config)
                created_inds[new.ID] = new
            self.fitness_function(created_inds.values(), self.config)
            self.inds = newPop | created_inds
            return

        fits[0, 2] = fits[0, 1]
        for i in range(1, fits.shape[0]):
            fits[i, 2] = fits[i-1, 2] + fits[i, 1]

        # for each missing individual
        r = np.random.rand(hm_inds)
        r.sort()
        fit_idx = 0
        for r_temp in r:
            while fits[fit_idx, 2] < r_temp:
                fit_idx = fit_idx + 1
            ID = fits[fit_idx, 0]
            new = copy.deepcopy(inds[ID])
            new.ID = next(self.__indsIndexer)
            newPop[new.ID] = new
        self.inds = newPop

    def run(self, fitness_function, n=None, startTime=time.time(), setTime=None,
            fitnessTermination=None, config=None):
        self.fitness_function = fitness_function

        if config is not None:
            self.config = config

        # create reporter
        if not hasattr(self, 'reporter'):
            self.reporter = reporting.reporter(self.config, self.generation)

        # check that termination condition is set?
        if n is None and setTime is None and fitnessTermination is None:
            raise RuntimeError(
                "The termination condition is not set.")

        durationTime = time.time() - startTime

        # Evaluate all genomes using the user-provided function.
        self.fitness_function(self.inds.values(), self.config)

        # main loop of EA
        while ((n is None or self.generation < n) and
               (setTime is None or durationTime < setTime) and
               (fitnessTermination is None)):

            loop_time = time.time()
            self.genFitnesses = [ind.fitness for ind in self.inds.values()]

            # find the best ind from this generation
            genBest = max(self.inds.values(), key=lambda x: x.fitness)
            # save the best ind at all
            if self.best is None or self.best.fitness < genBest.fitness:
                self.best = genBest

            self.reporter.report(self.inds, genBest)

            # if has reached fitnes terminantion then terminate
            if fitnessTermination is not None and \
                    self.best.fitness >= fitnessTermination:
                return self.best, self.reporter

            # create new individuals from crossing and mutation
            childs = self.crossover()
            mutants = self.mutate()

            # evaluate them
            self.newInds = {**childs, **mutants}
            self.fitness_function(self.newInds.values(), self.config)
            # retrain_inds = copy.deepcopy(self.inds)
            # for ind in retrain_inds.values():
            #     newID = next(self.__indsIndexer)
            #     ind.ID = newID
            #     self.newInds[newID] = ind
            # self.fitness_function(self.newInds.values(), self.config)

            # select new population
            self.selection()
            self.generation += 1

            loop_time = time.time() - loop_time
            print(f'{self.generation}. Loop time: {loop_time:.2f} Best: {self.best.ID}:{self.best.fitness}')

            durationTime = time.time() - startTime

        return self.best, self.reporter
