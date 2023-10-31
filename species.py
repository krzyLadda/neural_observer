from math import sqrt
import numpy as np
import sys


def compute_spawn(adjusted_fitness, pop_size, min_species_size):
    """Compute the proper number of offspring per species (proportional to fitness)."""
    # suma ajd_fitness gatunkow
    af_sum = sum(adjusted_fitness)

    spawn_amounts = []
    # dla adj_fitness każdego gatunku
    for af in adjusted_fitness:
        """
        AF_SUM == 0, CZY TO SIE MOŻE ZDARZYC?
        odp. może się zdarzyć gdy funcje przystosowania wszystkich osobników będą równe 0
        w moim zadaniu praktycznie nie mozliwe, ale zostawię ten if

        """
        if af_sum > 0:  # jezeli suma adj_fit jest dodatnia
            # to zmienna pomocnicza s to max z minimalnej ilosci osobnikow
            # w gatunku i liniowego skalowania ajd_fit do pop_size:
            # czyli liczba dzieci jest wprost propocjonalna do przystosowania gatunku
            # i wielkosci populacji
            s = max(min_species_size, af / af_sum * pop_size)
        else:
            # tu wejdziemy tylko gdy wszystki gatunki maja przystosowanie == 0
            s = min_species_size

        spawn_amounts.append(round(s))
    # Normalize the spawn amounts so that the next generation is roughly
    # the population size requested by the user.
    total_spawn = sum(spawn_amounts)
    norm = pop_size / total_spawn
    spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]
    return spawn_amounts


class DistanceCache():
    def __init__(self):
        self.distances = {}

    def __call__(self, ind1, ind2, config):
        ID1 = ind1.ID
        ID2 = ind2.ID
        d = self.distances.get((ID1, ID2))
        if d is None:
            # Distance is not already computed.
            d = self.distance(ind1, ind2, config)
            self.distances[ID1, ID2] = d
            self.distances[ID2, ID1] = d
        return d

    def distance(self, rep, ind, config):
        """ Liczy odległosc genetyczna osobników"""
        # róznice w macierzach A:
        # odległosc kartezjanska
        diffA = sqrt(np.sum((rep.A - ind.A)**2))
        # różnice w biegunach, bieguny POWINNY BYC posortowane rosnąco
        # TODO: będzie działać dla biegunów które mających jedynie częsć Re
        # odleglosc kartezjanska
        diffPoles = sqrt(sum((rep.poles - ind.poles)**2))
        # różnica w ilosci neuronów ukrytych
        diffHmN = abs(rep.hmN - ind.hmN)/max([ind.hmN, ind.hmN])

        # distance
        d = config["distanceWeightA"]*diffA + config["distanceWeightPoles"] *\
            diffPoles + config["distanceWeightHmN"]*diffHmN
        return d


class species():
    def __init__(self, sID, generation):
        self.ID = sID
        # self.created = generation
        self.lastImproved = generation
        self.lastImprovmentFitness = -sys.float_info.max
        self.representative = None
        self.members = {}  # członkowie gatunku
        self.membersFitnesses = None  # fitnesy wszystkich członków gatunków
        self.adjustedFitness = None
        # self.fitness_history = []
        # fitness gatunku liczony według funkcji podanej w config
        self.fitness = None

    def update(self, representative, members, generation, config):
        # w razie potrzeby te funkcje mozna rozbic na pare innych
        # zapisuje reprezentanta, osobników gatunku, ich fitnessy
        # oblicza fitness gatunku i zapisuje czy gatunek jest w stagnacji
        # zapisz reprezentanta
        self.representative = representative
        # zapisz członków gatunku
        self.members = members
        # zapisz fitnessy członków gatunku
        self.membersFitnesses = [ind.fitness for ind in members]
        # oblicz obecny fitness
        self.calcFitness(config)
        # sprawdz czy gatunek się poprawił względem ostatniego poprawienia
        # inaczej mówiąc czy poprawa nastąpiła w mniejszej liczbie generacji
        # niż maxStagnation okreslione w config
        self.checkIsStagnant(generation, config)

    def checkIsStagnant(self, generation, config):
        # Sprawdza czy osobnik jest w stagnacji dłużej niż ustawiono w config
        # inaczej mówiąc sprawdza czy w mniejszej liczbie niz maxStagnation
        # wystapiła poprawa fintessu gatunku

        # jezeli obecny fitness jest lepszy niz fitness ostatniego polepszenia
        if self.fitness > self.lastImprovmentFitness:
            # to zapisz kiedy nastąpiła poprawa
            self.lastImproved = generation
            # oraz zapisz fitness tej poprawy
            self.lastImprovmentFitness = self.fitness
        # jezeli ostatnia poprawa była dalej niz maxStagnation generacji temu
        if generation - self.lastImproved > config["maxStagnation"]:
            # gatunek jest w stagnacji
            self.isStagnant = True
        else:
            # jezeli nie, to nie jest w stagncji
            self.isStagnant = False

    def calcFitness(self, config):
        # liczby fitness całego gatunku według funkcji podanej w config
        # np. max, mean, min
        func = eval('lambda fits:' + config["stagnationFunction"] + '(fits)')
        self.fitness = func(self.membersFitnesses)
