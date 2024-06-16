import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy.io
import pickleData
import time
from population import population
import parallel
import show as sh
import prep_data
np.seterr(all='raise')

nr_col = 0  # 0, 1, 2, "all"
v = "rain"
load = False

dt = 1/(24*60)
samples_per_day = 1/dt
hm_days = 7*2
hm_samples = int(hm_days*samples_per_day)
d_names = ["rain", "dry", "storm"]

###########################################################################

""" load data """
inputs = {}
r_inputs = {}
targets = {}
r_targets = {}
measurements = {}
r_measurements = {}
bias = {}
coeff = {}
prep_data.load_data('data_dry.mat', 'dry', hm_samples,
                    inputs, targets, measurements, bias, coeff,
                    calc_bias_and_coeff=True,
                    r_inputs=r_inputs, r_targets=r_targets, r_measurements=r_measurements)
prep_data.load_data('data_rain.mat', 'rain', hm_samples,
                    inputs, targets, measurements, bias, coeff,
                    r_inputs=r_inputs, r_targets=r_targets, r_measurements=r_measurements)
prep_data.load_data('data_storm.mat', 'storm', hm_samples,
                    inputs, targets, measurements, bias, coeff,
                    r_inputs=r_inputs, r_targets=r_targets, r_measurements=r_measurements)
labels = [r"$X_\mathrm{DCO}(t) ~ \mathrm{[mg~COD~l^{-1}]}$",
          r"$X_\mathrm{BH}(t) ~ \mathrm{[mg~COD~l^{-1}]}$",
          r"$X_\mathrm{BA}(t) ~ \mathrm{[mg~COD~l^{-1}]}$",
          r"$S_\mathrm{O_2}(t) ~ \mathrm{[mg~O_2~l^{-1}]}$",
          r"$S_\mathrm{NO}(t) ~ \mathrm{[mg~N~l^{-1}]}$",
          r"$S_\mathrm{NH}(t) ~ \mathrm{[mg~N~l^{-1}]}$"]

if nr_col != "all":
    for d in d_names:
        targets[d] = np.concatenate((targets[d][:, nr_col].reshape(-1, 1),
                                     targets[d][:, 3:]), axis=1)
        r_targets[d] = np.concatenate((r_targets[d][:, nr_col].reshape(-1, 1),
                                       r_targets[d][:, 3:]), axis=1)
    bias['targets'] = np.concatenate((bias["targets"][nr_col].reshape(-1, 1),
                                      bias["targets"][3:].reshape(-1, 1)))
    coeff['targets'] = np.concatenate((coeff["targets"][nr_col].reshape(-1, 1),
                                      coeff["targets"][3:].reshape(-1, 1)))
    labels = [labels[nr_col]] + labels[3:]
    # output matrix
    C = [ [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
    # output matrix used in learning
    C1 = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
else:
    C = [[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]
    C1 = np.eye(6)
C = np.array(C)
C1 = np.array(C1)
xEst0 = targets[v][0].reshape(-1, 1)
# number of inputs to the observer
hmU = inputs['dry'].shape[1]
# number of outputs from the observer
hmX = targets['dry'].shape[1]

############################################################################
"""configuration of evolution"""
config = {
    "BP_ver": "BP",  # BP, abdollahi
    "hmGen": 500,  # number of generations
    "maxTime": None,  # maximal time of evolution
    "hmSamples": hm_samples,  # number of learning data samples
    "popSize": 50,  # size of population
    "useParralel": False,  # use parralel computation?
    "hmWorkers": 10,  # number of parralel workers (threads)
    "hmDays": hm_days,
    "sysRank": hmX,
    "hmU": hmU,

    "wInitMax": 1.0,  # maximum value of network weights in initialisation
    "wInitMin": -1.0,  # minimum value of network weights in initialisation
    "hmN": 50,  # number of neurons in the hidden layer
    "min_n": 0.0001,  # minimum value of learning rate in initialisation
    'max_n': 1.0,  # maximum value of learning rate in initialisation

    "C": C,
    "C1": C1,
    "maxA": 0.0,  # maximum values of elements of matrix A in initialisation
    "minA": -1.0,  # minimum value of elements of matrix A in initialisation
    "minPoleRe": -15,  # minimum observer pole in initialisation
    "maxPoleRe": -0.1,  # maximum observer pole in initialization
    "x0": xEst0,  # initial observer estimate
    "dt": dt,  # sample time

    "pMutation": 0.8,  # probability of mutation of an individual
    "singleMutation": True,  # sets whether to have one mutation at a time
    # in an individual
    # means it will mutate only one of: A, poles, n1, n2, p1, p2
    "mutationAProb": 0.4,  # 0.2,  # probability of mutation of matrix A
    "mutationPolesProb": 0.4,  # 0.2,  # probability of pole mutation
    "mutationObserverParamProb": 0.8,  # probability of n1 n2 p1 p2 (together) mutation
    "mutationWeights": 0.2,  # probability of weights mutation
    "maxMutationFloatRange": 0.5,  # Specifies in percentage
    # (e.g. 0.5 -> 50%), by how much up or down MAXIMUM the float value can change
    "pCross": 0.8,  # probability that an individual will become a parent

    "elitism": 10,  # number of individuals subjected to elitism
    "maxE": 1000000,  # maxmial value of MSE
}


def evalInd(ind, config):
    output, V, W = ind.estimate(inputs["rain"], measurements["rain"],
                                targets["rain"], config['x0'], hm_samples, learn=True)
    try:
        ind.net.V = V / hm_samples
        ind.net.W = W / hm_samples
        output, *_ = ind.estimate(inputs["rain"], measurements["rain"],
                                  None, config['x0'], hm_samples, learn=False)
        # TODO ! hmSamples*liczba kolumna?
        MSE = 1000*np.sum((output - targets["rain"])**2) / hm_samples
    except TypeError:
        MSE = config['maxE']

    if MSE > config['maxE']:
        MSE = config['maxE']

    fitness = -MSE
    return fitness, MSE, ind.net.V, ind.net.W


def evalIndsNotParallel(inds, config):
    # dla każdego ind
    for ind in inds:
        ind.fitness, ind.MSE, V, W = evalInd(ind, config)
        if V is not None and W is not None:
            ind.net.V = V
            ind.net.W = W

save = []
for i in range(10):
    print(f'Call {i}')
    if load:
        loadFilename = "....myFile"
        loadData = pickleData.load_object(loadFilename)
        loadData = loadData[0][0]
        # config = loadData[0]
        best = loadData[1]
        rep = loadData[2]
        pop = dict([(x, best) for x in range(50)])
        # pop = rep.genInds[-1]
    else:
        best = None
        pop = None
        rep = None

    timeStart = time.time()
    pop = population(config, pop, best, rep)
    if config['useParralel']:
        if __name__ == '__main__':
            pe = parallel.ParallelEvaluator(config['hmWorkers'], evalInd)
            best, rep = pop.run(pe.evaluate, setTime=config['maxTime'], n=config['hmGen'])
    else:
        best, rep = pop.run(evalIndsNotParallel, setTime=config['maxTime'], n=config['hmGen'])

    timeEnd = time.time()
    timeSpan = timeEnd-timeStart
    print(f'Running time in seconds:{timeSpan}')

    # plot answer of result ind
    red = colors.to_rgba('red')
    color = [red]*hmX

    sh.showIndOutputs(best, inputs[v], measurements[v],
                      targets[v], xEst0, hm_samples, colors=color,
                      labels=labels, copy_ind=True, display=True,
                      r_targets=r_targets[v], bias=bias,
                      coeff=coeff, rescale=True)
    sh.showPopMeanFitnessAndErrorInGen(rep)
    sh.showBestFitAndErrorInGen(rep)

    save.append([config, best, rep])

    filename = f"linIsiecNaRaz_col_{nr_col}_BP.myFile"
    pickleData.save_object(save, filename)
