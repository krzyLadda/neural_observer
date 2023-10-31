import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
import prep_data
import pathlib
import os

fontSize = 10
matplotlib.rcParams.update({'font.size': fontSize,
                            'lines.linewidth': 0.5})
legend_line_width = 1.0

def __flatten__(t):
    # zamienia liste list na spłaszczoną jedną liste
    return [item for sublist in t for item in sublist]


def __showOutputAndTarget__(output, target, fig=None, size=(8, 5), color=None,
                            label=None, dt=None):
    # plotuje jedną odpowiedż i target
    if fig is None:
        fig = plt.figure(figsize=size)
    x = np.arange(0, max(len(output), len(target)))
    if dt is not None:
        x = x*dt
    plt.plot(x, target, 'b--')
    if color is None:
        plt.plot(x, output)
    else:
        plt.plot(x, output, color=color)
    plt.grid()
    plt.xlabel(r'$t[days]$')
    if label is not None:
        plt.ylabel(label, rotation=90)
    plt.show()


def showOutputsAndTargets(outputs, targets, size=(8, 5), colors=None, labels=None,
                          dt=None):
    # plotuje odpowiedzi i targety
    n = targets.shape[1]
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, n))
    if labels is None:
        labels = [None]*n
    for i in range(n):
        color = colors[i]
        label = labels[i]
        fig = plt.figure(i, figsize=size)
        __showOutputAndTarget__(outputs[:, i], targets[:, i], fig, size, color,
                                label, dt)


def showIndOutputs(ind, inputs, measurements, targets, x0, hmSamples, size=(8, 5),
                   colors=None, labels=None, startSample=0, endSample=None,
                   copy_ind=True, display=True, r_targets=None, bias=None,
                   coeff=None, rescale=False):
    if endSample is None:
        endSample = inputs.shape[0]
    if copy_ind:
        indForShow = copy.deepcopy(ind)
    else:
        indForShow = ind

    outputs, V, W = indForShow.estimate(inputs, measurements, None, x0,
                                        hmSamples, learn=False)

    MSE = 1000*np.sum((outputs[startSample:endSample, :] - targets[startSample:endSample, :])
                      ** 2) / (hmSamples - startSample)
    if rescale:
        r_outputs = prep_data.rescale(outputs, bias, coeff, "targets")
        rMSE = 1000*np.sum((r_outputs[startSample:endSample, :] - r_targets[startSample:endSample, :])
                           ** 2) / (hmSamples - startSample)
        if display:
            showOutputsAndTargets(r_outputs[startSample:endSample, :], r_targets[startSample:endSample, :],
                                  size, colors, labels, dt=indForShow.dt)
    else:
        rMSE = 0
        if display:
            showOutputsAndTargets(outputs[startSample:endSample, :], targets[startSample:endSample, :],
                                  size, colors, labels, dt=indForShow.dt)

    # print("Individual with ID = {} has MSE = {}".format(indForShow.ID, round(MSE, 4)))
    return MSE, rMSE, outputs, r_outputs


def __showThroughGens__(course, startGen=0, ylabel=None, fig=None, size=(8, 5),
                        color=None):
    # plotuje przebieg nad kolejnymi generacjami
    if fig is None:
        fig = plt.figure(figsize=size)
    # podpisy osi x
    x = np.arange(startGen, startGen+len(course), 1)
    # color lini
    if color is None:
        plt.plot(x, course)
    else:
        plt.plot(x, course, color=color)
    # zadbaj żeby krok na osi x nie był mniejszy niz 1
    ax = fig.axes[0]
    xticks = ax.get_xticks()
    if min(np.diff(xticks)) < 1:
        ax.set_xticks(x)
    # plotuj
    plt.grid()
    plt.xlabel(r'Number of generation $[\#]$')
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def showPopMeanFitnessAndErrorInGen(rep):
    # pokaz sredniego przystosowania i MSE populacji w kolejnych generacjach
    genPopMeanFitness = []
    genPopMeanError = []
    # dla osobnikow każdej generacji
    # for gen in rep.genInds:
    #     # licz mean fit
    #     # TODO: tutaj można liczyć też odchylenia standardowe lub inne
    #     genPopFitness = [ind.fitness for ind in gen.values()]
    #     genPopMeanFitness.append(np.mean(genPopFitness))

    #     # licz mean error
    #     # dla każdego ind w każdym gatunku w tej generacji
    #     genPopError = [ind.MSE for ind in gen.values()]
    #     genPopMeanError.append(np.mean(genPopError))
    # dla błędów MSE z każdej generacji
    for genMSE in rep.genMSEs:
        genfit = [-x for x in genMSE]
        genPopMeanError.append(np.mean(genMSE))
        genPopMeanFitness.append(np.mean(genfit))

    # __showThroughGens__(genPopMeanFitness, ylabel=r'Average fitness of population [-]',
    #                     startGen=rep.startGen)
    __showThroughGens__(genPopMeanError,
                        ylabel=r'Average value of MSE of population [-]',
                        startGen=rep.startGen)


def showBestFitAndErrorInGen(rep):
    # pokaz przystosowania i MSE najlepszego w populacji w kolejnych generacjach
    bestFitness = [best.fitness for best in rep.genBest]
    bestError = [best.MSE for best in rep.genBest]
    # __showThroughGens__(bestFitness, ylabel=r'Fitness of best individual [-]')
    __showThroughGens__(bestError, ylabel=r'MSE of best individual [-]')


def showSpeciesSize(rep, size=(8, 5)):
    # pokaz jaki miał rozmiar każdy gatunek w kolejnych generacjach
    # inicjalizuj liste zerami
    maxGenID = rep.maxSpeciesID()
    hmGen = len(rep.genSpecies)
    foo = np.zeros((hmGen, maxGenID+1))
    # dla gatunków każdej generacji
    for genIdx, genSpecies in enumerate(rep.genSpecies):
        # dla każdego gatunku w tej generacji
        for sID, s in genSpecies.items():
            # zapisz ile ma osobników
            foo[genIdx, sID] = len(s.members)
    foo = foo.T
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(rep.startGen, rep.startGen+hmGen, 1)
    ax.stackplot(x, *foo)
    ax.set_xlabel(r'Number of generation $[\#]$')
    ax.set_ylabel(r'Species size $[\#]#')
    # zadbaj żeby krok na osi x nie był mniejszy niz 1
    xticks = ax.get_xticks()
    if min(np.diff(xticks)) < 1:
        ax.set_xticks(x)

    plt.show()


def plot_signal(y, fig=None, size=(8, 5), color=None, label=None, dt=None):
    # plotuje jedną odpowiedż i target
    if fig is None:
        fig = plt.figure(figsize=size)
    x = np.arange(0, len(y))
    if dt is not None:
        x = x*dt
    if color is None:
        plt.plot(x, y)
    else:
        plt.plot(x, y, color=color)
    plt.grid()
    plt.xlabel(r'$t[days]$')
    if label is not None:
        plt.ylabel(label, rotation=90)
    # plt.show()


def plot_signals(ys, fig=None, size=(8, 5), colors=None, label=None, dt=None, startIdx=0):
    # fig = plt.figure(figsize=size)
    for y, c in zip(ys, colors):
        plot_signal(y[startIdx:], fig, size, c, label, dt)


def answers_on_one(answers, labels, targets, dt, size=(8, 5), path=None, data_name=None):
    if path is not None:
        path = path + "/all_on_one/"
        if not os.path.isdir(path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    hm_s, n = answers[0].shape
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    x = np.arange(0, hm_s)
    x = x*dt
    for i in range(n):
        fig = plt.figure(figsize=size)
        ans = [x[:, i] for x in answers]
        plot_signals(ans, fig, colors=colors, label=labels[i], dt=dt)
        plt.plot(x, targets[:, i], 'b--', label="Target trajectory")
        plt.xlim(0, max(x)+0.001)
        # if i == 0 and nr_col == 0 and name == "dry":
        #     plt.ylim(0.99, 20.001)
        plt.xlabel(r'$t[days]$')
        plt.ylabel(labels[i], rotation=90)
        leg = plt.legend()
        for legobj in leg.legendHandles:
            legobj.set_linewidth(legend_line_width)
        plt.grid()
        if path is not None:
            filename = f"{data_name}_{i+1}.png"
            plt.savefig(path+filename, bbox_inches='tight')
        plt.show()


def best_worst_on_one(ans_best, ans_worst, labels, targets, dt, size=(8, 5),
                      path=None, data_name=None):
    if path is not None:
        path = path + "/best_worst/"
        if not os.path.isdir(path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    hm_s, n = targets.shape
    x = np.arange(0, hm_s)
    x = x*dt
    for i in range(n):
        fig = plt.figure(figsize=size)
        plt.plot(x, ans_best[:, i], 'g', label="Response trajectory of the best")
        plt.plot(x, ans_worst[:, i], 'r', label="Resposne trajectory of the worst")
        plt.plot(x, targets[:, i], 'b--', label="Target trajectory")  # , alpha=0.2, linewidth=1.0)
        plt.xlim(0, max(x)+0.001)
        plt.xlabel(r'$t[days]$')
        plt.ylabel(labels[i], rotation=90)
        leg = plt.legend()
        for legobj in leg.legendHandles:
            legobj.set_linewidth(legend_line_width)
        plt.grid()
        if path is not None:
            filename = f"{data_name}_{i+1}.png"
            plt.savefig(path+filename, bbox_inches='tight')
        plt.show()
