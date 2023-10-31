import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy.io
import pickleData
import time
from population import population
import parallel
import show as sh
import sys
import copy

data = pickleData.load_object('hmGen250_hmN50_hmDays14_eval14_obsZsiecia_10x10_col0.myFile')

for obs_lin in data:
    for obs in obs_lin:
        rep = obs[2]
        MSE = [x.MSE for x in rep.genBest]
        plt.plot(MSE[100:])
    plt.show()
