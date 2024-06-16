import numpy as np
import scipy.io
import pickleData
import matplotlib.pyplot as plt


def scale(signals, bias=None, coeff=None):
    if bias is not None or coeff is not None:
        have_biases_and_coeff = True
    else:
        have_biases_and_coeff = False
    # skaluje sygnaly do zakresu 0-1
    s = np.zeros(signals.shape)
    # ile wejsc do modelu
    hmSignals = signals.shape[1]
    # obróbka wejsc
    if not have_biases_and_coeff:
        coeff = np.zeros(hmSignals)
        bias = np.zeros(hmSignals)
    for i in range(hmSignals):
        col = signals[:, i]
        if not have_biases_and_coeff:
            bias[i] = min(col)
        col = col - bias[i]
        if not have_biases_and_coeff:
            coeff[i] = max(col) or 1
        s[:, i] = col / coeff[i]
    return s, bias, coeff


def rescale(signals, bias, coeff, t):
    r = np.zeros(signals.shape)
    hmSignals = signals.shape[1]
    for i in range(hmSignals):
        r[:, i] = (signals[:, i] * coeff[t][i]) + bias[t][i]
    return r
################################################################################


""" load data """
def load_data(filename, version, hm_samples,
              inputs_dict, targets_dict, measurements_dict,
              bias_dict, coeff_dict, calc_bias_and_coeff=False,
              r_inputs=None, r_targets=None, r_measurements=None):

    types = ["inputs", "targets", "measurements"]
    data = scipy.io.loadmat("""W:/neural_observer/data/"""+filename)

    for i, t in enumerate(types):
        d = data[t]
        d = d[:hm_samples, :]
        if eval("r_" + t) is not None:
            exec("r_" + t + "[version] = d")
        if calc_bias_and_coeff:
            d, d_bias, d_coeff = scale(d)
            bias_dict[t] = d_bias
            coeff_dict[t] = d_coeff
        else:
            d, *_ = scale(d, bias_dict[t], coeff_dict[t])
        exec(t+"_dict[version]=d")
        # plt.plot(d)
        # plt.show()