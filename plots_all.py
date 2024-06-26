from __future__ import print_function
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy.io
import pickleData
import show as sh
import prep_data
import pandas as pd

for task_id in [1, 2, 3, 4]:
    nr_col = task_id-1
    for version in ["BP_with_retrain", "BP_without_retrain",
                    "Abdollahi_with_retrain", "Abdollahi_without_retrain"]:

        print(f"Version: {version}, task: {task_id}")
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
        labels = [r"$X_\mathrm{DCO}(t) ~ \mathrm{[g~COD~m^{-3}]}$",
                  r"$X_\mathrm{BH}(t) ~ \mathrm{[g~COD~m^{-3}]}$",
                  r"$X_\mathrm{BA}(t) ~ \mathrm{[g~COD~m^{-3}]}$",
                  r"$S_\mathrm{O_2}(t) ~ \mathrm{[g~O_2~m^{-3}]}$",
                  r"$S_\mathrm{NO}(t) ~ \mathrm{[g~N~m^{-3}]}$",
                  r"$S_\mathrm{NH}(t) ~ \mathrm{[g~N~m^{-3}]}$"]

        if nr_col != 3:
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
            # initial estimate
            xEst0 = [[0],
                     [0],
                     [0],
                     [0]]
        else:
            # initial estimate
            xEst0 = [[0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0]]
        xEst0 = np.array(xEst0)
        # number of inputs to the observer
        hmU = inputs['dry'].shape[1]
        # number of outputs from the observer
        hmX = targets['dry'].shape[1]
        ###########################################################################

        """ load observers """
        filename = f"results/{version}/task_{task_id}.myFile"
        data = pickleData.load_object(filename)

        """ get answers """
        red = colors.to_rgba('red')
        color = [red]*hmX

        answers = {}
        r_answers = {}
        mse = {}
        r_mse = {}
        foo = []
        for record in data:
            i = record[0]
            obs = record[1]
            rep = record[2]
            foo.append((obs.p1, obs.p2))
            # sh.showPopMeanFitnessAndErrorInGen(rep)
            # sh.showBestFitAndErrorInGen(rep)
            for d in d_names:
                e, r_e, out, r_out = sh.showIndOutputs(obs, inputs[d], measurements[d],
                                                       targets[d], xEst0, hm_samples, colors=color,
                                                       labels=labels, copy_ind=True, display=False,
                                                       r_targets=r_targets[d], bias=bias,
                                                       coeff=coeff, rescale=True)
                if d not in answers:
                    answers[d] = []
                    r_answers[d] = []
                    mse[d] = []
                    r_mse[d] = []
                answers[d].append(out)
                r_answers[d].append(r_out)
                mse[d].append(e)
                r_mse[d].append(r_e)

        # %%
        """plot all answers on one plot"""
        path = f"figs/task_x{task_id}/{version}"
        for d in d_names:
            sh.answers_on_one(r_answers[d], labels, r_targets[d], dt, path=path, data_name=d)

        # %%
        """plot answer of the best and the worst observer"""
        test = "dry"
        idx_best = np.where(mse[test] == np.min(mse[test]))[0][0]
        idx_worst = np.where(mse[test] == np.max(mse[test]))[0][0]
        for d in d_names:
            sh.best_worst_on_one(r_answers[d][idx_best], r_answers[d][idx_worst],
                                 labels, r_targets[d], dt, path=path, data_name=d)

        foo = [mse['rain'], mse['dry'], mse['storm']]
        foo = np.array(foo)
        foo = foo.T
        # open the file in the write mode
        df = pd.DataFrame(foo)
        df.to_csv(path+'/mses.csv', index=False, sep=',')
