import os
import numpy as np
import matplotlib.pyplot as plt
from double_boxplot import make_double_boxplot
from constants import data_path


def plot_delta(filenames, xmin=0, xmax=1.01, xstep=0.05):
    plt.figure(figsize=(8, 4))
    for filename in filenames:
        inter_list, cross_list = [], []
        delta_list = []
        with open(os.path.join(data_path, filename+'.out'), 'r') as f:
            num_deltas = int(next(f))
            for i in range(num_deltas):
                delta = float(f.readline())
                delta_list.append(delta)
                num_inter = int(f.readline())
                inter_list.append(list(map(float, [f.readline() for _ in range(num_inter)])))
                num_cross = int(f.readline())
                cross_list.append(list(map(float, [f.readline() for _ in range(num_cross)])))

            plot_means = []
            for i in range(num_deltas):
                plot_means.append(np.median(inter_list[i])/np.median(cross_list[i]))
                #plot_means[-1] /= np.max(inter_list[i]) + np.max(cross_list[i])
            x_axis = np.arange(xmin, xmax, xstep)
            plt.plot(x_axis, plot_means, label=filename)
            plt.xticks(np.arange(min(x_axis), max(x_axis) + 0.01, 0.05))


plot_delta(['delta_raw', 'delta_med5', 'delta_avg5'])
plt.legend(loc='best')
plt.show()
