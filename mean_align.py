import h5py
import my_dtw
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from my_scoring import ScoringScheme
from constants import data_path, barcode_list, prefix_length
from util import z_normalize, trim_blank
from rdp import rdp


f = h5py.File(data_path+'barcoded_reads.hdf5', 'r')
dtw = my_dtw.LocalDTW(scoring_scheme=ScoringScheme())
pool_size = 10
min_pathlen = 300


def plot_change(original, new):
    plt.plot(original, label='original mean')
    plt.plot(new, label='new mean')
    plt.legend(loc='upper right')
    plt.show()


def make_average(signal1, signal2, path):
    assert len(path) == 2
    assert len(path[0]) > 0
    actual_index = path[1][0]
    values = []
    mean_signal = []
    for i in range(len(path[1])):
        if actual_index != path[1][i]:
            actual_index = path[1][i]
            mean_signal.append(np.mean(values))
            values = []
        values.append(signal2[path[0][i]])
    mean_signal.append(np.mean(values))
    print(len(signal1), len(mean_signal))
    print('path range:', path[1][0], path[1][-1])

    return np.mean([signal1[path[1][0]:path[1][-1]+1], mean_signal], axis=0)


for barcode in reversed(barcode_list):
    drafted_keys = np.random.choice(list(f[barcode].keys()), pool_size, replace=False)
    mean_signal = None
    for key1, key2 in combinations(drafted_keys, 2):
        if mean_signal is None:
            signal1 = np.array(f[barcode][key1]).astype(float)
            signal1 = trim_blank(z_normalize(signal1))[:prefix_length]
        else:
            signal1 = mean_signal
        signal2 = np.array(f[barcode][key2]).astype(float)
        signal2 = trim_blank(z_normalize(signal2))[:prefix_length]
        score = dtw.align(signal1, signal2)
        path = dtw.get_path(dtw.A)
        print('path length:', len(path[0]))
        if len(path[0]) < min_pathlen:
            continue
        #aligned1, aligned2 = dtw.get_aligned(signal1, signal2, path)
        new_mean_signal = make_average(signal1, signal2, path)
        #new_mean_signal = rdp(np.array(zip(np.arange(0, len(new_mean_signal)), new_mean_signal)), epsilon=0.9)
        if mean_signal is not None:
            plot_change(mean_signal, new_mean_signal)
        mean_signal = new_mean_signal

