import h5py
import numpy as np
import matplotlib.pyplot as plt
from constants import barcodes_dict, barcode_list, data_path


def plot_means(means):
    plt.clf()
    for i in range(4):
        plt.plot(means[i], label='barcode {}'.format(i+1))
    plt.legend(loc='upper left')
    plt.show()


def make_signal(means, length):
    signal = []
    N = len(means)
    segment_len = length//(N-1)
    for i in range(N-1):
        signal.append(means[i])
        for j in range(1, segment_len):
            signal.append(means[i] + (means[i+1] - means[i])*(j/segment_len))
    return signal


f = h5py.File(data_path+'kmer_model.hdf5', 'r')
N_classes = len(barcode_list)
K = 6
kmers = {}
squiggle_length = 400

for x in list(f['model']):
    kmers[x[0].decode('utf-8')] = x[1]

mean_signals = []
for key in barcode_list:
    signal = np.array([])
    barcode = barcodes_dict[key]
    for i in range(0, len(barcode)-K):
        signal = np.append(signal, kmers[barcode[i:i + K]])
    mean_signals.append(make_signal(signal, squiggle_length))

for i in range(N_classes):
    plt.plot(mean_signals[i], label='barcode {}'.format(i))
plt.legend(loc='upper left')
plt.show()

