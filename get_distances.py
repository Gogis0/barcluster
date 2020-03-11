import os
import h5py
import numpy as np
from my_dtw import GlobalDTW
from itertools import combinations
from util import z_normalize, trim_blank, moving_average, moving_median
from constants import data_path, barcode_list, train_size

abs_dist = lambda x, y: abs(x - y)
sq_dist  = lambda x, y: (x - y)**2


def preprocess(sig, smoothness):
    normalized = z_normalize(sig.astype(float))
    return moving_median(normalized, smoothness)


barcodes_file = 'barcode_dataset_raw.hdf5'
whole_reads_file = 'barcoded_reads.hdf5'
out_aligned = 'aligned_pointwise_distances_med.txt'
out_random = 'random_pointwise_distances_med.txt'
dtw = GlobalDTW()
aligned_distances = 0
random_distances = 0
smoothness_degree = 5

align_values = []
f = h5py.File(os.path.join(data_path, barcodes_file), 'r')
for barcode in barcode_list:
    train_data = list(f[barcode].keys())[:train_size]
    for (barcode1, barcode2) in combinations(train_data, 2):
        sig1 = preprocess(f[barcode][barcode1][:], smoothness_degree)
        sig2 = preprocess(f[barcode][barcode2][:], smoothness_degree)
        dtw.align(sig1, sig2)
        path = dtw.get_path(dtw.A)
        for x in zip(path[0], path[1]):
            align_values.append(abs_dist(sig1[x[0]], sig2[x[1]]))
        aligned_distances += len(path[0])

of = open(os.path.join(data_path, out_aligned), 'w')
for i in range(aligned_distances):
    of.write(str(align_values[i]) + '\n')
of.close()

random_values = np.array([])
f = h5py.File(os.path.join(data_path, whole_reads_file), 'r')
while len(random_values) < aligned_distances:
        print(len(random_values)/aligned_distances)
        barcode1, barcode2 = np.random.choice(barcode_list, 2)
        key1 = np.random.choice(list(f[barcode1].keys())[:train_size])
        key2 = np.random.choice(list(f[barcode2].keys())[:train_size])
        sig1 = preprocess(trim_blank(f[barcode1][key1][:]), smoothness_degree)
        sig2 = preprocess(trim_blank(f[barcode2][key2][:]), smoothness_degree)
        N = min(len(sig1), len(sig2))//2
        draft1 = np.random.choice(sig1, N, replace=False)
        draft2 = np.random.choice(sig2, N, replace=False)
        values = np.abs(draft1-draft2)
        random_values = np.append(random_values, values)

of = open(os.path.join(data_path, out_random), 'w')
for i in range(aligned_distances):
    of.write(str(random_values[i])+'\n')
of.close()
