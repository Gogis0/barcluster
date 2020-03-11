import h5py
import my_dtw
import numpy as np
from itertools import combinations
from my_scoring import ScoringScheme
from constants import data_path, barcode_list, prefix_length
from util import z_normalize, trim_blank


f = h5py.File(data_path+'barcoded_reads.hdf5', 'r')
dtw = my_dtw.LocalDTW(scoring_scheme=ScoringScheme())
pool_size = 10
min_pathlen = 300

for barcode in barcode_list:
    drafted_keys = np.random.choice(list(f[barcode].keys()), pool_size, replace=False)
    mean_signal = None
    for key1, key2 in combinations(drafted_keys, 2):
        signal1 = np.array(f[barcode][key1]).astype(float)
        signal2 = np.array(f[barcode][key2]).astype(float)
        signal1 = trim_blank(z_normalize(signal1))
        signal2 = trim_blank(z_normalize(signal2))
        score = dtw.align(signal1, signal2)
        if len(dtw.path) < min_pathlen:
            continue
        aligned1, aligned2 = dtw.get_aligned(signal1, signal2, path=dtw.path)
        if mean_signal is None:
            mean_signal = np.mean([aligned1, aligned2])

