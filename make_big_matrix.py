import numpy as np
import h5py
from my_scoring import *
from my_dtw import LocalDTW
from itertools import combinations
from util import *
from constants import barcode_list, train_size

np.random.seed(18)

f = h5py.File(data_path+'barcoded_reads.hdf5', 'r')
ids = []
for bc in barcode_list:
    keys = list(f[bc].keys())[train_size:]
    keys = [k for k in keys if f[bc][k].attrs['score'] > 60]
    for read_id in np.random.choice(list(f[bc].keys()), 2, replace=False):
        ids.append((bc, read_id))
print(ids)

dtw = LocalDTW(score, bucket_size, cut_index)
prefix_size = 1000

for (a, b) in combinations(ids, 2):
    bc1, id1 = a[0], a[1]
    sig1 = np.array(f[bc1][id1])
    bc2, id2 = b[0], b[1]
    sig2 = np.array(f[bc2][id2])

    sig1 = z_normalize(trim_blank(sig1, 300).astype(float))[:prefix_size]
    sig2 = z_normalize(trim_blank(sig2, 300).astype(float))[:prefix_size]
    d = dtw.align(sig1, sig2, 0.5)
    print('{},{},{}'.format(id1, id2, d))
