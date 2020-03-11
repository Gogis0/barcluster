import h5py
import my_dtw
import numpy as np
import matplotlib.pyplot as plt
from my_scoring import *
from util import *


dtw = my_dtw.LocalDTW(score, bucket_size, cut_index)
REPS = 5
len_min = 100
len_max = 2001
len_step = 100
delta = 0.5
data_path = 'C:\\Users\\adria\\PycharmProjects\\BarcCluster\\data\\'
bc_keys = ['barcode01', 'barcode02', 'barcode03', 'barcode04']


f = h5py.File(data_path+'barcoded_reads.hdf5', 'r')
x, y = [], []
for l in range(len_min, len_max, len_step):
    print('len:', l)
    for rep in range(REPS):
        bc_key1, bc_key2 = np.random.choice(bc_keys, 2)
        sig_key1 = np.random.choice(list(f[bc_key1].keys()))
        sig_key2 = np.random.choice(list(f[bc_key2].keys()))
        sig1 = np.array(f[bc_key1][sig_key1])*1.0
        sig2 = np.array(f[bc_key2][sig_key2])*1.0
        pos1 = np.random.choice(len(sig1)-l)
        pos2 = np.random.choice(len(sig2)-l)
        sig1 = z_normalize(sig1[pos1:pos1 + l])
        sig2 = z_normalize(sig2[pos2:pos2 + l])
        d = dtw.align(sig1, sig2, delta)
        x.append(l)
        y.append(d)

plt.title('Random sequence dtw score (delta = {})'.format(delta))
plt.scatter(x, y)
plt.show()
