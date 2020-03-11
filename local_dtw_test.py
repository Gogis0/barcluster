import matplotlib.pyplot as plt
import numpy as np
import h5py
import my_dtw
from itertools import product
from constants import data_path
from my_scoring import ScoringScheme
from visualization import make_alignment_figure
from util import *


dtw = my_dtw.LocalDTW(scoring_scheme=ScoringScheme())
dtw_global = my_dtw.GlobalDTW()
bc1 = '1'
bc2 = '1'
f = h5py.File(data_path+'barcoded_reads.hdf5', 'r')
train_sz = 100
prefix_size = 1000
REPS = 0
mean = None

barcodes1 = list(f['barcode0'+bc1].keys())[train_sz:]
barcodes2 = list(f['barcode0'+bc2].keys())[train_sz:]

for (a, b) in product(barcodes1, barcodes2):
    if a == b: continue
    bar1, bar2 = np.array(f['barcode0'+bc1][a]), np.array(f['barcode0'+bc2][b])
    start1 = f['barcode0'+bc1][a].attrs['start'] - len(bar1)
    end1 = f['barcode0'+bc1][a].attrs['end'] - len(bar1)
    start2 = f['barcode0'+bc2][b].attrs['start'] - len(bar2)
    end2 = f['barcode0'+bc2][b].attrs['end'] - len(bar2)
    #bar1 = np.array(mean_barcodes.mean_signals[0])
    #bar2 = np.array(mean_barcodes.mean_signals[1])
    #bar1 = z_normalize(trim_blank(bar1, 300).astype(float))[400:400+prefix_size]
    #bar2 = z_normalize(trim_blank(bar2).astype(float))[400:400+prefix_size]
    bar1 = moving_average(z_normalize(trim_blank(bar1).astype(float)), 5)
    bar2 = moving_average(z_normalize(trim_blank(bar2).astype(float)), 5)
    #bar1 = bar1[cut:cut+prefix_size]
    #bar2 = bar2[cut:cut+prefix_size]
    bar1 = bar1[:prefix_size]
    bar2 = bar2[:prefix_size]
    d = dtw.align(bar1, bar2)
    print(a, b, 'dist:', d, 'scores:',f['barcode0'+bc1][a].attrs['score'], f['barcode0'+bc2][b].attrs['score'])
    path = dtw.get_path(dtw.A)
    sig1 = bar1[path[1][0]:path[1][-1]]
    sig2 = bar2[path[0][0]:path[0][-1]]
    sig1, sig2 = dtw.get_aligned(bar1, bar2, path)
    fig = make_alignment_figure(sig1, sig2, path, dtw.A)
    fig.show()

