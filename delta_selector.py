import h5py
import os
import my_dtw
import numpy as np
from itertools import combinations, product
from util import moving_average, z_normalize
from my_scoring import ScoringScheme
from constants import data_path, barcode_list,\
    train_size, bucket_size, max_distance, prefix_length
import ldtw


np.random.seed(1997)


def preprocess(sig):
    normalized = z_normalize(sig.astype(float))
    return moving_average(normalized, 5)[:prefix_length]


def delta_iterative_search(filename, delta_min, delta_max, delta_step, bucket_draft_size=5):
    """
    Searches for the optimal delta (a priori prob.) for the score correction.
    Delta arguments are integers due to floating point inaccuracy.
    :param filename: HDF5 file of training reads divided into barcode groups
    :param delta_min: minimum delta value
    :param delta_max: maximum delta value
    :param delta_step: step size at which delta is increased
    :param bucket_draft_size: The number of reads from each barcode chosen for evaluation.
    :return: both inter-barcode and cross-barcode alignment scores for each delta
    """
    delta_list = np.arange(delta_min, delta_max, delta_step)
    scoring_scheme = ScoringScheme()
    test_data = {}
    with h5py.File(os.path.join(data_path, filename), 'r') as f:
        for barcode in barcode_list:
            test_data[barcode] = np.random.choice(list(f[barcode].keys())[train_size:],
                                                  bucket_draft_size, replace=False)
        inter_barcode_scores = {}
        cross_barcode_scores = {}
        for delta in delta_list:
            delta /= 100
            print('delta:', delta)
            for barcode in barcode_list:
                for (a, b) in combinations(test_data[barcode], 2):
                    sig1 = preprocess(np.array(f[barcode][a]).astype(float))
                    sig2 = preprocess(np.array(f[barcode][b]).astype(float))
                    d = ldtw.LikelihoodAlignment(sig1, sig2, scoring_scheme.score,
                                                 scoring_scheme.bucket_size, delta)[0]
                    inter_barcode_scores.setdefault(delta, []).append(d)

            for (barcode1, barcode2) in combinations(barcode_list, 2):
                for (a, b) in product(test_data[barcode1], test_data[barcode2]):
                    sig1 = preprocess(np.array(f[barcode][a]).astype(float))
                    sig2 = preprocess(np.array(f[barcode][b]).astype(float))
                    d = ldtw.LikelihoodAlignment(sig1, sig2, scoring_scheme.score,
                                                 scoring_scheme.bucket_size, delta)[0]
                    cross_barcode_scores.setdefault(delta, []).append(d)

    return inter_barcode_scores, cross_barcode_scores


inter, cross = delta_iterative_search('barcoded_reads.hdf5', 0, 1001, 50, 50)
with open(os.path.join(data_path, 'delta_avg_5.out'), 'r') as f:
    num_deltas = len(inter.keys())
    f.write('{}\n'.format(str(num_deltas)))
    for delta in inter.keys():
        f.write('{}\n'.format(str(delta)))

        f.write('{}\n'.format(str(len(inter[delta]))))
        for d in inter[delta]:
            f.write('{}\n'.format(str(d)))
        f.write('{}\n'.format(str(len(cross[delta]))))
        for d in cross[delta]:
            f.write('{}\n'.format(str(d)))

#dbp.make_double_boxplot(delta_list/100, (same, diff), 'images/score_boxplot.png')
#plt.scatter(*zip(*same), label='same barcode')
#plt.scatter(*zip(*diff), label='different barcode')
#plt.scatter(delta_list, same_avg, marker='_', label='same barcode score mean')
#plt.scatter(delta_list, diff_avg, marker='_', label='different barcode score mean')
#plt.legend(loc='upper right')
#plt.savefig('./images/delta.png')
