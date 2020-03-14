import os
import h5py
import ldtw
import numpy as np
from sklearn.cluster import SpectralClustering
from constants import data_path, workplace_path, prefix_length, bucket_size
from util import trim_blank, z_normalize, moving_average
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def preprocess(sig):
    normalized = z_normalize(trim_blank(sig.astype(float)))
    return normalized[:prefix_length]


def knn_normalize(D, k):
    for i in range(len(D)):
        neighbours = np.argsort(D[i,:])[::-1]
        for j in range(k):
            D[i, neighbours[j]] = D[neighbours[j], i] = 1000
        for j in range(k+1, len(D)):
            D[i, neighbours[j]] = D[neighbours[j], i] = 10
    return D



def get_labels(distances, k):
    #distances = np.exp(-distances/(2*distances.std()))
    #distances = distances.max() - distances
    distances = knn_normalize(distances, int(len(distances)*.1))
    clustering = SpectralClustering(n_clusters=k,
                                    assign_labels="discretize",
                                    affinity='precomputed',
                                    random_state=0,
                                    n_jobs=1).fit(distances)
    return clustering.labels_ 


def generate_sample(filename, sample_size):
    filtered = {}
    with open(os.path.join(data_path, 'all2_filtered.txt'), 'r') as all:
        lines = all.readlines()
        for line in lines:
            name, label = line.split()
            filtered[name] = int(label)
    chosen_keys = np.random.choice(list(filtered.keys()), sample_size, replace=False)
    fronts = []
    ends = []
    with h5py.File(os.path.join(workplace_path, 'validation_dataset_big.hdf5'), 'r') as f_validation:
        for key in chosen_keys:
            print(key)
            signal = np.array(f_validation[key])
            fronts.append(list(preprocess(signal)))
            ends.append(list(preprocess(signal[::-1])))
        D = ldtw.ComputeMatrix([fronts, ends], os.path.join(data_path, 'scoring_scheme.txt'), bucket_size, 25)

        labels_pred = get_labels(np.maximum(D[0], D[1]), 4)
        labels_true = [filtered[x] for x in chosen_keys]
        print(contingency_matrix(labels_true, labels_pred))
        return adjusted_rand_score(labels_pred, labels_true)

rands = []
for i in range(25):
    print('Epoch {}'.format(i))
    rands.append(generate_sample('', 2000))
print(rands)
