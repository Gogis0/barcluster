import numpy as np
from sklearn.cluster import SpectralClustering


def get_labels(D, n_clusters):
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    assign_labels="discretize",
                                    affinity='precomputed',
                                    random_state=0,
                                    n_jobs=1).fit(D)
    return clustering.labels_


def compute_cut(D, labels):
    N = len(D)
    cut_value = 0
    for i in range(N):
        for j in range(N):
            if labels[i] != labels[j]:
                cut_value += D[i, j]
    return cut_value


def RMD(D, classes, l=20):
    n_classes = len(classes)
    # computation of special G 'rank' for each data point
    N = len(D)
    G = [
        np.mean(sorted(D[i, :])[l:(l*2)]) for i in range(N)
    ]
    # sort by 'ranks'
    rank = np.argsort(G)/N
    # try different k,lambda to construct graphs
    min_cut = 2**1000
    labels = None
    for k in range(10, N//2, 10):
        for lam in [0.1, 0.2, 0.4, 0.6, 0.8, 1]:
            M = np.zeros(D.shape) + 1
            for i in range(N):
                deg = k*(lam + 2*(1-lam)*rank[i])
                neighbours = np.argsort(D[i, :])[::-1]
                for j in range(deg):
                    M[neighbours[j], i] = M[i, neighbours[j]] = 1
                labels_pred = get_labels(M, n_classes)
                cut_value = compute_cut(D, labels_pred)
                if cut_value < min_cut:
                    labels = labels_pred
                    min_cut = cut_value
    return labels
