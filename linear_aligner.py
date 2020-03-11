import os
import h5py
import logging
import my_dtw
import numpy as np
import matplotlib.pyplot as plt
from constants import data_path, workplace_path, prefix_length
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn import mixture
from sklearn.metrics import silhouette_score
from my_scoring import ScoringScheme
from visualization import plot_embedding, plot_scores
from util import *

np.random.seed(1)


def load_training_data(filename):
    reads = []
    key_to_index = []
    with open(os.path.join(data_path, filename+'.txt'), 'r') as f:
        N, _ = list(map(int, f.readline().split()))
        for i in range(N):
            key = f.readline()
            key_to_index.append(key)
            reads.append(list(map(float, f.readline().split())))

    D = np.zeros((N, N))
    frontiers = [[] for i in range(N)]
    index_to_key = {}
    with open(os.path.join(data_path, filename+'_out_idx.csv'), 'r') as f:
        f.readline()
        for i in range(N):
            index_to_key[i] = f.readline()
            for j in range(N-1):
                frontiers[i].append(list(map(int, f.readline().split())))
            row = list(map(float, f.readline().split(',')))
            for j in range(N):
                D[i, j] = D[j, i] = row[j]

    return reads, D, key_to_index, index_to_key, reads, frontiers


def get_inverse_permutation(labels_true, labels_pred):
    N = np.max(labels_true + 1)
    inverse_permutation = [-1]*N
    for i in range(N):
        majority = np.argmax(np.bincount(labels_pred[labels_true == i]))
        inverse_permutation[majority] = i
    return inverse_permutation


def depermutate_lables(labels_true, labels_pred):
    inverse_permutation = get_inverse_permutation(labels_true, labels_pred)
    for i in range(len(labels_pred)):
        labels_pred[i] = inverse_permutation[labels_pred[i]]
    return labels_pred


def get_representants(distances, labels_true, k, num_representatives):
    """

    :param distances: matrix of pairwise DTW scores
    :param labels_true:
    :param k: number of clusters
    :param num_representatives: the number of representatives from each cluster
    :return: list of lists of representatives for each cluster
    """
    distances = np.exp(-distances/(2*distances.std()))
    distances = distances.max() - distances
    clustering = SpectralClustering(n_clusters=k,
                                    assign_labels="discretize",
                                    affinity='precomputed',
                                    random_state=0,
                                    n_jobs=4).fit(distances)

    print(clustering.labels_)
    labels = depermutate_lables(labels_true, clustering.labels_)
    print(labels)
    cluster_list = [(np.argwhere(labels == i)).flatten() for i in range(k)]
    cluster_sizes = [len(x) for x in cluster_list]

    """
    for each cluster pick the representatives with most discriminative power, that is, the ones with
    maximal inside_barcode_mean/outside_barcode_mean scores
    """
    representatives = []
    for cluster in range(k):
        discriminative_scores = []
        for cluster_member in cluster_list[cluster]:
            within_cluster_score = np.mean([distances[cluster_member, x] for x in cluster_list[cluster]
                                            if cluster_member != x])
            cross_cluster_score = np.mean([distances[cluster_member, x]
                                           for other_cluster in range(k) if cluster != other_cluster
                                           for x in cluster_list[other_cluster]])

            discriminative_scores.append(within_cluster_score/cross_cluster_score)

        discriminative_scores = sorted(enumerate(discriminative_scores), key=lambda x: x[1], reverse=True)
        #np.random.shuffle(discriminative_scores)
        # select 1/3 of the representatives greedily, the rest randomly
        selected_random = 0
        selected_greedy = num_representatives - selected_random
        cluster_representatives = []
        # greedy selection
        for i in range(cluster_sizes[cluster] if num_representatives == -1 else selected_greedy):
            id_in_cluster = discriminative_scores[i][0]
            cluster_representatives.append(cluster_list[cluster][id_in_cluster])
        # random selection
        for i in np.random.choice(selected_greedy + np.arange(len(discriminative_scores)-selected_greedy),
                                    selected_random, replace=False):
            id_in_cluster = discriminative_scores[i][0]
            cluster_representatives.append(cluster_list[cluster][id_in_cluster])

        representatives.append(cluster_representatives)

    return representatives, labels, clustering.affinity_matrix_


def get_true_label(labels_true, labels_pred, x):
    N = np.max(labels_true + 1)
    inverse_permutation = [-1]*N
    for i in range(N):
        majority = np.argmax(np.bincount(labels_pred[labels_true == i]))
        inverse_permutation[majority] = i

    return inverse_permutation[x]


def make_validation_input(filename, representatives):
    with h5py.File(os.path.join(data_path, 'validation_dataset_big.hdf5'), 'r') as f_validation:
        print(len(list(f_validation.keys())))
        with open(os.path.join(data_path, filename+'_cluster.in'), 'w') as of:
            of.write('{} {}\n'.format(4, len(representatives[0])))
            for i in range(4):
                for j in range(len(representatives[i])):
                    starts = [x[0] for x in frontiers[representatives[i][j]]]
                    ends = [x[1] for x in frontiers[representatives[i][j]]]
                    qstart = int(np.quantile(starts, 0.05))
                    qend = int(np.quantile(ends, 0.95))
                    print(qstart, qend)
                    of.write(str(qend-qstart) + '\n')
                    of.write(' '.join(str(x) for x in reads[representatives[i][j]][qstart:qend]) + '\n')

            names = []
            barcodes = []
            with open(os.path.join(workplace_path, 'all2_filtered.txt'), 'r') as all:
                lines = all.readlines()
                for line in lines:
                    name, barcode = line.split()
                    names.append(name)
                    barcodes.append(barcode)

            #of.write(str(len(list(f_validation.keys()))) + '\n')
            for i in range(len(names)):
                if names[i] not in f_validation: continue
                print(names[i])
                signal = z_normalize(trim_blank(np.array(f_validation[names[i]]).astype(float)))[:prefix_length]
                of.write('{} {}\n'.format(names[i], barcodes[i]))
                of.write(' '.join(str(x) for x in signal) + '\n')


def get_scores(filename):
    with open(os.path.join(data_path, filename)) as f:
        N, r = map(int, f.readline().split())
        names = []
        labels = []
        scores = []
        X = []
        for i in range(N):
            line = list(f.readline().split())
            # if len(line) == 1: name = line[0]
            if len(line) == 1: break
            name, label = line[0], int(line[1])
            names.append(name)
            labels.append(label)
            rating = list(map(float, f.readline().split(',')))
            scores.append(rating)
            X.append(np.argmax([
                np.mean(sorted(rating[i * r: (i + 1) * r])) for i in range(4)
            ]))

    X = np.array(X)
    labels = np.array(labels)
    scores = np.array(scores)
    return X, labels, scores


def get_label_gmm(D, k, labels_true):
    embedding = SpectralEmbedding(n_components=2, affinity='precomputed').fit_transform(D)
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, n_init=50)
    gmm.fit(embedding)
    probs = gmm.predict_proba(embedding)
    labels_pred = np.array([np.argmax(probs[i]) for i in range(len(probs)-1)])
    corrected_labels = np.array(get_inverse_permutation(labels_true, labels_pred))
    ans = corrected_labels[np.argmax(probs[-1])]
    #plt.plot(probs[-1])
    #plt.show()
    scores_sorted = sorted(probs[-1], reverse=True)
    if scores_sorted[0] - scores_sorted[1] < 0.6:
        ans = -2
    return ans, embedding


def get_label_spectral(D, k, labels_true):
    clustering = SpectralClustering(n_clusters=k,
                                    assign_labels='discretize',
                                    affinity='precomputed',
                                    random_state=0,
                                    n_init=10,
                                    n_jobs=4).fit(D)
    labels_pred = clustering.labels_[:-1]
    inverse_permutation = get_inverse_permutation(labels_true, labels_pred)
    ans = inverse_permutation[clustering.labels_[-1]]
    return ans


def validate(D, representatives, scores):
    D_small = D[np.array(representatives).flatten(), :][:, np.array(representatives).flatten()]
    main_mean = np.mean([D_small[i, i] for i in range(len(D_small))])
    predictions = []
    for i in range(len(scores)):
        print(i)
        D_ = np.zeros((D_small.shape[0] + 1, D_small.shape[1] + 1))
        D_[:-1, :-1] = D_small
        D_[-1, :-1] = scores[i]
        D_[:-1, -1] = scores[i]
        D_[-1, -1] = main_mean
        D_ = np.exp(-D_ / (2*np.std(D_)))
        D_ = D_.max() - D_
        #plt.imshow(D_)
        #plt.show()
        labels_true = np.array([len(representatives[0]) * [i] for i in range(4)]).flatten()
        ans = get_label_spectral(D_, 4, labels_true)
        predictions.append(ans)
        #plot_scores(scores[i], ans, labels[i])
        #plot_embedding(embedding, 4, ans+1, labels[i])

    return D_small, predictions


def purify_frontiers(labels, frontiers):
    N = len(labels)
    new_frontiers = [[] for i in range(N)]
    for i in range(np.max(labels)+1):
        cluster_indices = np.argwhere(labels == i).flatten()
        for j in cluster_indices:
            for x in frontiers[j]:
                if x[0] in cluster_indices:
                    new_frontiers[j].append(x[1:])
    return new_frontiers


def representatives_selector(D, num_representatives, cluster_sizes, n_iters=10):
    """
    Try many random selections of representatives and validate them on the remainder of the reads.
    :param D: distance matrix
    :param num_representatives: the number of demanded representatives
    :param cluster_sizes: a list of
    :param n_iters: the number of random selections performed
    :return: the best selection of representatives
    """
    N = len(cluster_sizes)
    for iteration in range(n_iters):
        candidates_idx = np.array([
            np.random.choice(cluster_sizes[i], num_representatives) for i in range(N)
        ]).flatten()
        scores = []
        for i in range(len(D)):
            if i not in candidates_idx:
                scores.append(D[i, candidates_idx])
        D_ = D[candidates_idx,:][:, candidates_idx]
        _, pred = validate(D_, candidates_idx, scores)


def correctness(pred, labels):
    pred = np.array(pred)
    labels = np.array(labels)
    return np.sum(pred == labels)/len(pred)*100


def correctness_summary(pred, labels):
    assert len(pred) == len(labels)
    pred = np.array(pred)
    complete_accuracy = correctness(pred, labels)
    pred_, labels_ = [], []
    for i in range(len(pred)):
        if pred[i] != -1:
            pred_.append(pred[i])
            labels_.append(labels[i])
    labeled_accuracy = correctness(pred_, labels_)
    labeled_percentage = (len(pred_)/len(pred))*100
    print('{:20}:{:6} %'.format('Accuracy (complete)', complete_accuracy))
    print('{:20}:{:6} %'.format('Accuracy (labeled)', labeled_accuracy))
    print('{:20}:{:6} %'.format('Percentage of labeled', labeled_percentage))
    return complete_accuracy, labeled_accuracy, labeled_percentage


file = 'matrix_2000'
cluster_sizes = [500]*4
labels_true = np.array([i for i in range(len(cluster_sizes)) for j in range(cluster_sizes[i])])
reads, D, key_to_index, index_to_key, reads, frontiers = load_training_data(file)
representatives, labels_pred, affinity = get_representants(D, labels_true, 4, 20)
labels_pred = depermutate_lables(labels_true, labels_pred)
frontiers = purify_frontiers(labels_pred, frontiers)
make_validation_input('matrix_2000', representatives)

#X, labels, scores = get_scores(file+'_cluster.out')
#for i in range(len(labels)):
#    if labels[i] == 2:
#        labels[i] = 3
#    elif labels[i] == 3:
#        labels[i] = 2

#D_, predictions = validate(D, representatives, scores)
#print(correctness(predictions, labels))
