import os
import h5py
import ldtw
import numpy as np
from sklearn.cluster import SpectralClustering
from constants import data_path, workplace_path, prefix_length, bucket_size
from util import trim_blank, z_normalize, moving_average, save_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import silhouette_score


np.random.seed(0)


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


def get_train_labels(D, k):
    distances = D[0] + D[1]
    #distances = np.exp(-distances/(2*distances.std()))
    #distances = distances.max() - distances
    distances = knn_normalize(distances, int(len(distances)*.1))
    clustering = SpectralClustering(n_clusters=k,
                                    assign_labels="discretize",
                                    affinity='precomputed',
                                    random_state=0,
                                    n_jobs=1).fit(distances)
    return clustering.labels_


def get_inverse_permutation(labels_true, labels_pred):
    N = np.max(labels_true + 1)
    inverse_permutation = [-1]*N
    for i in range(N):
        majority = np.argmax(np.bincount(labels_pred[labels_true == i]))
        inverse_permutation[majority] = i
    return inverse_permutation


def depermutate_labels(labels_true, labels_pred):
    inverse_permutation = get_inverse_permutation(labels_true, labels_pred)
    for i in range(len(labels_pred)):
        labels_pred[i] = inverse_permutation[labels_pred[i]]
    return labels_pred


def get_test_labels(scores, k):
    labels = []
    scores = np.maximum(np.array(scores[0]), np.array(scores[1]))
    for s in scores:
        print(np.mean(s))
        label = k if np.mean(s) > 100 else -1
        labels.append(label)
    return labels


def choose_representatives(D, labels, num_representatives, k=4):
    distances = np.maximum(D[0], D[1])
    distances = np.exp(-distances/(2*distances.std()))
    distances = distances.max() - distances
    clustering = SpectralClustering(n_clusters=k,
                                    assign_labels="discretize",
                                    affinity='precomputed',
                                    random_state=0,
                                    n_jobs=4).fit(distances)

    labels = clustering.labels_
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
    with h5py.File(os.path.join(workplace_path, filename), 'r') as f_validation:
        for key in chosen_keys:
            #print(key)
            signal = np.array(f_validation[key])
            fronts.append(list(preprocess(signal)))
            ends.append(list(preprocess(signal[::-1])))
    return [fronts, ends], [filtered[x] for x in chosen_keys]


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


def nonzero_summary(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    return adjusted_rand_score(labels_true[labels_pred != -1], labels_pred[labels_pred != -1])


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
    D = D[0] + D[1]
    D_small = D[np.array(representatives).flatten(), :][:, np.array(representatives).flatten()]
    main_mean = np.mean([D_small[i, i] for i in range(len(D_small))])
    predictions = []
    for i in range(len(scores)):
        D_ = np.zeros((D_small.shape[0] + 1, D_small.shape[1] + 1))
        D_[:-1, :-1] = D_small
        D_[-1, :-1] = scores[i]
        D_[:-1, -1] = scores[i]
        D_[-1, -1] = main_mean
        D_ = np.exp(-D_ / (2*np.std(D_)))
        D_ = D_.max() - D_
        labels_true = np.array([len(representatives[0]) * [i] for i in range(4)]).flatten()
        ans = get_label_spectral(D_, 4, labels_true)
        predictions.append(ans)
    return predictions


def clustering_test(filename, train_size, test_size, k, num_representatives, N_threads):
        train_data, train_labels_true = generate_sample(filename, train_size)
        train_data = np.array(train_data)
        D = ldtw.ComputeMatrix(train_data, os.path.join(data_path, 'scoring_scheme.txt'), bucket_size, N_threads)
        D = np.array(D)

        #train_labels_pred = get_train_labels(D, k)
        represent_idx, train_labels_pred, _ = choose_representatives(D, train_labels_true, num_representatives)
        represent_idx = np.array(represent_idx)

        representatives = [train_data[0][represent_idx.flatten()], train_data[1][represent_idx.flatten()]]
        test_data, test_labels_true = generate_sample(filename, test_size)
        score_matrix = ldtw.AlignToRepresentatives(representatives, test_data, os.path.join(data_path, 'scoring_scheme.txt'),
                                                       bucket_size, N_threads)
        score_matrix = np.array(score_matrix)
        score_matrix = score_matrix[0]+score_matrix[1]
        test_labels_pred = validate(D, represent_idx, score_matrix)
        #print(test_labels_pred)
        #print(test_labels_true)

        print('Train ARI:', nonzero_summary(train_labels_true, train_labels_pred))
        print('Test ARI:', nonzero_summary(test_labels_true, test_labels_pred))
        test_labels_pred = depermutate_labels(test_labels_true, test_labels_pred)
        print(correctness_summary(test_labels_pred, test_labels_true))
        #print('Cassified:', np.sum(test_labels_pred==-1)/len(test_labels_pred))


def matrix_test(filename, num_epochs, train_size, k, N_threads):
    rand_scores = []
    silhouette_scores = []
    accuracies = []
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        train_data, labels_true = generate_sample(filename, train_size)
        train_data = np.array(train_data)
        labels_true = np.array(labels_true)
        D = ldtw.ComputeMatrix(train_data, os.path.join(data_path, 'scoring_scheme.txt'), bucket_size, N_threads)
        D = np.array(D)
        D = D[0] + D[1]
        clustering = SpectralClustering(n_clusters=k,
                                        assign_labels='discretize',
                                        affinity='precomputed',
                                        random_state=0,
                                        n_init=10).fit(D)
        labels_pred = np.array(clustering.labels_)
        labels_pred = depermutate_labels(labels_true, labels_pred)

        rand_scores.append(adjusted_rand_score(labels_true, labels_pred))
        silhouette_scores.append(silhouette_score(D, labels_true))
        accuracies.append(np.sum(labels_true==labels_pred)/train_size*100)
    print('mean ARI:', np.mean(rand_scores))
    print('mean SC:', np.mean(silhouette_scores))
    print('mean ACC:', np.mean(accuracies))


matrix_test('validation_dataset_big.hdf5', 5, 10, 4, 1)

"""
for i in range(100):
    print('Epoch {}'.format(i))
    clustering_test('validation_dataset_big.hdf5', 2000, 1000, 4, 20, 25)
"""
