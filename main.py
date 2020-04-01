import os
import h5py
import ldtw
import numpy as np
import argparse
from sklearn.cluster import SpectralClustering
from constants import data_path, workplace_path, prefix_length, bucket_size
from util import trim_blank, z_normalize, moving_average, save_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import silhouette_score
from sklearn.manifold import SpectralEmbedding
from sklearn import mixture



parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default='base',
                    help='the dataset you want to run the pipeline on')
parser.add_argument('barcodes', nargs='+', type=int, default=None,
                    help='tha barcodes to use')
parser.add_argument('sample_size', type=int, default=2000,
                    help='the size of the sample in the initial phase')
parser.add_argument('test_size', type=int, default=10000,
                    help='the size of the sample in the testing phase')
parser.add_argument('representatives', type=int, default=25,
                    help='the number of representatives to select in the initial phase')
parser.add_argument('n_iters', type=int, default=1,
                    help='the number of iterations ...')
parser.add_argument('threads', type=int, default=1,
                    help='the number of threads you want to utilize')
args = parser.parse_args()

np.random.seed(3)


class MajorityPermutationFail(Exception):
    pass


def preprocess(sig):
    normalized = z_normalize(trim_blank(sig.astype(float)))
    return normalized[:prefix_length]


def convert_to_distance(S):
    """ Converts a similarity matrix to distance matrix """
    np.fill_diagonal(S, 0)
    S = -S
    S -= np.min(S)
    np.fill_diagonal(S, 0)
    return S


def gaussian_kernel(D):
    D = np.exp(-D/(2*D.std()))
    D = convert_to_distance(D)
    return D


def knn_normalize(D, k):
    for i in range(len(D)):
        neighbours = np.argsort(D[i,:])[::-1]
        for j in range(k):
            D[i, neighbours[j]] = D[neighbours[j], i] = 1000
        for j in range(k+1, len(D)):
            D[i, neighbours[j]] = D[neighbours[j], i] = 1
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


def get_inverse_permutation(labels_true, labels_pred, classes=None):
    if classes is None:
        classes = list(range(np.max(labels_true) + 1))
    N = len(classes)
    inverse_permutation = {}
    for i in range(N):
        counts = np.bincount(labels_pred[labels_true == classes[i]])
        print('class: ', classes[i], 'counts:', counts)
        majority = np.argmax(counts)
        inverse_permutation[majority] = classes[i]
    print(inverse_permutation)

    return inverse_permutation


def depermutate_labels(labels_true, labels_pred, classes):
    inverse_permutation = get_inverse_permutation(labels_true, labels_pred, classes)
    for i in range(len(labels_pred)):
        try:
            labels_pred[i] = inverse_permutation[labels_pred[i]]
        except:
            raise MajorityPermutationFail
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
    distances = D[0] + D[1]
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
            within_cluster_score = np.prod([distances[cluster_member, x] for x in cluster_list[cluster]
                                            if cluster_member != x])
            cross_cluster_score = np.prod([distances[cluster_member, x]
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

    return representatives, labels


def generate_sample(filename, sample_size, use_classes=None):
    filtered = {}
    with open(os.path.join(data_path, 'all_deepbinner.txt'), 'r') as all:
        lines = all.readlines()
        for line in lines:
            name, label = line.split()
            label = int(label)
            if use_classes is None or label in use_classes:
                filtered[name] = label
    chosen_keys = np.random.choice(list(filtered.keys()), sample_size, replace=False)
    fronts = []
    ends = []
    with h5py.File(os.path.join(workplace_path, filename), 'r') as f_validation:
        for key in chosen_keys:
            #print(filtered[key])
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


def get_label_spectral(D, k, labels_true, classes):
    D = gaussian_kernel(D)
    clustering = SpectralClustering(n_clusters=k,
                                    assign_labels='discretize',
                                    affinity='precomputed',
                                    random_state=0,
                                    n_init=20,
                                    n_jobs=4).fit(D)
    labels_pred = clustering.labels_[:-1]
    inverse_permutation = get_inverse_permutation(labels_true, labels_pred, classes)
    ans = inverse_permutation[clustering.labels_[-1]]
    return ans


def get_labels_gmm(D, k):
    embedding = SpectralEmbedding(n_components=4, affinity='precomputed').fit_transform(D)
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, n_init=50)
    gmm.fit(embedding)
    probs = gmm.predict_proba(embedding)
    labels_pred = np.array([np.argmax(probs[i]) for i in range(len(probs))])
    return labels_pred


def validate(D, representatives, scores):
    D_small = D[np.array(representatives).flatten(), :][:, np.array(representatives).flatten()]
    main_mean = np.mean([D_small[i, i] for i in range(len(D_small))])
    predictions = []
    for i in range(len(scores)):
        D_ = np.zeros((D_small.shape[0] + 1, D_small.shape[1] + 1))
        D_[:-1, :-1] = D_small
        D_[-1, :-1] = scores[i]
        D_[:-1, -1] = scores[i]
        D_[-1, -1] = main_mean
        labels_true = np.array([len(representatives[0]) * [i] for i in range(4)]).flatten()
        ans = get_label_spectral(D_, 4, labels_true)
        predictions.append(ans)
    return predictions


def clustering_test(filename, use_classes, train_size, test_size, k, num_representatives, N_threads, num_iters=1):
        test_data, test_labels_true = generate_sample(filename, test_size, use_classes=use_classes)
        print('test data sampled with class distribution:', np.bincount(test_labels_true))
        test_data = np.array(test_data)
        test_labels_true = np.array(test_labels_true)
        test_labels_pred_all = []
        matrices = []

        for iteration in range(num_iters):
            print('Iteration {}:'.format(iteration))
            if iteration == 0:
                train_data, train_labels_true = generate_sample(filename, train_size)
            else:
                class_size = train_size//k
                print(np.nonzero(test_labels_pred_all[-1] == 0)[0].shape)
                selected_idx = [np.random.choice(np.nonzero(test_labels_pred_all[-1] == barcode)[0],
                        class_size, replace=False) for barcode in range(k)]
                selected_idx = np.concatenate(selected_idx, axis=0)
                train_data = [test_data[0][selected_idx], test_data[1][selected_idx]]
                train_labels_true = np.array(test_labels_true[selected_idx])
            train_data = np.array(train_data)
            D = ldtw.ComputeMatrix(train_data, os.path.join(data_path, 'scoring_scheme.txt'), bucket_size, N_threads)
            D = np.array(D)
            print('computing train matrix finished')

            train_labels_true = np.array(train_labels_true)
            train_labels_pred = depermutate_labels(train_labels_true, np.array(get_train_labels(D, k)), use_classes)
            print('Train ARI:', nonzero_summary(train_labels_true, train_labels_pred))
            print('Train ACC:', np.sum(train_labels_true==train_labels_pred)/train_size*100)
            represent_idx, train_labels_pred = choose_representatives(D, train_labels_pred, num_representatives)
            represent_idx = np.array(represent_idx)

            representatives = [train_data[0][represent_idx.flatten()], train_data[1][represent_idx.flatten()]]
            score_matrix = ldtw.AlignToRepresentatives(representatives, test_data, os.path.join(data_path, 'scoring_scheme.txt'),
                                                           bucket_size, N_threads)
            print('aligning the sample finished')
            init_matrix = D[0]+D[1]
            score_matrix = np.array(score_matrix)
            score_matrix = score_matrix[0] + score_matrix[1]

            test_labels_pred = validate(init_matrix, represent_idx, score_matrix)
            test_labels_pred = np.array(test_labels_pred)
            #test_labels_pred = np.array([depermutate_labels(test_labels_true, test_labels_pred[i]) for i in range(2)])
            #test_labels_pred = np.array([test_labels_pred[0][i] if test_labels_pred[0][i]==test_labels_pred[1][i] else -1 for i in range(len(test_labels_pred[0]))])
            #print(test_labels_pred)
            #print(test_labels_true)
            test_labels_pred = depermutate_labels(test_labels_true, test_labels_pred, use_classes)

            print('Test ARI:', nonzero_summary(test_labels_true, test_labels_pred))
            print('Test ACC:', np.sum(test_labels_true==test_labels_pred)/test_size*100)
            #print('Cassified:', np.sum(test_labels_pred==-1)/len(test_labels_pred))
            test_labels_pred_all.append(test_labels_pred)
            matrices.append(D)

        print('iterations completed')
        test_labels_pred = np.array(test_labels_pred_all)
        maximums = np.array([np.argmax(np.bincount(test_labels_pred[:, i])) for i in range(test_size)])
        test_labels_pred = np.array([maximums[i] if np.sum(test_labels_pred[:, i] == maximums[i]) >= num_iters//2\
                            else -1 for i in range(test_size)])
        correctness_summary(test_labels_pred, test_labels_true)

        return matrices, test_data, test_labels_true, test_labels_pred_all


def matrix_test(filename, num_epochs, train_size, k, num_samples, N_threads):
    rand_scores = []
    silhouette_scores = []
    accuracies = []
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        train_data, labels_true = generate_sample(filename, train_size)
        train_data = np.array(train_data)
        labels_true = np.array(labels_true)
        D = ldtw.ComputeMatrix(train_data, os.path.join(data_path, 'scoring_scheme.txt'), bucket_size, N_threads)
        print('ldtw finished')
        D = np.array(D)
        labels_pred = []
        for i in range(num_samples):
            clustering = SpectralClustering(n_clusters=k,
                                            assign_labels='discretize',
                                            affinity='precomputed',
                                            random_state=0,
                                            n_init=10).fit(gaussian_kernel(D[i]))
            labels = np.array(clustering.labels_)
            labels = depermutate_labels(labels_true, labels)
            print('one side ACC:', np.sum(labels_true==labels)/train_size*100)
            print(list(labels)[:20])
            labels_pred.append(labels)
        
        labels_pred = np.array([labels_pred[0][i] if labels_pred[0][i]==labels_pred[1][i] else -1 for i in range(len(labels_pred[0]))])
        print(labels_pred[:20])

        #rand_scores.append(adjusted_rand_score(labels_true, labels_pred))
        #silhouette_scores.append(silhouette_score(convert_to_distance(D), labels_pred))
        accuracies.append(np.sum(labels_true==labels_pred)/train_size*100)
        correctness_summary(labels_pred, labels_true)
    #print('mean ARI:', np.mean(rand_scores))
    #print('mean SC:', np.mean(silhouette_scores))
    print('mean ACC:', np.mean(accuracies))


#matrix_test('validation_dataset_big.hdf5', 50, 2000, 4, 2, 15)

for i in range(10):
    print('Epoch {}'.format(i))
    M, data, labels_true, labels_pred_all = clustering_test(args.dataset+'.hdf5',
                                                            tuple(args.barcodes),
                                                            args.sample_size, args.test_size, 4,
                                                            args.representatives, args.threads,
                                                            num_iters=args.n_iters)
