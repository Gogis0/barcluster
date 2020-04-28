import os
import h5py
import ldtw
import numpy as np
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from constants import data_path, workplace_path, prefix_length, bucket_size
from util import trim_blank, z_normalize, moving_average, moving_median, save_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics import f1_score
from sklearn.manifold import SpectralEmbedding
from sklearn import mixture
from itertools import permutations
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default='base',
                    help='the dataset you want to run the pipeline on')
parser.add_argument('--barcodes', nargs='+', type=int, default=None,
                    help='the barcodes to use')
parser.add_argument('--delta', type=float, default=0,
                    help='the delta value to subtract from the score')
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

#np.random.seed(89)

# filtered read names file
filtered_file = {'base': 'all2_filtered_3.txt', 'deepbinner': 'all_deepbinner.txt'}

# a dict to store (read_id, label) pairs fof the whole dataset
filtered = None

# a flag sgnalizing unsignificant barcode match
NO_BARCODE = -1

# placeholder for alignment scores
scores = None
labels = None


class MajorityPermutationFail(Exception):
    pass


def preprocess(sig, is_front=True):
    if is_front:
        normalized = z_normalize(trim_blank(sig.astype(float)))
    else:
        normalized = z_normalize(trim_blank(sig.astype(float)[::-1]))
    return moving_median((normalized[:prefix_length])[::-1], 5)


def norm_classes(labels):
    """
    Maps class labels to form a continous interval 0,...,N
    Example: (1, 2, 2, 7) -> (0, 1, 1, 2)
    """
    inv = {-1: -1}
    for idx, val in enumerate(sorted(np.unique(labels[labels != NO_BARCODE]))):
        inv[val] = idx
    return np.array(list(map(inv.get, labels)))


def drop_uncertain(D, labels_pred, labels_true, drop_ratio=0.05):
    D_ = D[0] + D[1]
    N = D_.shape[0]
    varlist = [np.var(D_[i, :]) for i in range(N)]
    varlist = np.argsort(varlist)[::-1]
    taken = int(len(varlist)*(1-drop_ratio))
    varlist = varlist[:taken]
    return D[:, varlist, varlist], labels_pred[varlist], labels_true[varlist]


def convert_to_distance(S):
    """ Converts a similarity matrix to distance matrix """
    np.fill_diagonal(S, 0)
    S = -S
    S -= np.min(S)
    np.fill_diagonal(S, 0)
    return S


def gaussian_kernel(D, sigma=1):
    #sigma = 2*D.std()
    D = np.exp(-D/sigma)
    D = convert_to_distance(D)
    return D


def knn_normalize(D_, k):
    D = D_.copy()
    for i in range(len(D)):
        neighbours = np.argsort(D_[i,:])[::-1]
        for j in range(k):
            D[i, neighbours[j]] = D[neighbours[j], i] = 1000
        for j in range(k+1, len(D)):
            D[i, neighbours[j]] = D[neighbours[j], i] = 1
    return D


def get_train_labels(D, n_clusters, k=0.1):
    distances = D[0] + D[1]
    distances = knn_normalize(distances, int(len(distances)*k))
    clustering = SpectralClustering(n_clusters=n_clusters,
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
        #print('class: ', classes[i], 'counts:', counts)
        majority = np.argmax(counts)
        inverse_permutation[majority] = classes[i]
    #print(inverse_permutation)

    return inverse_permutation


def max_likelihood_permutation(labels_true, labels_pred, classes):
    assert labels_true.shape == labels_pred.shape
    all_permutations = set(permutations(list(classes)))
    best_perm = None
    best_score = -1
    for permutation in all_permutations:
        new_labels_pred = labels_pred.copy()
        for i in range(len(new_labels_pred)):
            new_labels_pred[i] = permutation[new_labels_pred[i]]
        score = correctness(new_labels_pred, labels_true)
        #print(permutation, score)
        if score > best_score:
            best_score = score
            best_perm = permutation
    return best_perm


def depermutate_labels(labels_true, labels_pred, classes):
    labels_pred = norm_classes(labels_pred)
    inverse_permutation = max_likelihood_permutation(labels_true[labels_pred != NO_BARCODE], labels_pred[labels_pred != NO_BARCODE], classes)
    #print('inverse permutation:', inverse_permutation)
    for i in range(len(labels_pred)):
        try:
            if labels_pred[i] != NO_BARCODE:
                labels_pred[i] = inverse_permutation[labels_pred[i]]
        except:
            raise MajorityPermutationFail
    return labels_pred


def get_test_labels(scores, k):
    labels = []
    scores = np.maximum(np.array(scores[0]), np.array(scores[1]))
    for s in scores:
        print(np.mean(s))
        label = k if np.mean(s) > 100 else NO_BARCODE
        labels.append(label)
    return labels


def choose_representatives(D, labels, num_representatives, classes=None):
    k = len(classes)
    distances = np.log(D[0] + D[1]) # take log for the numerical stability
    #print(distances.shape)
    cluster_list = [(np.argwhere(labels == c)).flatten() for c in classes]
    #print(cluster_list)
    cluster_sizes = [len(x) for x in cluster_list]

    """
    for each cluster pick the representatives with most discriminative power, that is, the ones with
    maximal inside_barcode_mean/outside_barcode_mean scores
    """
    representatives = []
    for cluster in range(len(classes)):
        discriminative_scores = []
        for cluster_member in cluster_list[cluster]:
            within_cluster_score = np.mean([distances[cluster_member, x] for x in cluster_list[cluster]
                                            if cluster_member != x])
            cross_cluster_score = np.mean([distances[cluster_member, x]
                                           for other_cluster in range(k) if cluster != other_cluster
                                           for x in cluster_list[other_cluster]])
            #cross_cluster_score = np.sum(np.max([distances[cluster_member, x]
            #                               for x in cluster_list[other_cluster]])
            #                               for other_cluster in range(k) if cluster != other_cluster)

            discriminative_scores.append(within_cluster_score-cross_cluster_score)

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


def generator(filename, key):
    windows = []
    with h5py.File(os.path.join(workplace_path, filename), 'r') as f:
        signal = np.array(f[key])
        windows.append(list(preprocess(signal)))
        windows.append(list(preprocess(signal, is_front=False)))
    return windows


def generate_sample(filename, sample_size, use_classes=None):
    global filtered
    if filtered is None:
        filtered = {}
        with open(os.path.join(data_path, filtered_file[args.dataset]), 'r') as all:
            lines = all.readlines()
            for line in lines:
                name, label = line.split()
                label = int(label)
                if use_classes is None or label in use_classes:
                    filtered[name] = label
    chosen_keys = np.random.choice(list(filtered.keys()), sample_size, replace=False)
    fronts = []
    ends = []
    #with h5py.File(os.path.join(workplace_path, filename), 'r') as f_validation:
    """
        for key in tqdm(chosen_keys):
            #print(filtered[key])
            signal = np.array(f_validation[key])
            fronts.append(list(preprocess(signal)))
            ends.append(list(preprocess(signal, is_front=False)))
    """
    pool = multiprocessing.Pool()
    worker_partial = partial(generator, filename)
    fronts, ends = [], []
    for prefix, suffix in tqdm(pool.imap(worker_partial, [key for key in chosen_keys], chunksize=1), total=sample_size):
        fronts.append(prefix)
        ends.append(suffix)
    pool.close()
    pool.join()
    #fronts = [pooled_windows[i][0] for i in range(sample_size)]
    #ends = [pooled_windows[i][1] for i in range(sample_size)]
    return [fronts, ends], [filtered[x] for x in chosen_keys]


def class_accuracy(labels_pred, labels_true, k):
    labeled_idx = labels_pred != NO_BARCODE
    labels_pred = labels_pred[labeled_idx]
    labels_true = labels_true[labeled_idx]
    print(np.bincount(labels_pred))
    contingency = contingency_matrix(labels_true, labels_pred)
    print(contingency)
    precisions, recalls, = [], []
    for i in range(k):
        precisions.append(contingency[i, i]/np.sum(contingency[i, :])*100)
        recalls.append(contingency[i, i]/np.sum(contingency[:, i])*100)
    print('Precision per class ::', end='')
    for i in range(len(precisions)):
        print('class {}: {}%'.format(i+1, precisions[i]), end='  ')
    print()
    print('Recall per classes  ::', end='')
    for i in range(len(recalls)):
        print('class {}: {}%'.format(i+1, recalls[i]), end='  ')
    print()


def correctness(pred, labels):
    pred = np.array(pred)
    labels = np.array(labels)
    return np.sum(pred == labels)/len(pred)*100


def print_summary(summary, phase):
    complete_accuracy, labeled_accuracy, labeled_percentage, rand_index = summary
    print('{} =================================================='.format(phase))
    print('ARI:', rand_index)
    print('{:20}:{:6} %'.format('Accuracy (labeled)', labeled_accuracy))
    print('{:20}:{:6} %'.format('Accuracy (complete)', complete_accuracy))
    print('{:20}:{:6} %'.format('Percentage of labeled', labeled_percentage))
    print('==================================================')


def correctness_summary(pred, labels, k):
    assert len(pred) == len(labels)
    pred = np.array(pred)
    complete_accuracy = correctness(pred, labels)
    pred_, labels_ = [], []
    for i in range(len(pred)):
        if pred[i] != NO_BARCODE:
            pred_.append(pred[i])
            labels_.append(labels[i])
    labeled_accuracy = correctness(pred_, labels_)
    labeled_percentage = (len(pred_)/len(pred))*100
    rand_index = adjusted_rand_score(labels_, pred_)
    return complete_accuracy, labeled_accuracy, labeled_percentage, rand_index


def nonzero_summary(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    return adjusted_rand_score(labels_true[labels_pred != NO_BARCODE], labels_pred[labels_pred != NO_BARCODE])


def get_label_spectral(D, labels_true, classes):
    k = len(classes)
    clustering = SpectralClustering(n_clusters=k,
                                    assign_labels='discretize',
                                    affinity='precomputed',
                                    n_init=20,
                                    random_state=0,
                                    n_jobs=1).fit(D)
    labels_pred = clustering.labels_[:-1]
    inverse_permutation = max_likelihood_permutation(labels_true, labels_pred, classes)
    ans = inverse_permutation[clustering.labels_[-1]]
    return ans


def get_label_knn(scores, k, classes, num_representatives):
    # sort the scores ascendingly, preserving the classes
    scores_sorted = np.array(sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
    scores_sorted[:, 0] //= num_representatives
    scores_sorted[:, 0] = scores_sorted[:, 0]
    first_class_idx = scores_sorted[0, 0]
    knn_window = scores_sorted[:k, 0]
    class_counts = np.bincount(knn_window.astype(int))
    #print(class_counts)
    #if np.max(class_counts) > 2:
    #    return classes[np.argmax(class_counts)]
    if np.all(scores_sorted[:k, 0] == first_class_idx):
        return classes[int(first_class_idx)]
    else:
        return NO_BARCODE


def get_label_maxmean(scores, k, identify_unambiguous=True):
    assert len(scores) > 0
    N = len(scores)
    chunk_size = N // k
    means = [np.mean(scores[i*chunk_size:(i+1)*chunk_size]) for i in range(k)]
    #print('means:', means)
    means_sorted = sorted(means, reverse=True)
    #print(means_sorted[1]/means_sorted[0])
    if identify_unambiguous and means_sorted[1]/means_sorted[0] > 0.9:
        return NO_BARCODE
    else:
        return np.argmax(means)


def get_labels_gmm(D, k):
    embedding = SpectralEmbedding(n_components=4, affinity='precomputed').fit_transform(D)
    gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, n_init=50)
    gmm.fit(embedding)
    probs = gmm.predict_proba(embedding)
    labels_pred = np.array([np.argmax(probs[i]) for i in range(len(probs))])
    return labels_pred


def validator(D, main_mean, labels_true, classes, scores):
    k = len(classes)
    D_ = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
    D_[:-1, :-1] = D
    D_[-1, :-1] = scores
    D_[:-1, -1] = scores
    D_[-1, -1] = main_mean
    hard_label = get_label_spectral(D_, labels_true, classes)
    soft_label = get_label_maxmean(scores, k, True)
    if soft_label != NO_BARCODE:
        soft_label = hard_label
    return soft_label, hard_label



def validate(D, representatives, scores, classes, identify_unambiguous):
    k = len(classes)
    N = len(scores)
    #D_small = D[np.array(representatives).flatten(), :][:, np.array(representatives).flatten()]
    main_mean = np.mean([D[i, i] for i in range(len(D))])
    labels_true = np.array([len(representatives[i])*[i] for i in range(k)]).flatten()
    predictions = []
    predictions_forced = []

    pool = multiprocessing.Pool()
    worker_partial = partial(validator, D, main_mean, labels_true, classes)
    for soft, hard in tqdm(pool.imap(worker_partial, [scores[i] for i in range(N)], chunksize=1), total=N):
        predictions.append(soft)
        predictions_forced.append(hard)
    pool.close()
    pool.join()

    """
    for i in range(len(scores)):
        D_ = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
        D_[:-1, :-1] = D
        D_[-1, :-1] = scores[i]
        D_[:-1, -1] = scores[i]
        D_[-1, -1] = main_mean
        labels_true = np.array([len(representatives[i])*[i] for i in range(k)]).flatten()
        ans = get_label_maxmean(scores[i], k, identify_unambiguous)
        #ans = get_label_knn(scores[i], 3, classes, len(representatives[0]))
        if ans != NO_BARCODE:
            ans = get_label_spectral(D_, labels_true, classes)
        predictions_forced.append(get_label_spectral(D_, labels_true, classes))
        predictions.append(ans)
    """
    return predictions, predictions_forced


def get_gud_cols(S, drop, n_repr):
    colvar = np.var(S, axis=0)
    gone = []
    for i in range(4):
        gone.append(np.argsort(colvar[i*n_repr:(i+1)*n_repr])[drop:])
    return np.array(gone).flatten()


def adaptive_knn(D, n_clusters, labels_true, classes, min_k, max_k, step_k):
    min_silhouette = -1
    best_labels = None
    D = D[0] + D[1]
    for k in np.arange(min_k, max_k, step_k):
        D_ = knn_normalize(D, k)
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels="discretize",
                                        affinity='precomputed',
                                        random_state=0,
                                        n_jobs=1).fit(D_)
        labels_pred = np.array(clustering.labels_)
        #labels_pred = depermutate_labels(labels_true, labels_pred, classes)
        act_silhouette = silhouette_score(convert_to_distance(gaussian_kernel(D)), labels_pred)
        if act_silhouette > min_silhouette:
            min_silhouette = act_silhouette
            best_labels = labels_pred
    return best_labels


def plot_f1_agains_tau(init_matrix, represent_idx, score_matrix, label_true, classes):
    tau_range = np.arange(0, 1.01, 0.05)
    f1_scores = []
    for tau in tau_range:
        labels_pred, _ = validate(init_matrix, represent_idx, score_matrix, classes, tau)
        f1_scores.append(f1_score(labels_true, labels_pred, average='weighted'))
    plt.plot(f1_scores)
    plt.xticks(range(len(tau_range)), tau_range)
    plt.savefig('f1_score.png')


def discovery_aggregator(filename, train_size, classes, n_iters, n_representatives, delta, n_threads=1):
    k = len(classes)
    n_repr = 10
    representatives = [
            [ [] for i in range(k) ] for j in range(2)
        ]
    represent_idx = np.array([ list(range(i*n_repr*n_iters, (i+1)*n_repr*n_iters)) for i in range(k) ])
    for iteration in range(n_iters):
        train_data, train_labels_true = generate_sample(filename, train_size, classes)
        train_data = np.array(train_data)
        print('train data sampled with class distribution:', np.bincount(train_labels_true))
        print('starting train matrix computation ...', end='')
        D = ldtw.ComputeMatrix(train_data, os.path.join(data_path, 'scoring_scheme_med5.txt'), 0, 1,
                               bucket_size, delta, n_threads)
        print('finished')
        D = np.array(D)
        train_labels_true = np.array(train_labels_true)
        train_labels_pred = np.array(adaptive_knn(D, k, train_labels_true, classes, 5, train_size // 4, 10))
        # D, train_labels_pred, train_labels_true = drop_uncertain(D, train_labels_pred, train_labels_true)
        train_labels_pred = depermutate_labels(train_labels_true, train_labels_pred, classes)
        train_summary = correctness_summary(train_labels_pred, train_labels_true, k)
        print_summary(train_summary, 'TRAIN -- iteration {}'.format(iteration+1))
        print('Contingency of representatives:')
        class_accuracy(train_labels_pred, train_labels_true, k)

        idx, _ = choose_representatives(D, train_labels_pred, n_repr, classes)
        idx = np.array(idx)
        #print(contingency_matrix(train_labels_true[represent_idx], train_labels_pred[represent_idx]))
        for i in range(2):
            for j in range(k):
                representatives[i][j].extend(train_data[i][idx[j]])

    for i in range(2):
        representatives[i] = np.concatenate(representatives[i])

    matrix = ldtw.ComputeMatrix(representatives, os.path.join(data_path, 'scoring_scheme_med5.txt'), 0, 1,
                           bucket_size, delta, n_threads)
    return representatives, represent_idx, np.array(matrix)


def clustering_test(filename, use_classes, train_size, test_size, num_representatives, delta, N_threads, num_iters=1):
        global scores
        global labels
        identify_unambiguous = True
        k = len(use_classes)
        test_data, test_labels_true = generate_sample(filename, test_size, use_classes=use_classes)
        print('test data sampled with class distribution:', np.bincount(test_labels_true))
        test_data = np.array(test_data)
        test_labels_true = np.array(test_labels_true)
        test_labels_pred_all = []
        latest_summary = None

        for iteration in range(num_iters):
            print('Iteration {}:'.format(iteration))
            if iteration > 0:
                class_size = train_size//k
                #print(np.nonzero(test_labels_pred_all[-1] == 0)[0].shape)
                selected_idx = [np.random.choice(np.nonzero(test_labels_pred_all[-1] == barcode)[0],
                        class_size, replace=False) for barcode in range(k)]
                selected_idx = np.concatenate(selected_idx, axis=0)
                train_data = [test_data[0][selected_idx], test_data[1][selected_idx]]
                train_labels_true = np.array(test_labels_true[selected_idx])

            representatives, represent_idx, D = discovery_aggregator(filename, train_size, use_classes, 5, num_representatives, delta, N_threads)
            print('starting aligning to representatives ...', end='')
            score_matrix = ldtw.AlignToRepresentatives(representatives, test_data, os.path.join(data_path, 'scoring_scheme_med5.txt'),
                                                           1, 0, bucket_size, delta, N_threads)
            scores = score_matrix
            labels = test_labels_true
            print('finished')
            init_matrix = D[0] + D[1]
            score_matrix = np.array(score_matrix)
            score_matrix = score_matrix[0] + score_matrix[1]
            #score_matrix[score_matrix < 300] = 1

            #gud_cols = get_gud_cols(score_matrix, 5, num_representatives)
            test_labels_pred, test_labels_pred_forced = validate(init_matrix, represent_idx, score_matrix, use_classes, identify_unambiguous)
            #plot_f1_agains_tau(init_matrix, represent_idx, score_matrix, test_labels_true, use_classes)
            test_labels_pred = np.array(test_labels_pred)
            test_labels_pred_forced = np.array(test_labels_pred_forced)
            #test_labels_pred = np.array([depermutate_labels(test_labels_true, test_labels_pred[i]) for i in range(2)])
            #test_labels_pred = np.array([test_labels_pred[0][i] if test_labels_pred[0][i]==test_labels_pred[1][i] else -1 for i in range(len(test_labels_pred[0]))])
            #print(test_labels_pred)
            #print(test_labels_true)
            test_labels_pred = depermutate_labels(test_labels_true, test_labels_pred, use_classes)
            latest_summary = correctness_summary(test_labels_pred, test_labels_true, k)
            print_summary(latest_summary, 'TEST -- uncertain labels ON')
            if identify_unambiguous == True:
                test_labels_pred_forced = depermutate_labels(test_labels_true, test_labels_pred_forced, use_classes)
                latest_summary = correctness_summary(test_labels_pred_forced, test_labels_true, k)
                print_summary(latest_summary, 'TEST -- uncertain labels OFF')
            class_accuracy(test_labels_pred, test_labels_true, k)
            test_labels_pred_all.append(test_labels_pred)

        print('iterations completed')
        #test_labels_pred = np.array(test_labels_pred_all)
        #maximums = np.array([np.argmax(np.bincount(test_labels_pred[:, i])) for i in range(test_size)])
        #test_labels_pred = np.array([maximums[i] if np.sum(test_labels_pred[:, i] == maximums[i]) >= num_iters//2\
        #                    else -1 for i in range(test_size)])

        return test_data, test_labels_true, test_labels_pred_all, latest_summary


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
    print(D.shape)
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
    for k in range(10, 50, 10):
        for lam in [0.1, 0.2, 0.4, 0.6, 0.8, 1]:
            M = np.zeros(shape=D.shape) + 1
            for i in range(N):
                deg = int(k*(lam + 2*(1-lam)*rank[i]))
                neighbours = np.argsort(D[i, :])[::-1]
                for j in range(deg):
                    M[neighbours[j], i] = M[i, neighbours[j]] = 1000
            labels_pred = get_labels(M, n_classes)
            cut_value = compute_cut(D, labels_pred)
            if cut_value < min_cut:
                labels = labels_pred
                min_cut = cut_value
            #print(k, lam, cut_value)
    return labels


def plot_mislabeled(D, labels_true, labels_pred, name):
    alpha = np.zeros(D.shape)
    mislabeled = (labels_pred != labels_true)
    alpha[mislabeled, :] = 1
    alpha[:, mislabeled] = 1
    plt.imshow(D*alpha)
    plt.savefig(name)


def matrix_test(filename, num_epochs, train_size, classes, delta, N_threads):
    k = len(classes)
    summaries = []
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        train_data, labels_true = generate_sample(filename, train_size, use_classes=classes)
        train_data = np.array(train_data)
        labels_true = np.array(labels_true)
        D = np.array(ldtw.ComputeMatrix(train_data, os.path.join(data_path, 'scoring_scheme_med5.txt'), 0, 1, bucket_size, delta, N_threads))
        #return D, labels_true
        #print('ldtw finished')
        labels_pred = []
        #labels = adaptive_knn(D, k, labels_true, classes, 10, train_size//2, 10)
        #labels = RMD(D[0]+D[1], classes)
        clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed').fit(convert_to_distance(D[0]+D[1]))
        labels = clustering.labels_
        labels = depermutate_labels(labels_true, labels, classes)
        #print('one side ACC:', np.sum(labels_true==labels)/train_size*100)
        #print(list(labels)[:20])
        labels_pred = labels
       
        #plot_mislabeled(D[0]+D[1], labels_true, labels_pred, os.path.join('images', 'mislabeled{}.png'.format(epoch)))
        summary = correctness_summary(labels_pred, labels_true, k)
        print_summary(summary, 'TRAIN')
        summaries.append(summary)
        class_accuracy(labels_pred, labels_true, k)
        #labels_pred = np.array([labels_pred[0][i] if labels_pred[0][i]==labels_pred[1][i] else -1 for i in range(len(labels_pred[0]))])
        #print(labels_pred[:20])

        #rand_scores.append(adjusted_rand_score(labels_true, labels_pred))
        #silhouette_scores.append(silhouette_score(convert_to_distance(D), labels_pred))
    #print('mean ARI:', np.mean(rand_scores))
    #print('mean SC:', np.mean(silhouette_scores))
    print_summary(np.mean(summaries, axis=0), 'final')


#matrix_test(args.dataset + '.hdf5', args.n_iters, args.sample_size, tuple(args.barcodes), args.delta, args.threads)
metrics = []
for i in range(10):
    print('Epoch {}'.format(i))
    data, labels_true, labels_pred_all, summary = clustering_test(args.dataset+'.hdf5',
                                                            tuple(args.barcodes),
                                                            args.sample_size, args.test_size,
                                                            args.representatives, args.delta,
                                                            args.threads,
                                                            num_iters=args.n_iters)
    metrics.append(summary)
print('Mean performance:')
print_summary(np.mean(metrics, axis=0), 'Mean of all runs')
