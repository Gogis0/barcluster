import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS


def make_alignment_figure(squiggle1, squiggle2, path, warping_matrix, barcode_positions=None):
    """
    Plots the two aligned squiggles together with their alignment matrix and warping path in a double plot.
    :param squiggle1: a squiggle aligned with 'squiggle2'
    :param squiggle2: a squiggle aligned with 'squiggle1'
    :param path: the warping path of the alignemnt
    :param warping_matrix: the cumulative alignment score matrix
    :param barcode_positions: 4-tuple of starting and ending positions of barcodes in the two squiggles
    :return: the built figure ready to be plotted
    """
    fig = plt.figure(figsize=(16, 4), constrained_layout=True)
    spec = fig.add_gridspec(1, 2, width_ratios=[5, 1])
    fig_left = fig.add_subplot(spec[0, 0])
    fig_right = fig.add_subplot(spec[0, 1])
    fig_left.plot(squiggle1)
    fig_left.plot(squiggle2)
    fig_right.imshow(warping_matrix)
    fig_right.plot(path[1], path[0], color='red')
    if barcode_positions is not None:
        #axs[0].axvline(x=np.argmin(dtw.path[0]!=start1), color='orange',  linestyle='--', label='barcode start')
        #axs[0].axvline(x=np.argmin(dtw.path[0]!=end1), color='orange',  linestyle='--', label='barcode end')
        #axs[0].axvline(x=np.argmin(dtw.path[1]!=start2), color='yellow',  linestyle='--', label='barcode start')
        #axs[0].axvline(x=np.argmin(dtw.path[1]!=end2), color='yellow',  linestyle='--', label='barcode end')
        fig_right.axhline(y=barcode_positions[0], color='orange',  linestyle='--', label='barcode start')
        fig_right.axhline(y=barcode_positions[1], color='orange',  linestyle='--', label='barcode end')
        fig_right.axvline(x=barcode_positions[2], color='yellow',  linestyle='--', label='barcode start')
        fig_right.axvline(x=barcode_positions[3], color='yellow',  linestyle='--', label='barcode end')
    return fig


def plot_embedding(points, k, predictions, true_class):
    """
    Plots an embedding of representatives and a new point into a 2D plane.
    :param points: 2D numpy array of points
    :param k: the number of clusters
    :param predictions:
    :param true_class: ground truth class of the new point
    :return: nothing
    """
    assert points.shape[1] <= 2 # points should be 2D
    class_size = (len(points) - 1) // k
    colors = ['red', 'green', 'blue', 'black']
    print(k, points.shape)
    for i in range(k):
        start, end = i*class_size, (i+1)*class_size
        plt.scatter(points[start:end, 0], points[start:end, 1], marker='o', c=colors[i], label='barcode {}'.format(i+1))
    plt.scatter(points[-1, 0], points[-1, 1], marker='o', label='new read')
    plt.legend(loc='upper right')
    plt.title('predicted class = {}; ground truth class = {}'.format(predictions, true_class+1))
    plt.show()


def plot_scores(scores, prediction, ground_truth):
    plt.bar(range(0, len(scores)), scores)
    plt.title('predicted class = {}; ground truth class = {}'.format(prediction+1, ground_truth+1))
    plt.xlabel('Representatives')
    plt.ylabel('Alignment score')
    plt.show()


def matrix_balanced_imshow(D, cluster_sizes):
    num_clusters = len(cluster_sizes)
    positions = [sum(cluster_sizes[:i]) + cluster_sizes[i]//2 for i in range(num_clusters)]
    plt.xticks(positions, ['barcode {}'.format(i+1) for i in range(num_clusters)])
    plt.yticks(positions, ['barcode {}'.format(i+1) for i in range(num_clusters)])
    plt.imshow(D)
    plt.show()


def plot_equiloaded_matrix(N, num_threads):
    M = np.zeros((N, N))
    indices = np.array([
        (i, j) for i in range(N) for j in range(i+1, N)
    ])
    chunk_size = len(indices) // num_threads
    for thread in range(num_threads):
        start, end = thread*chunk_size, (thread+1)*chunk_size
        for i, j in indices[start:end]:
            M[i, j] = thread + 1
    plt.imshow(M, cmap='inferno')
    plt.show()


def mds_plot(D, dim, cluster_sizes):
    embedding = MDS(n_components=dim, dissimilarity='precomputed', n_jobs=4)
    X_transformed = embedding.fit_transform(D)
    k = len(cluster_sizes)
    index = 0
    if dim == 2:
        for i in range(k):
            plt.scatter(X_transformed[index:index+cluster_sizes[i], 0],
                        X_transformed[index:index+cluster_sizes[i], 1],
                        label='barcode {}'.format(i+1), s=8)
            index += cluster_sizes[i]
        plt.title('Stress: {0:.4f}'.format(embedding.stress_/np.sum(D**2)))
        plt.legend(loc='upper right')
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(k):
            ax.scatter(X_transformed[index:index+cluster_sizes[i], 0],
                       X_transformed[index:index+cluster_sizes[i], 1],
                       X_transformed[index:index+cluster_sizes[i], 2],
                       label='barcode {}'.format(i+1))
            index += cluster_sizes[i]
        plt.title('Stress: {0:.4f}'.format(embedding.stress_/np.sum(D**2)))
        plt.legend(loc='upper right')
        plt.show()


def plot_contingency(M):
    """
    seaborn is now broken, we must fix it manually
    https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    """
    labels = ['barcode {}'.format(i+1) for i in range(len(M))]
    plt.figure(figsize=(8, 8))
    sn.set(font_scale=1.4)
    ax = sn.heatmap(M, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cbar=False)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


def plot_matrix_labelings(D, labels_true, labels_pred):
    """
    A function to plot the score matrix with the rows sorted according to labels.
    The matrix is plotted two times: once for ground truth labels and once for the predicted ones.
    :param D: matrix of scores
    :param labels_true: ground truth labels
    :param labels_pred: predicted labels
    :return: nothings, plots the images
    """
    sorted_idx_true = np.argsort(labels_true)
    sorted_idx_pred = np.argsort(labels_pred)
    for idx in [sorted_idx_true, sorted_idx_pred]:
        M = np.array(D.shape)
        for i in range(D.shape[0]):
            for j in range(i, D.shape[1]):
                M[i, j] = M[j, i] = D[idx[i], idx[j]]
        plt.imshow(M)
        plt.show()

