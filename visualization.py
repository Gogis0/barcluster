import matplotlib.pyplot as plt


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

