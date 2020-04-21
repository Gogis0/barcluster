import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from constants import data_path


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_distribution(filenames, dataset_names, n_barcodes=12):
    bincounts = []
    for filename in filenames:
        f = open(os.path.join(data_path, filename), 'r')
        labels = [int(x.split()[1]) for x in f.readlines()]
        bincounts.append(np.bincount(labels, minlength=n_barcodes))

    x = np.arange(n_barcodes)  # the label locations
    y_labels = [str(i+1) for i in range(n_barcodes)]
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bincounts[0], width, label=dataset_names[0])
    rects2 = ax.bar(x + width/2, bincounts[1], width, label=dataset_names[1])
    ax.set_xticks(x)
    ax.set_xticklabels(y_labels)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.tight_layout()
    plt.title('Training data distribution')
    plt.show()


plot_distribution(['all_deepbinner.txt', 'all2_filtered_3.txt'], ['deepbinner', 'base'])
