import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
from constants import data_path, in_aligned, in_random,\
    bucket_size, num_buckets, max_distance, low_treshold, delta

logging.basicConfig(level=logging.INFO)


class ScoringScheme:
    def __init__(self, aligned_file=in_aligned, random_file=in_random,
                 bucket_size=bucket_size, num_buckets=num_buckets, low_treshold=low_treshold,
                 max_distance=max_distance, delta=delta):
        self.aligned_file = aligned_file
        self.random_file = random_file
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        self.low_treshold = low_treshold
        self.max_distance = max_distance
        self.delta = delta
        self._load_score()

    def _load_score(self):
        assert self.aligned_file is not None
        assert self.random_file is not None

        with open(os.path.join(data_path, self.aligned_file)) as f:
            logging.info('Loading aligned errors from file: {}'.format(self.aligned_file))
            align = list(map(float, f.readlines()))
        with open(os.path.join(data_path, self.random_file)) as f:
            logging.info('Loading random errors from file: {}'.format(self.random_file))
            random = list(map(float, f.readlines()))

        ahist, rhist = [0] * (num_buckets + 1), [0] * (num_buckets + 1)

        for x in align:
            x = min(x, bucket_size)
            ahist[int(x * bucket_size)] += 1
        for x in random:
            x = min(x, bucket_size)
            rhist[int(x * bucket_size)] += 1

        self.ahist = np.array(ahist)
        self.rhist = np.array(rhist)
        self.score = self.ahist / self.rhist
        self.score[self.score < self.low_treshold] = self.low_treshold
        self.score = [(log10(self.score[i]) - self.delta) for i in range(self.max_distance + 1)]
        logging.info('Scoring scheme created successfully')

        #self.advanced_score = []
        #for i in range(len(self.score)-1):
        #    for x in np.arange(0, 1, 0.01):
        #        self.advanced_score.append(self.score[i] + x*(self.score[i+1]-self.score[i]))

    def _write_score_to_file(self, filename):
        with open(os.path.join(data_path, filename), 'w') as f:
            f.write('{} {}\n'.format(num_buckets, bucket_size))
            f.write(' '.join(str(x) for x in self.score))

    def __getitem__(self, key):
        key = int(key*self.bucket_size)
        key = min(key, self.max_distance)
        return self.score[key]
        #key = min(key, 4.5)
        #return self.advanced_score[int(key*1000)]


def plot_score_histogram(x_axis, aligned, random):
    plt.bar(x_axis, aligned, label='aligned')
    plt.bar(x_axis, random, label='random')
    plt.legend(loc='upper left')
    plt.show()
