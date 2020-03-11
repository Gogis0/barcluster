import numpy as np
from math import log10


class LocalDTW:
    def __init__(self, scoring_scheme):
        self.scoring_scheme = scoring_scheme
        self.dist = None
        self.A = None
        self.C = None
        self.path = None

    def align(self, x, y):
        assert self. scoring_scheme is not None
        n, m = x.shape[0], y.shape[0]
        self.A = np.zeros((n + 1, m + 1))

        self.C = np.zeros((n+1, m+1))
        for i in range(1, n+1):
            for j in range(1, m+1):
                self.dist = abs(x[i - 1] - y[j - 1])
                cost = self.scoring_scheme[self.dist]
                self.C[i, j] = cost
                self.A[i, j] = max(0, cost + max(self.A[i - 1, j], self.A[i, j - 1]))

        for i in range(1, n+1):
            for j in range(1, m+1):
                self.A[i, j] -= log10(i+j)

        return np.max(self.A)

    def get_path(self, A):
        self.path = ([], [])
        act = np.unravel_index(np.argmax(A), A.shape)
        while True:
            self.path[0].append(act[0])
            self.path[1].append(act[1])
            next_idx, max_val = act, 0
            if A[act[0] - 1, act[1]] > max_val:
                next_idx = (act[0] - 1, act[1])
                max_val = A[act[0] - 1, act[1]]
            if A[act[0], act[1] - 1] > max_val:
                next_idx = (act[0], act[1] - 1)
            if next_idx == act:
                break
            act = next_idx

        self.path[0].append(act[0])
        self.path[1].append(act[1])
        self.path = (self.path[0][::-1], self.path[1][::-1])
        return self.path

    @staticmethod
    def get_aligned(sig1, sig2, path):
        res1, res2 = [], []
        for (x, y) in zip(*path):
            res1.append(sig1[x - 1])
            res2.append(sig2[y - 1])
        return res1, res2


class GlobalDTW:
    def __init__(self):
        self.dist = None
        self.A = None
        self.C = None
        self.path = None

    def align(self, x, y):
        n, m = x.shape[0], y.shape[0]
        self.A = np.zeros((n + 1, m + 1))
        self.C = np.zeros((n+1, m+1))
        for i in range(n+1): self.A[i, 0] = 2*100
        for i in range(m+1): self.A[0, i] = 2*100
        self.A[0, 0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                self.dist = abs(x[i - 1] - y[j - 1])
                self.C[i, j] = self.dist
                self.A[i, j] = self.dist + min(self.A[i - 1, j - 1], self.A[i - 1, j], self.A[i, j - 1])

        return self.A[n, m]

    def get_path(self, A):
        i, j = A.shape[0] - 1, A.shape[1] - 1
        self.path = ([i-1], [j-1])
        while i > 1 and j > 1:
            trace = np.argmin((self.A[i - 1, j - 1], self.A[i - 1, j], self.A[i, j - 1]))
            if trace == 0:
                i -= 1
                j -= 1
            elif trace == 1:
                i -= 1
            else:
                j -= 1
            self.path[0].append(i-1)
            self.path[1].append(j-1)

        self.path = (self.path[0][::-1], self.path[1][::-1])
        return self.path

    @staticmethod
    def get_aligned(sig1, sig2, path):
        res1, res2 = [], []
        for (x, y) in zip(*path):
            res1.append(sig1[x - 1])
            res2.append(sig2[y - 1])
        return res1, res2
