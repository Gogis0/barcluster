import os, sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from my_dtw import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn import linear_model


abs_dist = lambda x, y: abs(x - y)
sq_dist = lambda x, y: (x - y)**2

def dtw_metric(x, y):
    return fastdtw(x, y , dist=abs_dist)[0]


def DTW(x, y):
    n, m = x.shape[0], y.shape[0]
    A = np.zeros((n+1, m+1))
    A = np.add(A, 1000000000000)
    A[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (x[i-1]-y[j-1])**2
            A[i, j] = cost + min(A[i-1, j], A[i, j-1], A[i-1, j-1])

    #print(A)
    #plt.imshow(A[1:, 1:].astype(float))
    #plt.show()
    print(np.min(A[1:, 1:]))
    #exit()
    return np.min(A[-100:n, -100:m])

l2_norm = lambda x, y: (x - y) ** 2

def dtw_metric(x, y):
    return dtw(x, y, dist=euclidean)[0]

def ls_scaling(reg, path, signal_1, signal_2):
    N = path[0].shape[0]
    A = signal_1[path[0]]
    B = signal_2[path[1]]
    reg.fit(A.reshape(-1, 1), B)
    return reg.coef_[0], reg.intercept_

path = sys.argv[1]
N_choice = int(sys.argv[2])
barcodes = pickle.load(open(path, 'rb'))
data = []
labels = []

for key, val in barcodes.items():
    print(val[1])
    data.append(tuple(val[0]))
    labels.append(val[1])

data = np.array(data)
labels = np.array(labels)
D = np.zeros((data.shape[0], data.shape[0]))
reg = linear_model.LinearRegression()
aligned = []
for i in range(len(data)):
    for j in range(i+1, len(data)):
        print('#', i, j)
        sig1, sig2 = np.array(data[i]), np.array(data[j])
        #for k in range(2):
        #    _, _, _, path = dtw(sig1, sig2, dist=euclidean)
        #    a, b = ls_scaling(reg, path, sig1, sig2)
        #    sig1 = a*sig1 + b
        d, cost_matrix, _, path = dtw(sig1, sig2, dist=abs_dist)
        D[i,j] = d[0]


#DTW(np.array(data[2]), np.array(data[10]))
print(D)
plt.imshow(D)
plt.show()

same, diff = [], []
for i in range(20):
    for j in range(i + (5 - (i % 5)), 20):
        diff.append(D[i, j])
for i in range(20):
    for j in range(i + (5 - (i % 5)), 20):
        diff.append(D[i, j])

plt.hist(same, alpha=0.3, label='same barcodes')
plt.hist(diff, alpha=0.3, label='different barcodes')
plt.legend(loc='upper left')
plt.show()

N = len(data)
idx = np.random.choice(N, N_choice)
print(idx)
chosen = data[idx]

for i in range(4,6):
    plt.plot(data[i])

plt.figure(figsize=(20,12))
plt.title("{} raw signals of barcodes".format(N_choice))
plt.xlabel("Index")
plt.ylabel("Signal value")
plt.show()
