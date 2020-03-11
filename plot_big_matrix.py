import matplotlib.pyplot as plt
import numpy as np
from constants import *
import sklearn
from sklearn.manifold import MDS

f = open(data_path+'dtw_out.csv', 'r')
N = int(next(f))

D = np.zeros((N, N))
for i in range(N):
    row = list(map(float, f.readline().split(',')))
    for j in range(N):
        D[i, j] = D[j, i] = row[j]

#D = D.max() - D
#for i in range(N): D[i, i] = 0
D[D < 300] = 0
plt.imshow(D)
plt.show()

embedding = sklearn.manifold.MDS(n_components=3,
                                 dissimilarity='precomputed',
                                 n_init=500,
                                 max_iter=30000000,
                                 eps=0.000000001,
                                 n_jobs=4)

#X = embedding.fit_transform(D)

