import matplotlib.pyplot as plt
import numpy as np
from math import log10

bucket_size = 10
num_buckets = 101

data_path = 'C:\\Users\\adria\\PycharmProjects\\BarcCluster\\data\\'

f = open(data_path+'pos_align_20.txt')
align = list(map(float, f.readlines()))
f.close()

f = open(data_path+'pos_random_20.txt')
random = list(map(float, f.readlines()))
f.close()

ahist, rhist = [0]*num_buckets, [0]*num_buckets
x_axis = np.arange(0, bucket_size+0.1, 0.1)

for x in align:
    x = min(x, bucket_size)
    ahist[int(x*bucket_size)] += 1
for x in random:
    x = min(x, bucket_size)
    rhist[int(x*bucket_size)] += 1

ahist = np.array(ahist)
rhist = np.array(rhist)
score = ahist/rhist
score[score < 0.01] = 0.01

plt.bar(x_axis, height=ahist, width=0.1, alpha=0.6, label='aligned positions', align='edge')
plt.bar(x_axis, height=rhist, width=0.1, alpha=0.6, label='random positions', align='edge')
plt.legend(loc='upper right')
plt.show()

plt.clf()
plt.plot([log10(x) for x in score])
plt.title('Scoring scheme for local DTW')
plt.xlabel('Absolute error (multiplied by 10)')
plt.show()
