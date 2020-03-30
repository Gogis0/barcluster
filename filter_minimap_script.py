import os
import numpy as np
from constants import data_path


with open(os.path.join(data_path, 'all2.tab'), 'r') as f:
    lines = f.readlines()
    filtered = []
    for line in lines:
        line = line.split('\t')
        name = line[0]
        alignments = list(map(int, line[1:5]))
        coverages = np.array(list(map(float, line[-4:])))
        #coverages_sorted = np.array(sorted(coverages, reverse=True))
        #alignments_sorted = sorted(alignments, reverse=True)
        num_significant = np.sum(coverages > 85)
        if num_significant >= 2 or num_significant == 0: continue
        filtered.append((name, np.argmax(coverages)))
    with open(os.path.join(data_path, 'all2_filtered_2.txt'), 'w') as of:
        for i in range(len(filtered)):
            of.write('{} {}\n'.format(filtered[i][0], filtered[i][1]))
