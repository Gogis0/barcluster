import os
import numpy as np
from double_boxplot import make_double_boxplot
from constants import data_path

inter_list, cross_list = [], []
delta_list = []
with open(os.path.join(data_path, 'delta_avg.out'), 'r') as f:
    num_deltas = int(next(f))
    for i in range(num_deltas):
        delta = float(f.readline())
        delta_list.append(delta)
        num_inter = int(f.readline())
        inter_list.append(list(map(float, [f.readline() for _ in range(num_inter)])))
        num_cross = int(f.readline())
        cross_list.append(list(map(float, [f.readline() for _ in range(num_cross)])))
    make_double_boxplot(delta_list, (inter_list, cross_list), 'images/score_boxplot_raw.png')

    for i in range(num_deltas):
        print(np.mean(inter_list[i]), np.mean(cross_list[i]))
