#!/bin/python
import os, sys
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file


def get_raw_data(fast5_filepath, prefix_size):
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        print(f5)
        for read_id in f5.get_read_ids():
            read = f5.get_read(read_id)
            raw_data = read.get_raw_data()
            return 1

reads = []
path = os.path.realpath(sys.argv[1])
prefix_size = 100
#out_path = sys.argv[3]

for f in os.listdir(path):
    reads.append(get_raw_data(os.path.join(path, f), prefix_size))

#np.savetxt(out_path, np.array(reads).astype(int), fmt='%i')
