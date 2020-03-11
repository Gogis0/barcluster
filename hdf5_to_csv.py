import os
import h5py
from constants import workplace_path


def hdf5_to_csv(h_file, csv_file):
    with h5py.File(os.path.join(workplace_path, h_file), u"r") as hf:
        keys = list(hf.keys())
        with open(os.path.join(workplace_path, csv_file), u"r") as cf:
            for key in keys:
                pass
