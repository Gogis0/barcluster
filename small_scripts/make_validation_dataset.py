import os
import h5py
import numpy as np
from constants import workplace_path, data_path


of = h5py.File(os.path.join(workplace_path, "validation_dataset_deepbinner.hdf5"), "a")

#path = os.path.join(workplace_path, 'validation')
path = '/extdata2/deepbinner-dataset/data'
for root, _, files in os.walk(path, topdown=False):
    for file in files:
        print(file)
        f = h5py.File(os.path.join(root, file), u"r")
        read_number = (list(f['Raw']['Reads']))[0]
        read_id = str(f['Raw']['Reads'][read_number].attrs['read_id'].decode("utf-8"))
        print(np.array(f['Raw']['Reads'][read_number]['Signal']))
        dataset = of.create_dataset(read_id, data=f['Raw']['Reads'][read_number]['Signal'])

of.close()
