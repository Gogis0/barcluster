import h5py
import logging
import numpy as np
from my_scoring import *
from util import z_normalize, trim_blank
from constants import data_path, workplace_path, barcode_list, train_size, prefix_length

np.random.seed(126)


def sample_unlcassified(file, count=None):
    f = h5py.File(os.path.join(data_path, file), 'r')
    keys = list(f['unclassified'].keys())
    if count is None:
        count = len(keys)
    key_draft = np.random.choice(keys, count, replace=False)

    data = []
    for key in key_draft:
        data.append(z_normalize(trim_blank(f['unclassified'][key]).astype(float))[:prefix_length])
    return data


def sample_unsorted(file, counts=(100, 100, 100, 100), both_ends=True):
    assert type(counts) is tuple
    names_dict = {}
    barcodes = {}
    with open(os.path.join(workplace_path, 'all2_filtered.txt'), 'r') as all:
        lines = all.readlines()
        for line in lines:
            name, barcode = line.split()
            names_dict[name] = 1
            barcodes[name] = barcode

    data = [[] for _ in range(len(counts))]
    names = [[] for _ in range(len(counts))]
    labels = [[] for _ in range(len(counts))]
    logging.info('Starting the sampling process')
    with h5py.File(os.path.join(data_path, file), 'r') as f:
        keys = list(f.keys())
        np.random.shuffle(keys)
        for key in keys:
            print([len(x) for x in data])
            if key not in names_dict:
                continue
            if len(f[key]) > 100000:
                continue
            barcode = int(barcodes[key])
            actual_count = len(data[barcode])
            if actual_count == counts[barcode]:
                logging.info('Bucket {} is already filled'.format(barcode))
                continue

            selected_windows = [(z_normalize(trim_blank(f[key]).astype(float))[:prefix_length])]
            if both_ends:
                reversed_signal = np.array(f[key]).astype(float)[::-1]
                selected_windows.append(z_normalize(trim_blank(reversed_signal))[:prefix_length])

            data[barcode].append(selected_windows)
            names[barcode].append(key)
            labels[barcode].append(barcode)
    logging.info('Sampling done')

    data = np.concatenate(data)
    names = np.concatenate(names)
    labels = np.concatenate(labels)
    return data, names, labels


def sample_reads(file, counts=None):
    f = h5py.File(os.path.join(data_path, file), 'r')
    data = []
    names = []
    labels = []

    for i in range(len(counts)):
        logging.info('Sampling {} reads of class={}'.format(counts[i], i))
        barcode = barcode_list[i]
        keys = list(f[barcode].keys())[train_size:]
        N = min(counts[i], len(keys))
        #keys = [k for k in keys if f[barcode][k].attrs['score'] > 60]
        for key in np.random.choice(keys, N):
            signal = np.array(f[barcode][key])
            data.append(z_normalize(trim_blank(signal).astype(float))[:prefix_length])
            names.append(key)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels)
    return data, names, labels


sampled_data, sampled_names, sampled_labels = sample_unsorted('validation_dataset_big.hdf5', counts=(500,)*4)
with open(os.path.join(data_path, 'matrix_2000_both.txt'), 'w') as f:
    logging.info('Writing reads to file: {}'.format(f))
    N_reads = len(sampled_data)
    f.write('{} {} {}\n'.format(N_reads, len(sampled_data[0]), prefix_length))
    for i in range(N_reads):
        f.write(sampled_names[i] + '\n')
        for j in range(len(sampled_data[i])):
            f.write(' '.join(str(x) for x in sampled_data[i][j]) + '\n')
    logging.info('Writing done')
f.close()

