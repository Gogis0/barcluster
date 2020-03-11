import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
from util import *
from constants import *

bc1 = '1'
f = h5py.File(data_path+'barcoded_reads.hdf5', 'r')
np.random.seed(1)


def auto_trim_test(test_size, window=300):
    errors = []
    for barcode in barcode_list:
        print(barcode)
        for read_id in np.random.choice(list(f[barcode].keys()), test_size, replace=False):
            read = f[barcode][read_id]
            _, cut_idx = trim_blank(np.array(read), window)
            if cut_idx-read.attrs['start'] < -2000 or cut_idx-read.attrs['start'] > 0:
                print(read_id, cut_idx, read.attrs['start'])
                plot_trim(read, barcode, read_id, cut_idx)
            errors.append(cut_idx-read.attrs['start'])

    plt.hist(errors, bins=60, edgecolor='black', linewidth=0.7)
    plt.xlabel('trim_index-barcode_start')
    plt.ylabel('Count')
    plt.title('Error distribution of prefix cutting for window={}'.format(window))
    plt.show()
    return np.mean(errors)


def plot_trim(sig, barcode, read_id, cut_idx):
    plt_len = min(len(sig), 5000)  # only consider the first 1000 values
    plt.plot(sig[:plt_len], label='raw signal')
    plt.axvline(x=cut_idx, color='red', linestyle='--', label='trim index')
    plt.axvline(x=f[barcode][read_id].attrs['start'], color='yellow', linestyle='--', label='barcode start')
    plt.axvline(x=f[barcode][read_id].attrs['end'], color='purple', linestyle='--', label='barcode start')
    # plt.plot(sig)
    plt.legend(loc='upper right')
    plt.xlabel('Signal value')
    plt.xlabel('Signal index')
    plt.title('Cutting of the blank signal for read id {}'.format(read_id))
    plt.show()


def visual_trim_test(barcode, num=100):
    id_list = np.random.choice(list(f[barcode].keys()), num)
    for read_id in id_list:
        bar = np.array(f[barcode][read_id])
        plt_len = min(len(bar), 5000)  # only consider the first 1000 values
        plt.plot(bar[:plt_len], label='raw signal')
        sig, cut_idx = trim_blank(bar, 300)
        plt.axvline(x=cut_idx, color='red',  linestyle='--', label='trim index')
        plt.axvline(x=f[barcode][read_id].attrs['start'], color='yellow',  linestyle='--', label='barcode start')
        plt.axvline(x=f[barcode][read_id].attrs['end'], color='purple',  linestyle='--', label='barcode start')
        #plt.plot(sig)
        plt.legend(loc='upper right')
        plt.xlabel('Signal value')
        plt.xlabel('Signal index')
        plt.title('Cutting of the blank signal for read id {}'.format(read_id))
        plt.show()
        #plt.plot(sig)
        #plt.title('Variances of the signal for read id {}'.format(read_id))
        #plt.show()
