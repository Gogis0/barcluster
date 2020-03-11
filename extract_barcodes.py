#!/bin/python
import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt


def get_barcode(fast5_filepath):
    f = h5py.File(fast5_filepath, "r")
    #print(f)
    analyses = f['Analyses']

    #barcode_summary = analyses['Barcoding_000']['Summary']['barcoding']
    if 'Barcoding_000' not in analyses:
        return None, -1, -1, -1

    attrs = analyses['Barcoding_000']['Summary']['barcoding'].attrs
    barcode_front_start, barcode_front_length = attrs['front_begin_index'], attrs['front_foundseq_length']
    barcode_rear_start, barcode_rear_length = attrs['front_rear_index'], attrs['rear_foundseq_length']
    #print(bc['front_score'])
    #if bc['front_score'] < 70:
    #    return  None, -1, -1
    signal = list(f['Raw']['Reads'][list(f['Raw']['Reads'])[0]]['Signal'])

    df_events = pd.DataFrame(np.array(analyses['Basecall_1D_000']['BaseCalled_template']['Events']),
                    columns=['mean', 'start', 'stdv', 'length', 'model_state', 'move', 'p_model_state', 'weights'])

    pos, idx = df_events.loc[0, 'length'], 0
    raw_front = (-1, -1)
    raw_rear = (-1, -1)
    while pos < barcode_rear_start + barcode_rear_length:
        if raw_front[0] == -1 and pos >= barcode_front_start:
            raw_front[0] = df_events.loc[idx, 'start']
        if pos < barcode_front_start + barcode_front_length:
            raw_front[1] = df_events.loc[idx, 'start']

        if raw_rear[0] == -1 and pos >= barcode_front_start:
            raw_rear[0] = df_events.loc[idx, 'start']
        raw_front[1] = df_events.loc[idx, 'start']
        pos += df_events.loc[idx, 'move']
        idx += 1

    return signal, attrs, raw_front, raw_rear


idx, cnt = 0, 0
#starts, ends = [], []

data_path = 'C:\\Users\\adria\\Documents\\workspace'
of = h5py.File("validation_dataset.hdf5", "a")

for bucket in ['unclassified', 'barcode01', 'barcode02', 'barcode03', 'barcode04']:
    grp = of.create_group(bucket)
    for res in ['fail', 'pass']:
        path = os.path.join(data_path, res, bucket, '0')
        print(path)
        for root, _, files in os.walk(path, topdown=False):
            for f in files:
                idx += 1
                signal, start, end, attrs = get_barcode(os.path.join(root, f))
                if signal is not None:
                    #starts.append(start)
                    #ends.append(end)
                    dset = grp.create_dataset(str(idx), data=signal)
                    for key, val in attrs.items():
                        dset.attrs[key] = val
                    dset.attrs['result'] = res
                    dset.attrs['start'] = start
                    dset.attrs['end'] = end
                    cnt += 1

                print(idx, 'barcode:', bucket)

of.close()

#plt.hist(starts, 50, facecolor='g', alpha=0.75)
#plt.hist(ends, 50, facecolor='b', alpha=0.75)
#plt.legend(['start indices', 'end indices'])
#plt.xlabel('Barcode indices')
#plt.ylabel('Count')
#plt.title('Histogram of barcode positions')
#plt.grid(True)
#plt.show()
