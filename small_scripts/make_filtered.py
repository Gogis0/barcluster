import h5py

selected = h5py.File('/projects2/nanopore-2016/barcode_cluster/validation_dataset_deepbinner.hdf5', 'r')

with open('wgs_classifications.tsv', 'r') as f:
    of = open('all_deepbinner.txt', 'w')
    for line in f:
        line = line.split()
        try:
            line[-1] = int(line[-1])
            selected[line[0]]
            of.write('{} {}\n'.format(line[0], line[-1]))
        except:
            print(line[-1])
of.close()
