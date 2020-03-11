import matplotlib.pyplot as plt
import h5py


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


f = h5py.File('C:\\Users\\adria\\PycharmProjects\\BarcCluster\\barcode_dataset.hdf5', 'r')
counts, labels = [], []
explode = (0, 0, 0, 0, 0.1)
for key in f.keys():
    counts.append(len(f[key]))
    labels.append(key)

plt.pie(counts, explode=explode, labels=labels, autopct=make_autopct(counts), startangle=90)
plt.title('Training data distribution')
plt.show()
