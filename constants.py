# directory where raw data is stored
workplace_path = 'C:\\Users\\adria\\Documents\\workspace'

# path to all the cultivated data
data_path = 'C:\\Users\\adria\\PycharmProjects\\BarcCluster\\data\\'

# files to load the sampled errors from
in_aligned = 'aligned_pointwise_distances_avg.txt'
in_random = 'random_pointwise_distances_avg.txt'

# list of barcode aliases
barcode_list = ['barcode01', 'barcode02', 'barcode03', 'barcode04']

# a dictionary of basecalled barcode sequences
barcodes_dict = {
    'flank_front:':  'AAGGTT ',
    'flank_rear': '   ACAGCACCT',
    'barcode01':     'CACAAAGACACCGACAACTTTCTT',
    'barcode02':     'ACAGACGACTACAAACGGAATCGA',
    'barcode03':     'CCTGGTAACTGGGACACAAGACTC',
    'barcode04':     'TAGGGAAACACGATAGAATCCGAA',
}

# the number of reads from each barcode class used for scoring scheme determination
train_size = 75

# the length of the read prefix we assume for finding barcodes
prefix_length = 1000

####################################################################################################
# scoring scheme parameters:
bucket_size = 10
num_buckets = 100
max_distance = 50
low_treshold = 0.01
delta = 0.5
####################################################################################################
