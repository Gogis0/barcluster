import numpy as np
from sklearn.linear_model import LinearRegression
from my_dtw import GlobalDTW

def z_normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x


def ls_normalize(squiggle1, squiggle2, n_iter=1):
    """
    Estimates the best normalization parameters using iterated DTW.
    Reference: https://arxiv.org/abs/1705.01620
    :param squiggle1: the first squiggle
    :param squiggle2: the second squiggle
    :param n_iter: the number of iterations performed
    :return: scale (A) and shift (B) parameters for normalization
    """
    assert n_iter >= 0
    dtw = GlobalDTW()
    A, B = 1, 0
    for i in range(n_iter):
        dtw.align(np.array(squiggle1), np.array(squiggle2))
        path = dtw.get_path(dtw.A)
        aligned1, aligned2 = dtw.get_aligned(squiggle1, squiggle2, path)
        reg = LinearRegression().fit(np.reshape(aligned1, (-1, 1)), aligned2)
        A, B = reg.coef_, reg.intercept_
        squiggle1 = A * squiggle1 + B
    return A, B


def median_normalize(x):
    m = np.median(x)
    d = np.median([abs(r-m) for r in x])
    return (x - m)/d


def moving_average(x, window):
    return np.array([np.mean(x[i:i+window]) for i in range(len(x)-window)])


def moving_median(x, window):
    return np.array([np.median(x[i:i+window]) for i in range(len(x)-window)])


def add_noise(signal, sigma):
    # adds gaussian noise
    return signal + np.random.normal(scale=sigma, size=len(signal))


def augment(sig, kmer_len):
    N = len(sig)
    values = []
    idx = np.random.choice(N, N//4)
    for i in idx: values.append(sig[i]*200)
    sig = np.insert(sig, idx, sig[idx])
    return sig


def trim_blank(sig, window=300):
    # cut off the blank signal prefix
    N = len(sig)
    prefix_size = min(5000, N)
    #sig = [np.var(sig[i:i+window]) for i in range(prefix_size-window)]
    variances = [np.var(sig[i:i+window]) for i in range(N//2, N-window, window)]
    mean_var = np.mean(variances)
    trim_idx = 20
    while window > 5:
        while np.var(sig[trim_idx: trim_idx + window]) < 0.3*mean_var:
            trim_idx += 1
        window //= 2

    #return sig, trim_idx
    return sig[trim_idx:]


