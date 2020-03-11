import numpy as np


def trim_blank(sig, window=300):
    N = len(sig)
    variances = [np.var(sig[i:i+window]) for i in range(N//2, N-window, window)]
    mean_var = np.mean(variances)
    trim_idx = 20
    while window > 5:
        while np.var(sig[trim_idx: trim_idx + window]) < 0.3*mean_var:
            trim_idx += 1
        window //= 2

    #return sig, trim_idx
    return sig[trim_idx:]
