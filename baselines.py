import numpy as np
import scipy
from scipy.sparse import issparse, csc_matrix, csr_matrix
import torch
from scipy.linalg import hadamard

def srht_slow(A, sketch_size, ind=None, signs=None):

    n = A.shape[0]

    n_padded = 2 ** (np.int64(np.ceil(np.log(n) / np.log(2))))
    if n_padded > n:
        A = np.vstack([A, np.zeros((n_padded-n, A.shape[1]), dtype=np.float32)])

    rng = np.random.default_rng()

    n = A.shape[0]

    if ind is None:
        ind = rng.choice(n, sketch_size, replace=False)
    if signs is None:
        signs = rng.choice([-1, 1], n, replace=True).reshape((-1, 1))

    if issparse(A):
        B = A.toarray()
    else:
        B = A.copy()

    if A.shape[0] == 1:
        B = B.reshape(-1,)

    B = signs * A

    h = 1
    while h < B.shape[0]:
        for i in range(0, B.shape[0], h * 2):
            for j in range(i, i + h):
                x = B[j].copy()
                y = B[j + h].copy()
                B[j] = x + y
                B[j + h] = x - y
        h *= 2

    if A.shape[0] == 1:
        B = B.reshape(1, -1)

    return B[ind] / np.sqrt(len(ind))
