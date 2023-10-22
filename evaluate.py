import numpy as np
import scipy
from scipy.sparse import issparse, csc_matrix, csr_matrix
import torch
from scipy.linalg import hadamard
import pandas as pd

from srht import srht_sketch
from baselines import srht_slow

from time import time


def get_dimensions_range():

    range_n = np.unique([2**jj for jj in range(2, 16)])
    range_d = np.unique([2**jj for jj in range(2, 14)])

    ndm_vals = [[n_, d_, m_] for n_ in range_n for d_ in range_d for m_ in range_d if d_ < n_ and n_ >= 128 and m_ < n_ and d_ >= 16 and m_ >= 4 and m_ <= 4*d_]

    return ndm_vals, range_n, range_d


def record_sketch_times(n_av=10):

    srht_vals, range_n, range_d = get_dimensions_range()

    srht_res = []
    srht_slow_res = []

    max_n = max(range_n)
    max_d = max(range_d)

    print(f"generating random matrix with {max_n=} and {max_d=}")
    a = np.random.randn(max_n, max_d) / np.sqrt(max_d)

    print(f"looping - {len(srht_vals)=}")
    for kk, (n, d, m) in enumerate(srht_vals):
        if kk % 10 == 0:
            print(f"{kk=}")

        a_ = a[:n, :d]

        av_t_ = 0
        for _ in range(n_av):
            start = time()
            _ = srht_sketch(a_, m)
            t_ = time() - start
            av_t_ += 1./n_av * t_
        srht_res.append([n, d, m, av_t_])

        av_t_ = 0
        for _ in range(n_av):
            start = time()
            _ = srht_slow(a_, m)
            t_ = time() - start
            av_t_ += 1./n_av * t_
        srht_slow_res.append([n, d, m, av_t_])

    srht_res = pd.DataFrame(srht_res, columns=['n', 'd', 'm', 'time'])
    srht_slow_res = pd.DataFrame(srht_slow_res, columns=['n', 'd', 'm', 'time'])

    srht_res['method'] = 'srht'
    srht_slow_res['method'] = 'srht_slow'

    df = pd.concat([srht_res, srht_slow_res], axis=0, ignore_index=True)

    df.to_parquet('/Users/jonathanlacotte/code/SRHT/sketch_times.parquet')


if __name__ == "__main__":

    record_sketch_times(n_av=3)






