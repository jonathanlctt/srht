import numpy as np
import scipy
from scipy.sparse import issparse, csc_matrix, csr_matrix
import torch
from scipy.linalg import hadamard

from srht.srht import srht_sketch
from srht.evaluate.baselines import srht_slow

from time import time

def generate_np_example(n=2048, d=128, dtype=np.float32):
    a = np.array(np.random.randn(n, d), dtype=dtype)
    return a / np.sqrt(d)


def generate_torch_example(n=2048, d=128, dtype=torch.float32):
    a = torch.randn(n, d, dtype=dtype)
    return a / np.sqrt(d)


def generate_sparse_example(n=2048, d=128, dtype=np.float32):

    data = np.ones((n,), dtype=dtype)
    indices = np.random.choice(d, n, replace=True)
    indptr = range(n+1)

    a = csr_matrix((data, indices, indptr), shape=(n, d))

    return a

def test_1(n=2048, d=128, sketch_size=512):
    '''
    Comparison with naive implementation
    '''

    a = generate_np_example(n=n, d=d, dtype=np.float32)

    signs = np.random.choice([-1, 1], n, replace=True).reshape((-1,1))
    indices_h = np.random.choice(n, sketch_size, replace=False)

    start = time()
    sa = srht_sketch(a, sketch_size, signs=signs, indices_h=indices_h)
    print(f"Time with optimized implementation: {time()-start}")

    start = time()
    n_padded = 2 ** (np.int64(np.ceil(np.log(n) / np.log(2))))
    h = hadamard(n_padded, dtype=np.float32)
    if n_padded > n:
        a_padded = np.vstack([signs * a, np.zeros((n_padded-n, d), dtype=np.float32)])
        sa_naive = (h @ a_padded)[indices_h] / np.sqrt(sketch_size)
    else:
        sa_naive = (h[indices_h] * signs.reshape((1,-1))) @ a / np.sqrt(sketch_size)
    print(f"Time with naive implementation: {time()-start}")

    assert np.allclose(sa, sa_naive), "sa and sa_naive differ"

    print("success")


def test_2():
    print("---------- Checks that power 2 padding is not necessary")

    test_1(n=2049, d=128, sketch_size=512)


def test_3(n=2048, d=128, sketch_size=512):

    print("---------- Comparison with O(n d log(n)) implementation")

    a = generate_np_example(n=n, d=d, dtype=np.float32)
    signs = np.random.choice([-1, 1], n, replace=True).reshape((-1,1))
    indices_h = np.random.choice(n, sketch_size, replace=False)

    start = time()
    sa = srht_sketch(a, sketch_size, signs=signs, indices_h=indices_h)
    print(f"Time with optimized implementation: {time()-start}")

    start = time()
    sa_slow = srht_slow(a, sketch_size, ind=indices_h, signs=signs)
    print(f"Time with O(n d log n) implementation: {time()-start}")

    assert np.allclose(sa, sa_slow), "sa and sa_slow differ"

    print("success")


def test_4(n=2048, d=128, sketch_size=512):
    print("---------- Compares results for torch, numpy and sparse arrays")

    signs = np.random.choice([-1, 1], n, replace=True).reshape((-1,1))
    indices_h = np.random.choice(n, sketch_size, replace=False)

    a = generate_sparse_example(n=n, d=d, dtype=np.float32)

    start = time()
    sa_sparse = srht_sketch(a, sketch_size, signs=signs, indices_h=indices_h)
    print(f"time for sparse float32: {time() - start}")

    a_np = a.toarray()
    start = time()
    sa_numpy = srht_sketch(a_np, sketch_size, signs=signs, indices_h=indices_h)
    print(f"time for np float32: {time() - start}")

    a_torch = torch.tensor(a.toarray())
    start = time()
    sa_torch = srht_sketch(a_torch, sketch_size, signs=signs, indices_h=indices_h)
    print(f"time for torch float32: {time() - start}")

    assert np.allclose(sa_sparse, sa_numpy), "sa_sparse and sa_numpy differ"
    assert np.allclose(sa_sparse, sa_torch.numpy()), "sa_sparse and sa_torch differ"

    a = generate_sparse_example(n=n, d=d, dtype=np.float64)

    start = time()
    sa_sparse = srht_sketch(a, sketch_size, signs=signs, indices_h=indices_h)
    print(f"time for sparse float64: {time() - start}")

    a_np = a.toarray()
    start = time()
    sa_numpy = srht_sketch(a_np, sketch_size, signs=signs, indices_h=indices_h)
    print(f"time for np float64: {time() - start}")

    a_torch = torch.tensor(a.toarray())
    start = time()
    sa_torch = srht_sketch(a_torch, sketch_size, signs=signs, indices_h=indices_h)
    print(f"time for torch float64: {time() - start}")

    assert np.allclose(sa_sparse, sa_numpy), "sa_sparse and sa_numpy differ"
    assert np.allclose(sa_sparse, sa_torch.numpy()), "sa_sparse and sa_torch differ"

    print("success")


if __name__ == "__main__":
    n = 4096  # sample size
    d = 128  # feature dimension
    m = 512  # sketch size

    test_1(n, d, m)
    test_2()
    test_3(n, d, m)
    test_4(n, d, m)
    test_4(n+1, d, m)