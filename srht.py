import numpy as np
import scipy
from scipy.sparse import issparse, csc_matrix, csr_matrix
from scipy.linalg import hadamard
import torch

from time import time


def srht_sketch(a, sketch_size, signs=None, indices_h=None):

    sketcher = SRHTSketcher()
    sketcher.sketch(a, sketch_size, signs=signs, indices_h=indices_h)
    return sketcher.sa


class SRHTSketcher:

    def __init__(self):

        self.rng = np.random.default_rng()

    @staticmethod
    def apply_random_signs(a, signs):

        if issparse(a):
            a_srht = a.copy()
        elif isinstance(a, np.ndarray):
            signs = signs.reshape((-1, 1))
            a_srht = signs * a
        elif isinstance(a, torch.Tensor):
            signs = torch.tensor(signs).view((-1, 1))
            a_srht = signs * a

        return a_srht

    def sketch(self, a, sketch_size, signs=None, indices_h=None):

        assert a.shape[0] >= a.shape[1], "left sketch! sample size must be greater than feature dimension - why would you want to embed your matrix?"
        assert a.shape[0] >= sketch_size, "left sketch! sketch size must be smaller than sample size - why would you want to embed your matrix?"

        with_torch = isinstance(a, torch.Tensor)

        if not with_torch:
            self.sa = np.zeros((sketch_size, a.shape[1]), dtype=a.dtype)
        else:
            self.sa = torch.zeros((sketch_size, a.shape[1]), dtype=a.dtype)

        n = a.shape[0]
        n_padded = 2 ** (np.int64(np.ceil(np.log(n) / np.log(2))))

        if signs is None:
            signs = self.rng.choice([-1, 1], n, replace=True)
        if indices_h is None:
            indices_h = self.rng.choice(n, sketch_size, replace=False)

        a_srht = self.apply_random_signs(a, signs)

        if issparse(a_srht):
            if n_padded > n:
                n_under = n_padded // 2
                h = hadamard(n_under, dtype=np.float32)
                h = np.vstack([np.hstack([h, h[:, :n - n_under]]), np.hstack([h, -h[:, :n - n_under]])])
            else:
                h = hadamard(n, dtype=np.float32)
            self.sa = (h[indices_h, :n] * signs.reshape((1, -1))) @ a_srht / np.sqrt(sketch_size)
        else:
            indices = np.arange(sketch_size, dtype=np.int64)

            self._sketch(indices_h, indices, a_srht, n_padded)

            self.sa /= np.sqrt(sketch_size)

    def _sketch(self, indices_hashed, indices, v, n_padded):
        """
        Helper function for srht.
        """
        n = v.shape[0]
        mid_ = n_padded // 2

        if n == 1:
            if issparse(v):
                self.sa[indices[0], v.indices] = v.data
            else:
                self.sa[indices[0]] = v
            return

        left_ = indices_hashed < mid_
        ih1 = indices_hashed[left_]
        ih2 = indices_hashed[~left_]

        if len(ih1) == 0:
            i2 = indices[~left_]
            vr_ = 1. * v[:mid_]
            vr_[:n-mid_] -= v[mid_:]
            self._sketch(ih2 - mid_, i2, vr_, mid_)
        elif len(ih2) == 0:
            i1 = indices[left_]
            vl_ = 1. * v[:mid_]
            vl_[:n - mid_] += v[mid_:]
            self._sketch(ih1, i1, vl_, mid_)
        else:
            i1 = indices[left_]
            i2 = indices[~left_]
            vl_ = 1. * v[:mid_]
            vl_[:n-mid_] += v[mid_:]
            vr_ = 1. * v[:mid_]
            vr_[:n-mid_] -= v[mid_:]
            self._sketch(ih1, i1, vl_, mid_)
            self._sketch(ih2 - mid_, i2, vr_, mid_)
