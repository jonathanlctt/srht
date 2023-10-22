## SRHT
Subsampled Randomized Hadamard Transform

- `O(n d log m)` implementation of the SRHT. 

- This implementation does not require zero-padding.
- Compatible with `numpy.ndarray`, `torch.Tensor` and `scicy.sparse` arrays.

Given a matrix `A` with size `n x d` and an embedding dimension (sketch size) `m`, transform the matrix A into `S x A` where 
`S = R x H x E`.

- `R`: row-subsampling matrix, i.e., subsamples `m` rows uniformly at random without replacement.

- `H`: Hadamard transform of size `N`, where `N` is the smallest power of `2` greater than n.

- `E`: random signs diagonal matrix


## Install

Download repo and
```bash
$ python setup.py install
```

## Usage

```python
# numpy 
from srht.srht import srht_sketch
import numpy as np

n = 20000  # sample size
d = 512  # feature dimension
sketch_size = 1024

a = np.random.randn(n, d) / np.sqrt(sketch_size)  

sa = srht_sketch(a, sketch_size)

print(f"{sa.shape=}")   # (1024, 512)

# torch - same usage
import torch
a = torch.tensor(a)

sa = srht_sketch(a, sketch_size)  

# scipy sparse arrays
from scipy.sparse import csr_matrix

data = np.ones((n,), dtype=np.float32)
indices = np.random.choice(d, n, replace=True)
indptr = range(n+1)

a = csr_matrix((data, indices, indptr), shape=(n, d))

sa = srht_sketch(a, sketch_size)
```
