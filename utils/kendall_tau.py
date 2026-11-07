import numpy as np
from numba import njit


@njit
def _bit_query(tree, i):
    s = 0
    while i > 0:
        s += tree[i]
        i -= i & -i
    return s


@njit
def _bit_update(tree, i, m):
    while i <= m:
        tree[i] += 1
        i += i & -i

@njit
def kendall_tau(rank_a, rank_b):
    m = rank_a.shape[0]
    order = np.argsort(rank_a)
    b_compressed = 1 + np.argsort(np.argsort(rank_b))
    tree = np.zeros(m + 1, dtype=np.int64)
    inv = 0
    for p in range(m):
        v = b_compressed[order[p]]
        inv += p - _bit_query(tree, v)
        _bit_update(tree, v, m)
    return inv