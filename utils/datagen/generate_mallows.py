import numpy as np
from numba import njit

from utils.helper import rankings_to_list_dicts

@njit
def _rim_single(ref_order, phi, m, u):
    """
    One draw from Mallows(ref_order, phi) via Repeated Insertion Model.
    u: pre-drawn uniform samples of shape (m,)
    Returns: sequence array where sequence[r] = item at rank r+1
    """
    sequence = np.empty(m, dtype=np.int64)
    seq_len = 0
    for k in range(m):
        item = ref_order[k]
        total = 0.0
        for j in range(k + 1):
            total += phi**j
        threshold = u[k] * total
        cumsum = 0.0
        j = 0
        for jj in range(k + 1):
            cumsum += phi**jj
            if cumsum >= threshold:
                j = jj
                break

        insert_pos = k - j
        for pos in range(seq_len, insert_pos, -1):
            sequence[pos] = sequence[pos - 1]
        sequence[insert_pos] = item
        seq_len += 1

    return sequence


@njit
def _rim_batch(ref_order, phi, m, n, uniforms):
    """Generate a batch of rankings from the Mallows model."""
    out = np.empty((n, m), dtype=np.int64)
    for i in range(n):
        seq = _rim_single(ref_order, phi, m, uniforms[i])
        for rank in range(m):
            out[i, seq[rank]] = rank + 1
    return out


def generate_mallows(n, m, ref_ranking, phi=0.7, seed=0):
    """
    Sample n rankings from Mallows(ref_ranking, phi) using RIM.
    """
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n, m))
    ref_order = np.argsort(ref_ranking).astype(np.int64)
    return _rim_batch(ref_order, float(phi), m, n, uniforms)

def generate_mallows_mixture(n, m, components, seed=0):
    """
    Sample n rankings from a mixture of Mallows models.
    """
    rng = np.random.default_rng(seed)

    weights = np.array([c["weight"] for c in components], dtype=float)
    weights /= weights.sum()

    counts = rng.multinomial(n, weights)

    all_rankings = []
    for comp, count in zip(components, counts):
        if count == 0:
            continue
        ref = np.asarray(comp["ref_ranking"])
        samples = generate_mallows(
            n=count,
            m=m,
            ref_ranking=ref,
            phi=comp["phi"],
            seed=int(rng.integers(0, 2**31)),
        )
        # samples[i, item] = rank (1-indexed, 0-indexed items)
        # convert to preflib tuple: items in rank order, 1-indexed
        for i in range(count):
            order = np.argsort(samples[i])
            all_rankings.append(tuple(order + 1))

    perm = rng.permutation(len(all_rankings))
    all_rankings = [all_rankings[i] for i in perm]

    return rankings_to_list_dicts(all_rankings), m, n
