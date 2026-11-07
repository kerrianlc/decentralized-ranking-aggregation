from functools import partial

import numpy as np
from numba import njit
from utils.kendall_tau import kendall_tau

@njit
def _barycenter_consensus(rankings):
    """
    Compute the Borda consensus and true scores from a set of rankings.
    """
    _, m = rankings.shape
    x = rankings.astype(np.float64)
    true_score = np.empty(m, dtype=np.float64)
    for k in range(m):
        true_score[k] = np.mean(x[:, k])
    true_consensus = 1 + np.argsort(np.argsort(true_score))
    return (x, true_score, true_consensus)

@njit
def _score_error(x, true_score):
    n, m = x.shape
    err = 0.0
    for a in range(n):
        row_err = 0.0
        for k in range(m):
            diff = x[a, k] - true_score[k]
            row_err += diff * diff
        err += row_err / m
    return err / n

@njit
def _consensus_error(x, true_consensus, mask):
    n, m = x.shape
    mean_dist = 0.0
    for a in range(n):
        ranking = 1 + np.argsort(np.argsort(x[a]))
        if mask is not None and (not mask[a]) or mask is None:
            mean_dist += kendall_tau(ranking, true_consensus)
    return mean_dist / n

@njit
def _average_consensus_error(x, true_consensus, mask=None):
    """
    """
    n, m = x.shape
    averaged_scores = np.empty(m, dtype=np.float64)
    for k in range(m):
        total = 0.0
        count = 0
        for agent in range(n):
            if mask is None or (mask is not None and not mask[agent]):
                total += x[agent, k]
                count += 1
        if count > 0:
            averaged_scores[k] = total / count
        else:
            averaged_scores[k] = 0.0
    ranking = 1 + np.argsort(np.argsort(averaged_scores))
    return kendall_tau(ranking, true_consensus)

@njit
def _trial(
    rankings,
    edges,
    consensus_fn,
    rand_idx,
    iterations,
    consensus_every,
    true_consensus=None,
    stubborn=False,
    mask=None,
):
    n, m = rankings.shape
    score_errors = np.empty(iterations, dtype=np.float64)
    consensus_errors = np.empty(iterations, dtype=np.float64)
    last_consensus = 0.0
    x, true_score, consensus = _barycenter_consensus(rankings)
    if true_consensus is None:
        true_consensus = consensus
    for t in range(iterations):
        idx = rand_idx[t]
        i = edges[idx, 0]
        j = edges[idx, 1]
        i_corrupted = mask is not None and mask[i]
        j_corrupted = mask is not None and mask[j]
        for k in range(m):
            avg = 0.5 * (x[i, k] + x[j, k])
            if not i_corrupted or not stubborn:
                x[i, k] = avg
            if not j_corrupted or not stubborn:
                x[j, k] = avg
        score_errors[t] = _score_error(x, true_score)
        if t % consensus_every == 0:
            last_consensus = consensus_fn(
                x, true_consensus, mask=mask
            )
        consensus_errors[t] = last_consensus
    return (score_errors, consensus_errors)

def borda_trial(
    data,
    edges,
    iterations=100,
    seed=42,
    consensus_every=10,
    true_consensus=None,
    consensus_fn=None,
    stubborn=False,
    mask=None,
    **kwargs
):
    rankings = np.asarray(data, dtype=np.int64)
    edges_arr = np.asarray(edges, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(0, len(edges_arr), size=iterations, dtype=np.int64)
    return _trial(
        rankings,
        edges_arr,
        consensus_fn,
        rand_idx,
        iterations,
        consensus_every,
        true_consensus,
        stubborn,
        mask,
        **kwargs
    )

def _average_borda_trial(*args, **kwargs):
    return borda_trial(
        *args,
        consensus_fn=_average_consensus_error,
        **kwargs,
    )


def _default_borda_trial(*args, **kwargs):
    return borda_trial(
        *args,
        consensus_fn=_consensus_error,
        **kwargs,
    )


def _stubborn_borda_trial(*args, **kwargs):
    return borda_trial(
        *args,
        consensus_fn=_average_consensus_error,
        stubborn=True,
        **kwargs,
    )


BORDA_FACTORY = {
    "average": _average_borda_trial,
    "default": _default_borda_trial,
    "stubborn": _stubborn_borda_trial,
}