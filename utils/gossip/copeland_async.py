import numpy as np
from numba import njit
from utils.kendall_tau import kendall_tau

@njit
def _barycenter_consensus(rankings):
    """
    Compute the Copeland consensus and true scores from a set of rankings.
    """
    n, m = rankings.shape
    x = np.zeros((n, m, m), dtype=np.float64)
    for agent_id in range(n):
        ranking = rankings[agent_id]
        for i in range(m):
            for j in range(m):
                if ranking[i] < ranking[j]:
                    x[agent_id, i, j] = 1.0
                elif ranking[i] == ranking[j]:
                    x[agent_id, i, j] = 0.5
    true_score = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            total = 0.0
            for a in range(n):
                total += x[a, i, j]
            true_score[i, j] = total / n
    si = np.zeros(m, dtype=np.float64)
    for i in range(m):
        count = 0.0
        for j in range(m):
            if true_score[i, j] > 0.5:
                count += 1.0
        si[i] = count
    true_consensus = 1 + np.argsort(np.argsort(-si))
    return (x, true_score, true_consensus)


@njit
def _score_error(x, true_score):
    n, m = x.shape[0], x.shape[1]
    err = 0.0
    for a in range(n):
        row_err = 0.0
        for i in range(m):
            for j in range(m):
                diff = x[a, i, j] - true_score[i, j]
                row_err += diff * diff
        err += row_err / (m * m)
    return err / n


@njit
def _consensus_error(x, true_consensus, mask):
    n, m = x.shape[0], x.shape[1]
    mean_dist = 0.0
    for a in range(n):
        scores = np.zeros(m, dtype=np.float64)
        for i in range(m):
            count = 0.0
            for j in range(m):
                if x[a, i, j] > 0.5:
                    count += 1.0
            scores[i] = count
        ranking = 1 + np.argsort(np.argsort(-scores))
        if mask is not None and (not mask[a]) or mask is None:
            mean_dist += kendall_tau(ranking, true_consensus)
    return mean_dist / n

@njit
def _average_consensus_error(x, true_consensus, mask=None):
    n, m = x.shape[0], x.shape[1]
    averaged_pairwise = np.empty((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            total = 0.0
            count = 0
            for agent in range(n):
                if mask is None or (mask is not None and not mask[agent]):
                    total += x[agent, i, j]
                    count += 1
            if count > 0:
                averaged_pairwise[i, j] = total / count
            else:
                averaged_pairwise[i, j] = 0.0

    scores = np.empty(m, dtype=np.float64)
    for i in range(m):
        cnt = 0.0
        for j in range(m):
            if averaged_pairwise[i, j] > 0.5:
                cnt += 1.0
        scores[i] = cnt

    ranking = 1 + np.argsort(np.argsort(-scores))
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
    n, m = rankings.shape[0], rankings.shape[1]
    score_errors = np.empty(iterations, dtype=np.float64)
    consensus_errors = np.empty(iterations, dtype=np.float64)

    x, true_score, consensus = _barycenter_consensus(rankings)
    if true_consensus is None:
        true_consensus = consensus
    last_consensus = 0.0
    for t in range(iterations):
        idx = rand_idx[t]
        i = edges[idx, 0]
        j = edges[idx, 1]
        i_corrupted = mask is not None and mask[i]
        j_corrupted = mask is not None and mask[j]
        for a in range(m):
            for b in range(m):
                avg = 0.5 * (x[i, a, b] + x[j, a, b])
                if not i_corrupted or not stubborn:
                    x[i, a, b] = avg
                if not j_corrupted or not stubborn:
                    x[j, a, b] = avg
        score_errors[t] = _score_error(x, true_score)
        if t % consensus_every == 0:
            last_consensus = consensus_fn(x, true_consensus, mask=mask)
        consensus_errors[t] = last_consensus
    return (score_errors, consensus_errors)

def copeland_trial(
    data,
    edges,
    consensus_fn,
    iterations=100,
    seed=42,
    consensus_every=10,
    true_consensus=None,
    stubborn=False,
    mask=None,
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
        mask
    )

def _average_copeland_trial(*args, **kwargs):
    return copeland_trial(
        *args,
        consensus_fn=_average_consensus_error,
        **kwargs,
    )


def _default_copeland_trial(*args, **kwargs):
    return copeland_trial(
        *args,
        consensus_fn=_consensus_error,
        **kwargs,
    )


def _stubborn_copeland_trial(*args, **kwargs):
    return copeland_trial(
        *args,
        consensus_fn=_average_consensus_error,
        stubborn=True,
        **kwargs,
    )


COPELAND_FACTORY = {
    "average": _average_copeland_trial,
    "default": _default_copeland_trial,
    "stubborn": _stubborn_copeland_trial,
}