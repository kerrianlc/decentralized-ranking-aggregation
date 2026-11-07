import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

from utils.kendall_tau import kendall_tau


@njit(parallel=True)
def _gossip(
    x0,
    true_score,
    true_consensus,
    items,
    indptr,
    indices,
    weights,
    self_weight,
    alpha,
    rounds,
    every,
    m,
    copeland,
    delta,
    patience,
):
    n, d = x0.shape
    x = x0.copy()
    x_prev = x0.copy()
    x_new = np.empty_like(x0)
    score_errors = np.empty(rounds)
    consensus_errors = np.empty(rounds)
    consensus = _consensus_error(x, m, copeland, true_consensus, items)
    below_since = -1

    for r in range(rounds):
        for i in prange(n):
            for k in range(d):
                x_new[i, k] = self_weight[i] * x[i, k]
            for p in range(indptr[i], indptr[i + 1]):
                w = weights[p]
                nb = indices[p]
                for k in range(d):
                    x_new[i, k] += w * x[nb, k]
            for k in range(d):
                x_new[i, k] = (1.0 - alpha) * x_prev[i, k] + alpha * x_new[i, k]
        x, x_prev, x_new = x_new, x, x_prev

        score_errors[r] = _score_error(x, true_score)
        if (r + 1) % every == 0 or r == rounds - 1:
            consensus = _consensus_error(x, m, copeland, true_consensus, items)
        consensus_errors[r] = consensus

        if patience > 0:
            if consensus <= delta:
                if below_since < 0:
                    below_since = r
                if r + 1 - below_since >= patience:
                    return score_errors[: r + 1], consensus_errors[: r + 1]
            else:
                below_since = -1

    return score_errors, consensus_errors


def _spectral_gap(w):

    second = eigsh(w, k=2, which="LA", return_eigenvectors=False).min()
    return max(1.0 - second, 1e-12)


def _optimal_alpha(gap):
    return (2.0 - 2.0 * np.sqrt(gap * (1.0 - gap / 4.0))) / (1.0 - gap / 2.0) ** 2


def sync_params(n, edges):
    edges = np.asarray(edges, dtype=np.int64)
    adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    adj.data[:] = 1.0

    degree = np.diff(adj.indptr)
    rows = np.repeat(np.arange(n), degree)
    weights = 1.0 / (1.0 + np.maximum(degree[rows], degree[adj.indices]))
    self_weight = 1.0 - np.bincount(rows, weights=weights, minlength=n)

    off_diagonal = csr_matrix((weights, adj.indices, adj.indptr), shape=(n, n))
    gap = _spectral_gap(off_diagonal + diags(self_weight, 0))
    return {
        "indptr": adj.indptr.astype(np.int64),
        "indices": adj.indices.astype(np.int64),
        "weights": weights,
        "self_weight": self_weight,
        "gap": gap,
        "alpha": _optimal_alpha(gap),
    }


def run_sync(
    data,
    edges,
    rule,
    rounds=1000,
    consensus_every=1,
    topk=-1,
    params=None,
    delta=0.0,
    patience=0,
):
    ranks = np.asarray(data, dtype=np.int64)
    n, m = ranks.shape
    if rule == "borda":
        x0 = ranks.astype(np.float64)
    elif rule == "copeland":
        x0 = (ranks[:, :, None] < ranks[:, None, :]).reshape(n, m * m).astype(np.float64)
    else:
        raise ValueError(f"unknown rule: {rule}")
    copeland = rule == "copeland"

    true_score = x0.mean(axis=0)
    true_consensus = _ranking(true_score, m, copeland)
    items = np.arange(m) if topk < 0 else np.argsort(true_consensus)[:topk]
    if params is None:
        params = sync_params(n, edges)

    return _gossip(
        x0,
        true_score,
        true_consensus,
        items,
        params["indptr"],
        params["indices"],
        params["weights"],
        params["self_weight"],
        params["alpha"],
        rounds,
        max(consensus_every, 1),
        m,
        copeland,
        float(delta),
        patience,
    )
