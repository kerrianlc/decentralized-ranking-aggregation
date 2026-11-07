from convergence_plot import run_trials
from utils.results import save_results

import numpy as np


def generate_linear_toy(N=100, n=6, dim=4, n_train=20, varsigma=(0.1, 0.4), seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(dim)
    theta /= np.linalg.norm(theta)
    q = rng.permutation(np.linspace(0.1, 0.9, n))
    D = rng.standard_normal((n, dim))
    D += np.outer(q - D @ theta, theta)
    scores = np.empty((N, n))
    for u, vs in enumerate(rng.uniform(*varsigma, N)):
        X = rng.standard_normal((n_train, dim))
        y = X @ theta + vs * rng.standard_normal(n_train)
        scores[u] = D @ np.linalg.solve(X.T @ X, X.T @ y)
    ranks = (-scores).argsort(1).argsort(1) + 1
    data = [dict(enumerate(r)) for r in ranks]
    return data, scores, n, N


if __name__ == "__main__":
    data, scores, m, N = generate_linear_toy()
    r = run_trials(
            data,
            scores=scores,
            n_trials=300,
            iterations=1000,
            n_jobs=8,
            methods=("Borda", "Copeland", "Average"),
    )
    save_results(r, "linear_model.pkl")
