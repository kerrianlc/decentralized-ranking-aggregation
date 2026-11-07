import pickle

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from utils.datagen.generate_mallows import generate_mallows_mixture
from utils.gossip.borda_async import BORDA_FACTORY
from utils.gossip.copeland_async import COPELAND_FACTORY
from utils.gossip.sync_gossip import run_sync, sync_params
from utils.graph import generate_graph
from utils.helper import list_dicts_to_array


OUTPUT = "results.pkl"

N = 250
M = 10
DELTA = 0.05
ASYNC_ITERATIONS = 100000
ASYNC_PATIENCE = 2000
ASYNC_EVERY = 100
SYNC_ROUNDS = 20_000
SYNC_PATIENCE = 20

TRIALS_BY_GRAPH = {
    "Complete": 6,
    "2D Grid": 6,
    "Cycle": 3,
    "Watts-Strogatz": 15,
    "Geometric": 15,
}


RULES = {
    "Borda": ("borda", BORDA_FACTORY["default"]),
    "Copeland": ("copeland", COPELAND_FACTORY["default"]),
}


def mixing_time(errors, patience, delta=DELTA):
    above = np.asarray(errors) > delta
    n = len(above)
    if n < patience:
        return n, True
    counts = np.concatenate(([0], np.cumsum(above)))
    clean = np.flatnonzero(counts[patience:] - counts[: n - patience + 1] == 0)
    if clean.size == 0:
        return n, True
    return int(clean[0]) + 1, False


def is_connected(n, edges):
    adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(adj, directed=False)[0] == 1


def run_case(data, graph_type, seed):
    graph = generate_graph(N, type=graph_type, seed=seed)
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    num_edges = len(edges)
    connected = bool(is_connected(N, edges))
    params = sync_params(N, edges)
    results = {}

    for name, (rule, async_trial) in RULES.items():
        _, errors = async_trial(
            data,
            edges,
            iterations=ASYNC_ITERATIONS,
            seed=seed,
            consensus_every=ASYNC_EVERY,
        )
        stride = max(1, round(ASYNC_ITERATIONS / len(errors)))
        steps, censored = mixing_time(errors, max(1, ASYNC_PATIENCE // stride))
        steps *= stride
        results[f"{name} async"] = {
            "edges": num_edges,
            "mixing": steps,
            "communications": steps,
            "censored": censored,
            "connected": connected,
        }

        _, errors = run_sync(
            data,
            edges,
            rule,
            rounds=SYNC_ROUNDS,
            params=params,
            delta=DELTA,
            patience=SYNC_PATIENCE,
        )
        rounds, censored = mixing_time(errors, SYNC_PATIENCE)
        results[f"{name} sync"] = {
            "edges": num_edges,
            "mixing": rounds,
            "communications": rounds * num_edges,
            "censored": censored,
            "connected": connected,
        }

    return results


def main():
    rng = np.random.default_rng(42)
    reference = rng.permutation(np.arange(1, M + 1))
    components = [
        {"weight": 0.7, "ref_ranking": reference, "phi": 0.4},
        {"weight": 0.3, "ref_ranking": M + 1 - reference, "phi": 0.6},
    ]
    data, _, _ = generate_mallows_mixture(n=N, m=M, components=components, seed=123)
    data = list_dicts_to_array(data)

    results = {}
    for graph_type, trials in TRIALS_BY_GRAPH.items():
        cases = [run_case(data, graph_type, seed=1000 + t) for t in range(trials)]
        results[graph_type] = {
            label: [case[label] for case in cases] for label in cases[0]
        }
        with open(f"mix_rate.pkl", "wb") as f:
            pickle.dump(results, f)

    return results


if __name__ == "__main__":
    main()