# from utils.datagen.load_data import *
from utils.helper import list_dicts_to_array
from utils.graph import generate_graph
# from utils.datagen.generate_mallows import generate_mallows_mixture
from utils.results import save_results
from utils.gossip.borda_async import BORDA_FACTORY
from utils.gossip.copeland_async import COPELAND_FACTORY

import os

from numba import set_num_threads
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

GRAPH_TYPES = ("Complete", "Watts-Strogatz", "2D Grid", "Geometric")
TRIAL_FNS = {
    "Borda": BORDA_FACTORY["default"],
    "Copeland": COPELAND_FACTORY["default"],
}

def _resolve_n_jobs(n_jobs):
    if n_jobs is not None:
        return max(1, int(n_jobs))
    env_jobs = os.getenv("SLURM_CPUS_PER_TASK")
    if env_jobs is None:
        return 1
    try:
        return max(1, int(env_jobs))
    except ValueError:
        return 1

def _configure_worker_numba_threads():
    env_threads = os.getenv("NUMBA_NUM_THREADS")
    if env_threads is None:
        set_num_threads(1)
        return
    try:
        threads = max(1, int(env_threads))
    except ValueError:
        set_num_threads(1)
        return
    set_num_threads(threads)




def _trial_worker(
        rankings,
        graph_type,
        trial,
        iterations,
        consensus_every,
        sync,
        methods
    ):
    G = generate_graph(rankings.shape[0], type=graph_type, seed=trial)
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    return {
        m: TRIAL_FNS[m](
            rankings,
            edges,
            iterations=iterations,
            seed=trial,
            consensus_every=consensus_every,
            synchronous=sync,
            parallel=True,
        )
        for m in methods
    }


def run_trials(
    data,
    n_trials=10,
    iterations=1000,
    consensus_every=5,
    n_jobs=None,
    sync=False,
    methods=("Borda", "Copeland"),
    graph_types=GRAPH_TYPES,
):
    rankings = list_dicts_to_array(data)
    n_jobs = _resolve_n_jobs(n_jobs)
    results = {m: {} for m in methods}
    _configure_worker_numba_threads()

    for g in graph_types:
        print(f"Graph type {g}")
        args = [
            (rankings, g, t, iterations, consensus_every, sync, methods)
            for t in range(n_trials)
        ]

        if n_jobs == 1:
            outs = [_trial_worker(a) for a in args]
        else:
            with ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=mp.get_context("spawn")
            ) as ex:
                outs = list(ex.map(_trial_worker, args))

        for m in methods:
            score, cons = zip(*(o[m] for o in outs))
            results[m][g] = [
                (np.mean(score, 0), np.std(score, 0)),
                (np.mean(cons, 0), np.std(cons, 0)),
            ]
    return results


# if __name__ == "__main__":
#     m_items = 10

#     n_trials = 100
#     iterations = 50000
#     phi_s = 0.65

#     components = [
#         {"ref_ranking": np.arange(1, m_items + 1), "phi": phi_s, "weight": 0.7},
#         {"ref_ranking": np.arange(m_items, 0, -1), "phi": 0.6, "weight": 0.3},
#     ]
#     data, m, n = generate_mallows_mixture(
#         n=20, m=m_items, components=components, seed=0
#     )
#     results = run_trials(data, n_trials=n_trials, iterations=iterations)
#     save_results(
#         results,
#         f"results_n{n}_m{m}_phi{phi_s}_trials{n_trials}_iter{iterations}.pkl",
#     )
