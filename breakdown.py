import os, pickle
import numpy as np

from utils.datagen.attacks import create_corrupted_dataset
from utils.datagen.generate_mallows import generate_mallows
from utils.graph import generate_graph
from utils.gossip.borda_async import BORDA_FACTORY
from utils.gossip.copeland_async import COPELAND_FACTORY

STUBBORN = True

if STUBBORN:
    trial_fns = {
        "Borda": BORDA_FACTORY["stubborn"],
        "Copeland": COPELAND_FACTORY["stubborn"],
    }
else:
    trial_fns = {"Borda": BORDA_FACTORY["average"], "Copeland": COPELAND_FACTORY["average"]}

def run_and_plot(cfg):
    return run_breakdown(**cfg)

def run_breakdown(
    n=1000,
    m=7,
    checkpoints=[500, 1000, 5000, 10000],
    epsilons=np.linspace(0.01, 0.45, 20),
    graph_type="Watts-Strogatz",
    phi=0.7,
    n_trials=500,
    consensus_every=250,
    seed=0,
):
    rng = np.random.default_rng(seed)
    G = generate_graph(n, type=graph_type, seed=seed)
    edges = np.array(list(G.edges()), dtype=np.int64)
    max_iter = max(checkpoints)
    ckpt_indices = {c: c - 1 for c in checkpoints}
    true_consensus = np.arange(1, m + 1)

    trial_seeds = [int(rng.integers(0, 2**31)) for _ in range(n_trials)]
    bad_trajs = {"Borda": {}, "Copeland": {}}
    for eps_idx, eps in enumerate(epsilons):
        bad_trajs["Borda"][eps_idx] = []
        bad_trajs["Copeland"][eps_idx] = []
        for s in trial_seeds:
            honest = np.asarray(
                generate_mallows(n, m, true_consensus, phi=phi, seed=s), dtype=np.int64
            )
            data, mask = create_corrupted_dataset(
                honest, true_consensus, eps, seed=s
            )
            for name in trial_fns:
                bad_trajs[name][eps_idx].append(
                    trial_fns[name](
                        data,
                        edges,
                        true_consensus=true_consensus,
                        mask=mask,
                        iterations=max_iter,
                        seed=s,
                        consensus_every=consensus_every,
                    )
                )

    all_results = {}
    for iterations in checkpoints:
        ci = ckpt_indices[iterations]
        excess_errors = {
            name: [
                [bad_trajs[name][eps_idx][t][1][ci] for t in range(n_trials)]
                for eps_idx in range(len(epsilons))
            ]
            for name in trial_fns
        }
        max_excess = max(
            e for name in excess_errors for errs in excess_errors[name] for e in errs
        )
        delta_range = np.linspace(0, max_excess, 200)

        result = {
            "params": {
                "n": n,
                "m": m,
                "iterations": iterations,
                "n_trials": n_trials,
                "seed": seed,
                "epsilons": np.asarray(epsilons),
            },
            "honest": honest,
            "delta_range": delta_range,
            "excess_errors": excess_errors,
        }
        all_results[iterations] = result

        os.makedirs("results", exist_ok=True)
        pickle_path = f"results/breakdown_n{n}_m{m}_phi{phi}_ntrials{n_trials}_type{graph_type}_iter{iterations}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f)

    return all_results


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    from itertools import product

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[3000])
    parser.add_argument("--m", type=int, nargs="+", default=[7])
    parser.add_argument("--phi", type=float, nargs="+", default=[0.6])
    parser.add_argument("--graph-type", type=str, nargs="+", default=["Watts-Strogatz"])
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[500, 1000, 1500, 2000, 3000, 3500, 4000, 4500, 5000],
        help="Default checkpoints used when no specific mapping is given for n",
    )
    parser.add_argument(
        "--checkpoints-map",
        type=str,
        nargs="+",
        default=[],
        metavar="N:C1,C2,...",
        help="Per-n checkpoint overrides, e.g. 1000:500,1000 3000:500,1000,2000,3000",
    )
    parser.add_argument("--epsilons-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ckpt_map = {}
    for entry in args.checkpoints_map:
        n_str, ckpts_str = entry.split(":")
        ckpt_map[int(n_str)] = [int(c) for c in ckpts_str.split(",")]

    configs = [
        dict(
            n=n,
            m=m,
            phi=phi,
            graph_type=gt,
            n_trials=args.n_trials,
            checkpoints=ckpt_map.get(n, args.checkpoints),
            epsilons=np.linspace(0.01, 0.45, args.epsilons_n),
            seed=args.seed,
        )
        for n, m, phi, gt in product(args.n, args.m, args.phi, args.graph_type)
    ]


    with Pool(processes=args.workers) as pool:
        pool.map(run_and_plot, configs)
