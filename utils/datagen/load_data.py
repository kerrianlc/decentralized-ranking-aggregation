from preflibtools.instances import OrdinalInstance
import numpy as np
import hashlib
import random

from utils.helper import rankings_to_list_dicts


def load_dataset(url, n=None, seed=42):
    instance = OrdinalInstance()
    instance.parse_url(url)
    m = instance.num_alternatives
    flat = instance.flatten_strict()
    rankings = []
    for ranking, count in flat:
        rankings.extend([tuple(ranking)] * count)

    if n is None:
        n = len(rankings)

    rng = np.random.default_rng(seed)
    permuted = [rankings[i] for i in rng.permutation(len(rankings))]
    rankings = permuted[:n]

    return rankings, m, n


def load_partial_dataset(url, n=None, seed=42):
    instance = OrdinalInstance()
    instance.parse_url(url)

    m = instance.num_alternatives
    n_voters = instance.num_voters

    flat = instance.flatten_strict()
    rankings = []

    for ranking, count in flat:
        kv = len(ranking)  # number of ranked alternatives
        avg_rank = (kv + 1 + m) / 2  # paper Appendix E

        rank = [avg_rank] * m
        for pos, item in enumerate(ranking, start=1):
            rank[item - 1] = pos  # overwrite default ranked items

        rankings.extend([tuple(rank)] * count)

    rng = np.random.default_rng(seed)
    permuted = [rankings[i] for i in rng.permutation(len(rankings))]
    rankings = permuted[: n_voters if n is None else n]
    n = len(rankings)

    return rankings, m, n


def load_sushi(n=5000, seed=42):
    """
    Load the sushi dataset from PrefLib. The dataset contains 5000 voters and 10 alternatives.
    """
    url = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00014%20-%20sushi/00014-00000001.soc"
    rankings, m, n = load_dataset(url, n=n, seed=seed)
    return rankings_to_list_dicts(rankings), m, n


def load_sushi_100(n=5000, seed=42):
    """
    Load the sushi dataset from PrefLib. The dataset contains 5000 voters and 100 alternatives but only top 10 ranked.
    """
    url = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00014%20-%20sushi/00014-00000002.soi"
    rankings, m, n = load_partial_dataset(url, n=n, seed=seed)
    return rankings_to_list_dicts(rankings), m, n


def load_tshirts(n=None, seed=42):
    """
    Load the t-shirt dataset from PrefLib. The dataset contains 30 voters and 10 alternatives.
    """
    url = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00012%20-%20shirt/00012-00000001.soc"
    rankings, m, n = load_dataset(url, n=n, seed=seed)
    return rankings_to_list_dicts(rankings), m, n


def load_netflix(n=None, seed=42):
    """
    Load the Netflix prize dataset from PrefLib.
    """
    url = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00004%20-%20netflix/00004-00000001.soc"
    rankings, m, n = load_dataset(url, n=n, seed=seed)
    return rankings_to_list_dicts(rankings), m, n


def load_debian(n=None, seed=42):
    """
    Load the Debian dataset from PrefLib.
    """
    url = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00002%20-%20debian/00002-00000003.soi"
    rankings, m, n = load_partial_dataset(url, n=n, seed=seed)
    return rankings_to_list_dicts(rankings), m, n
