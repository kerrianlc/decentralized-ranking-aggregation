import os
import urllib.request
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from convergence import run_trials
from utils.helper import rankings_to_list_dicts
from utils.results import save_results
warnings.filterwarnings("ignore")

SEED = 0
N_ITEMS, MIN_RATED = 25, 8
TEST_FRAC = 0.3
N_SEEDS = 3

NODES = 100
TOP = 3
N_LOC = 5
ALPHA = 0.3
TRAIN_FRAC = 0.5

DATA_DIR = "data"
os.makedirs("fig", exist_ok=True)
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def load_movielens():
    """
        Load the MovieLens 100k dataset and return a DataFrame of ratings and a feature matrix for the top N_ITEMS movies.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    zpath = os.path.join(DATA_DIR, "ml-latest-small.zip")
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
        movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))
    counts = ratings.groupby("movieId").size().sort_values(ascending=False)
    top = counts.head(N_ITEMS).index.tolist()
    idx = {m: i for i, m in enumerate(top)}
    sub = ratings[ratings.movieId.isin(top)].copy()
    sub["item"] = sub.movieId.map(idx)

    mv = movies.set_index("movieId").loc[top]
    genres = mv.genres.str.get_dummies("|").drop(
        columns="(no genres listed)", errors="ignore"
    )
    genres = genres.loc[:, genres.std() > 0]
    year = mv.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    year = year.fillna(year.mean())
    year = (year - year.mean()) / year.std()
    F = np.column_stack([genres.to_numpy(float), year.to_numpy()])
    return sub, F


def pl_grad(s, r):
    """
    Compute the gradient of the Plackett-Luce log-likelihood.
    """
    g = np.zeros_like(s)
    for k in range(len(r) - 1):
        z = s[r[k:]]
        w = np.exp(z - z.max())
        p = w / w.sum()
        g[r[k:]] += p
        g[r[k]] -= 1.0
    return g


def fit_pl(F, rankings, lam=1e-2, lr=0.5, steps=500):
    """
    Fit Plackett-Luce model to rankings using gradient descent.
    """
    w = np.zeros(F.shape[1])
    for _ in range(steps):
        s = F @ w
        g = sum(pl_grad(s, r) for r in rankings) / len(rankings)
        w -= lr * (F.T @ g + lam * w)
    return w


def split_users(sub, rng):
    eligible = sub.groupby("userId").size()[lambda s: s >= MIN_RATED].index.tolist()
    n = int(np.sqrt(min(N_USERS, len(eligible)))) ** 2
    users = rng.choice(eligible, size=n, replace=False)
    train = []
    for uid in users:
        d = sub[sub.userId == uid].sort_values(
            ["rating", "timestamp"], ascending=[False, True]
        )
        items, notes = d.item.to_numpy(), d.rating.to_numpy()
        te = np.zeros(len(d), dtype=bool)
        te[rng.choice(len(d), max(2, round(TEST_FRAC * len(d))), replace=False)] = True
        train.append(items[~te])
    return train


if __name__ == "__main__":
    sub, F = load_movielens()
    train = split_users(sub, np.random.default_rng(SEED))
    n = len(train)

    scores = np.array([F @ fit_pl(F, [r]) for r in train])
    orders = 1 + np.argsort(-scores, axis=1, kind="stable")
    ranks = rankings_to_list_dicts(orders)
    results = run_trials(ranks, n_trials=20, iterations=50000)
    save_results(results, "movie_lens.pkl")
    plt.show()
