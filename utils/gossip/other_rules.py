from abc import ABC, abstractmethod
import numpy as np
from utils.kendall_tau import kendall_tau
from utils.goal_pd import AsylADMM
from scipy.stats import rankdata
from utils.helper import list_dicts_to_array, consensus_to_array

class ConsensusMethod(ABC):
    def __init__(self, data, config):
        self.x = None
        self.n = len(data)
        self.m = len(data[0])
        self.ranking = [np.array(list(range(1, self.m + 1))) for _ in range(self.n)]
        self.config = config
        self.name = self.__class__.__name__
        self.score_errors = []
        self.consensus_errors = []
        self.true_score = None
        self.true_consensus = None
        self.reference = config.get("reference", None)
        self.bias = None
        self.flag = False

    @abstractmethod
    def init_true_values(self): 
        pass

    def init_x(self, data):
        self.x = np.array(data, dtype=float)

    def measure_convergence(self):
        """Measure how much agents' rankings differ."""
        mean_distance = 0
        for i in range(self.n):
            ranking = self.ranking[i]
            dist = kendall_tau(ranking, self.true_consensus)
            mean_distance += dist
        return mean_distance / self.n

    def get_local_ranking(self, i):
        """Get current ranking from agent's local scores."""
        if not self.flag:
            ranking = rankdata(self.x[i], method="ordinal")
        else:
            scores = np.ceil(self.x[i] * 2 - 0.5) / 2
            ranking = rankdata(scores, method="ordinal")
        return ranking

    def update(self, i: int, j: int):

        if not self.flag:
            avg = (self.x[i] + self.x[j]) / 2
            self.x[i], self.x[j] = avg, avg
        else:
            for method in self.methods:
                method.update(i, j)
            self.x[i] = np.array([method.x[i] for method in self.methods])
            self.x[j] = np.array([method.x[j] for method in self.methods])
        self.ranking[i] = self.get_local_ranking(i)
        self.ranking[j] = self.get_local_ranking(j)
        self._compute_errors()

    def _compute_errors(self):
        error = np.mean((self.x - self.true_score) ** 2, axis=1).mean()
        self.score_errors.append(error)
        consensus_error = self.measure_convergence()
        self.consensus_errors.append(consensus_error)

    def run_trial(self, edges, iterations: int = 100, seed: int = 42):
        np.random.seed(seed)
        for _ in range(iterations):
            idx = np.random.randint(len(edges))
            i, j = edges[idx]
            self.update(i, j)
        return self.score_errors, self.consensus_errors

class DecentralizedFootrule(ConsensusMethod):
    def __init__(self, data, config, degrees):
        super().__init__(data, config)
        # data is list_dicts from load_data (format: [{item: rank}, ...])
        self.list_dicts = data
        # Convert to array format for internal use
        array_data = list_dicts_to_array(data)
        self.init_x(array_data)
        self.init_true_values()
        self.flag = True
        self.methods = [
            AsylADMM([array_data[j][i] for j in range(self.n)], self.config, degrees)
            for i in range(self.m)
        ]
        if self.config.get("verbose", False):
            print("is correct?", np.all(self.true_consensus == range(1, self.m + 1)))
            print("Estimated footrule consensus", self.true_consensus)

    def init_true_values(self):
        consensus, f_scores = footrule_consensus(self.list_dicts)
        self.true_consensus = consensus_to_array(consensus, self.m)
        self.true_score = np.array([f_scores[item] for item in range(1, self.m + 1)])
        self.bias = kendall_tau(
            np.asarray(self.true_consensus), np.asarray(self.reference)
        )


def footrule_scores(list_dicts):
    items = sorted(list_dicts[0].keys())
    f_scores = {item: np.median([d[item] for d in list_dicts]) for item in items}
    return f_scores


def footrule_consensus(list_dicts):
    f_scores = footrule_scores(list_dicts)
    consensus = sorted(f_scores.keys(), key=lambda x: f_scores[x], reverse=False)
    return consensus, f_scores


def local_kemenization(
    initial_ranking: list, pairwise_scores: dict[tuple, float], n: int
) -> list:
    """
    Apply local Kemenization to make an initial ranking locally Kemeny optimal.
    Algorithm: For every pair of adjacent elements (u, v): if p_{u,v} < 0.5, then swap (u, v).
    """
    current_ranking = list(initial_ranking)

    iteration = 0
    while True:
        swapped = False
        iteration += 1
        for i in range(len(current_ranking) - 1):
            u = current_ranking[i]
            v = current_ranking[i + 1]

            # If p_{u,v} < 0.5, then majority prefers v over u, so swap
            p_uv = pairwise_scores[u][v] / n
            if p_uv < 0.5:
                current_ranking[i], current_ranking[i + 1] = v, u
                swapped = True
        swapped = False
        if not swapped:
            break
    return current_ranking
