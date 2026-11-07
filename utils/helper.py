from preflibtools.instances import OrdinalInstance
import numpy as np


def instance_to_rankings(instance):
    """
    Convert a preflib instance to a list of rankings.
    """
    flat = instance.flatten_strict()
    rankings = []
    for ranking, count in flat:
        rankings.extend([tuple(map(int, ranking))] * count)
    return rankings


def rankings_to_instance(rankings):
    """
    Convert a list of rankings to a preflib instance.
    """
    instance = OrdinalInstance()
    instance.num_alternatives = len(rankings[0])
    for ranking in rankings:
        instance.append_order(ranking)
    return instance


def rankings_to_list_dicts(rankings):
    list_dicts = []
    for ranking in rankings:
        m = len(ranking)
        values = list(ranking)
        is_permutation = set(values) == set(range(1, m + 1))

        if is_permutation:
            d = {item: rank + 1 for rank, item in enumerate(values)}
        else:
            # Already in rank-vector form: index is item_id-1, value is rank.
            d = {item_id: int(values[item_id - 1]) for item_id in range(1, m + 1)}

        list_dicts.append(d)
    return list_dicts


def list_dicts_to_array(list_dicts):
    """
    Convert list of dicts {item: rank} to numpy array format.
    Output array[agent_id, item_id-1] = rank
    """
    n = len(list_dicts)
    m = len(list_dicts[0])
    array = np.zeros((n, m), dtype=float)
    for agent_id, d in enumerate(list_dicts):
        for item, rank in d.items():
            array[agent_id, int(item) - 1] = rank
    return array


def consensus_to_array(consensus, m):
    array = np.zeros(m, dtype=int)
    for rank, item in enumerate(consensus, start=1):
        array[item - 1] = rank
    return array
