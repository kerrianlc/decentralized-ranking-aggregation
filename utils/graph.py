import numpy as np
import networkx as nx


def generate_graph(n, type="Watts-Strogatz", seed=42, **kwargs):
    """
    generate a graph with n nodes and the specified type. If the generated graph is not connected, it will generate a new graph until it finds a connected one.

    :param n: number of nodes
    :param type: type of graph to generate. Options are "Complete", "Watts-Strogatz", "2D Grid", "Cycle", and "Geometric". Default is "Watts-Strogatz".
    :param seed: random seed
    """

    if type == "Complete":
        G = nx.complete_graph(n)
    elif type == "Watts-Strogatz":
        if "k" not in kwargs:
            kwargs["k"] = 10
        if "p" not in kwargs:
            kwargs["p"] = 0.4
        G = nx.watts_strogatz_graph(n, seed=seed, **kwargs)
    elif type == "2D Grid":
        length, width = best_side_from_surface(n)
        G = nx.grid_2d_graph(length, width)
        G = nx.convert_node_labels_to_integers(G)
    elif type == "Cycle":
        G = nx.cycle_graph(n)
    elif type == "Geometric":
        G = generate_connected_rgg(n, seed=seed)
    elif type == "Erdos-Renyi":
        if "p" not in kwargs:
            kwargs["p"] = 0.2
        G = nx.erdos_renyi_graph(n, seed=seed, **kwargs)
    else:
        raise ValueError("Wrong graph type.")

    # check if graph is connected
    if not nx.is_connected(G):
        print("Graph is not connected. Generating a new graph.")
        return generate_graph(n, type, seed + 1)
    else:
        return G


def best_side_from_surface(S):
    root = int(S**0.5)
    for i in range(root, 0, -1):
        if S % i == 0:
            j = S // i
            return (i, j)


def generate_connected_rgg(n=100, c=8, max_attempts=100, seed=42):
    radius = np.sqrt((np.log(n) + c) / (np.pi * n))
    print("radius", np.round(radius, 2))
    for attempt in range(max_attempts):
        G = nx.random_geometric_graph(n=n, radius=radius, seed=seed + attempt)
        if nx.is_connected(G):
            print(f"Connected graph found after {attempt + 1} attempt(s)")
            return G
    raise ValueError("Failed to generate a connected graph. Try increasing the radius.")
