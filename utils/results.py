import pickle
import json
import numpy as np


def save_results(results, filepath):
    """Save results dict preserving structure. Use .pkl for pickle or .json for JSON."""
    if filepath.endswith(".pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
    elif filepath.endswith(".json"):

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (dict, list)):
                return obj
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        json_results = {}
        for method, graph_dict in results.items():
            json_results[method] = {}
            for graph_type, data_list in graph_dict.items():
                json_results[method][graph_type] = [
                    (convert_to_serializable(arr1), convert_to_serializable(arr2))
                    for arr1, arr2 in data_list
                ]
        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)
    else:
        raise ValueError("Use .pkl or .json extension")


def load_results(filepath):
    """Load results dict from pickle or JSON."""
    if filepath.endswith(".pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
        results = {}
        for method, graph_dict in data.items():
            results[method] = {}
            for graph_type, data_list in graph_dict.items():
                results[method][graph_type] = [
                    (np.array(arr1), np.array(arr2)) for arr1, arr2 in data_list
                ]
        return results
    else:
        raise ValueError("Use .pkl or .json extension")
