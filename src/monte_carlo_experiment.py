import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import DataGenerator, build_dist_nx, calculate_chromatic_number, calculate_clique_number, calculate_size_maximal_independent_set, calculate_size_dom_set


def monte_carlo_experiment_for_several_parameters(params, n_samples=1000, return_average=True):
    gen = DataGenerator()
    metrics_cromatic_number = []
    metrics_clique_number = []
    metrics_size_maximal_independent_set = []
    metrics_size_dom_set = []
    for _ in range(n_samples):
        data = []
        type_distribution = 0
        if params['distribution'] == 'h0':
            data = gen.generate_h0(params['n'], params['lambda'])
        else:
            data = gen.generate_h1(params['n'], params['lambda'])
            type_distribution = 1

        G = build_dist_nx(data, params['x'])
        metrics_cromatic_number.append(calculate_chromatic_number(G))
        metrics_clique_number.append(calculate_clique_number(G))
        metrics_size_maximal_independent_set.append(calculate_size_maximal_independent_set(G))
        metrics_size_dom_set.append(calculate_size_dom_set(G))
        
    if return_average:
        return {
            "cromatic_number": np.array(metrics_cromatic_number).mean(),
            "clique_number": np.array(metrics_clique_number).mean(),
            "size_maximal_independent_set": np.array(metrics_size_maximal_independent_set).mean(),
            "size_dom_set": np.array(metrics_size_dom_set).mean(),
            "type": type_distribution,
        }
    return {
        "metrics_cromatic_number": metrics_cromatic_number,
        "metrics_clique_number": metrics_clique_number,
        "metrics_size_maximal_independent_set": metrics_size_maximal_independent_set,
        "metrics_size_dom_set": metrics_size_dom_set,
    }