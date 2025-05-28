import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def build_knn_nx(data, k):
    G = nx.Graph()
    n = len(data)
    G.add_nodes_from(range(n))
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data.reshape(-1, 1))
    _, indices = nbrs.kneighbors(data.reshape(-1, 1))
    for i in range(n):
        for j in indices[i][1:]: 
            if j not in G[i]:
                G.add_edge(i, j)
    return G


def build_dist_nx(data, d):
    G = nx.Graph()
    n = len(data)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if abs(data[i] - data[j]) <= d:
                G.add_edge(i, j)
                
    return G


def calculate_connected_components(G):
    return nx.number_connected_components(G)


def calculate_chromatic_number(G):
    coloring = nx.greedy_color(G, strategy='largest_first')
    return max(coloring.values()) + 1


def calculate_clique_number(G):
    return max(len(c) for c in nx.find_cliques(G))


def calculate_size_maximal_independent_set(G):
    return len(nx.maximal_independent_set(G))

def calculate_size_dom_set(G):
    return len(nx.dominating_set(G))




class DataGenerator:
    def __init__(self, lambda0_exp=1.0, lambda0_weibull=1/np.sqrt(10), shape_weibull=0.5):
        self.lambda0_exp = lambda0_exp
        self.lambda0_weibull = lambda0_weibull
        self.shape_weibull = shape_weibull
        
    def generate_h0(self, n, lambda_param):
        return np.random.exponential(scale=1/lambda_param, size=n)
    
    def generate_h1(self, n, lambda_param):
        return np.random.weibull(a=self.shape_weibull, size=n) * lambda_param
    
    def generate_h0_lambda0(self, n):
        return np.random.exponential(scale=1/self.lambda0_exp, size=n)
    
    def generate_h1_lambda0(self, n):
        return np.random.weibull(a=self.shape_weibull, size=n) * self.lambda0_weibull


def monte_carlo_experiment(params, n_samples=1000, return_average=True):
    gen = DataGenerator()
    metrics = []
    for _ in range(n_samples):
        data = []

        if params['distribution'] == 'h0':
            data = gen.generate_h0(params['n'], params['lambda'])
        else:
            data = gen.generate_h1(params['n'], params['lambda'])

        if params['graph_type'] == 'knn':
            G = build_knn_nx(data, params['x'])
            mertic = calculate_connected_components(G)
            metrics.append(mertic)
        else:
            G = build_dist_nx(data, params['x'])
            mertic = calculate_chromatic_number(G)
            metrics.append(mertic)
    if return_average:
        return np.array(metrics).mean()
    return metrics

