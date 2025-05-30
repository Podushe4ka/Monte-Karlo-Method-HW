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
  
  
def calculate_max_degree(G):
    degree_dict = dict(G.degree())
    max_deg = max(degree_dict.values())
    return max_deg

def calculate_independent_set(G):
    mis = nx.maximal_independent_set(G)
    return len(mis)

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

class DataGenerator_other:
    def __init__(self, shape_Laplace = 0, beta0_Laplace = np.sqrt(1/2), v0_Student = 3 ):
        self.v0_Student = v0_Student
        self.beta0_Laplace = beta0_Laplace
        self.shape_Laplace = shape_Laplace
        
    def generate_h0(self, n, beta_param):
        return np.random.laplace(loc=self.shape_Laplace, scale=beta_param, size=n)
    
    def generate_h0_baseline(self, n):
        return np.random.laplace(loc=self.shape_Laplace,
                                 scale=self.beta0_Laplace,
                                 size=n)
    
    def generate_h1(self, n, nu_param):
        return np.random.standard_t(df=nu_param, size=n)
    
    def generate_h1_baseline(self, n):
        return np.random.standard_t(df=self.v0_Student,
                                    size=n)

def monte_carlo_experiment_other(params, n_samples=1000):
    gen = DataGenerator_other()
    metrics = []
    for _ in range(n_samples):
        data = []

        if params['distribution'] == 'h0_baseline':
            data = gen.generate_h0_baseline(params['n'])
        elif params['distribution'] == 'h1_baseline':
            data = gen.generate_h1_baseline(params['n'])
        elif params['distribution'] == 'h0':
            data = gen.generate_h0(params['n'], params['lambda'])
        else:
            data = gen.generate_h1(params['n'], params['lambda'])

        if params['graph_type'] == 'knn':
            G = build_knn_nx(data, params['x'])
            mertic = calculate_max_degree(G)
            metrics.append(mertic)
        else:
            G = build_dist_nx(data, params['x'])
            mertic = calculate_independent_set(G)
            metrics.append(mertic)

    return np.array(metrics).mean()
