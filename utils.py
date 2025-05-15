import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

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
