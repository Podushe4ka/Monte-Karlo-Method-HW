import pytest
import numpy as np
import networkx as nx
import unittest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Теперь можно импортировать модули через полный путь
from src.utils.utils import (
    build_dist_nx, 
    build_knn_nx, 
    calculate_chromatic_number,
    calculate_clique_number,
    calculate_connected_components,
    calculate_size_dom_set,
    calculate_size_maximal_independent_set,
    monte_carlo_experiment,
    monte_carlo_experiment_for_several_characteristics,
    DataGenerator
)
# Тесты для функций построения графов
def test_build_knn_nx():
    """Тестирование построения графа по k ближайшим соседям."""
    data = np.array([0, 1, 3, 6, 10])
    k = 2
    G = build_knn_nx(data, k)
    
    assert isinstance(G, nx.Graph), "Должен возвращаться объект nx.Graph"
    assert len(G.nodes) == len(data), "Количество узлов должно соответствовать данным"
    

def test_build_dist_nx():
    """Тестирование построения графа по порогу расстояния."""
    data = np.array([0, 1, 2, 5, 10])
    d = 2.0
    G = build_dist_nx(data, d)
    
    assert isinstance(G, nx.Graph), "Должен возвращаться объект nx.Graph"
    assert len(G.nodes) == len(data), "Количество узлов должно соответствовать данным"
    assert G.has_edge(0, 1), "Должно быть ребро между близкими точками"
    assert not G.has_edge(0, 3), "Не должно быть ребра между далекими точками"

# Тесты для функций вычисления характеристик графа
def test_calculate_connected_components():
    """Тестирование подсчета компонент связности."""
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (3,4)])
    assert calculate_connected_components(G) == 2, "Должно быть 2 компоненты связности"

def test_calculate_chromatic_number():
    """Тестирование вычисления хроматического числа."""
    G = nx.complete_graph(4)
    assert calculate_chromatic_number(G) == 4, "Для полного графа K4 хроматическое число должно быть 4"

def test_calculate_clique_number():
    """Тестирование вычисления числа клики."""
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,0), (2,3)])
    assert calculate_clique_number(G) == 3, "Максимальная клика должна быть размера 3"

def test_calculate_size_maximal_independent_set():
    """Тестирование вычисления независимого множества."""
    G = nx.path_graph(4)
    mis_size = calculate_size_maximal_independent_set(G)
    assert mis_size == 2, "Для пути из 4 вершин MIS должен быть размера 2"

def test_calculate_size_dom_set():
    """Тестирование вычисления доминирующего множества."""
    G = nx.path_graph(4)
    dom_size = calculate_size_dom_set(G)
    assert dom_size == 2, "Для пути из 4 вершин доминирующее множество должно быть размера 2"

# Тесты для DataGenerator
def test_data_generator_h0():
    """Тестирование генерации данных для H0."""
    gen = DataGenerator()
    data = gen.generate_h0(100, 1.0)
    assert len(data) == 100, "Должно генерироваться заданное количество точек"
    assert isinstance(data, np.ndarray), "Должен возвращаться numpy массив"

def test_data_generator_h1():
    """Тестирование генерации данных для H1."""
    gen = DataGenerator()
    data = gen.generate_h1(100, 1.0)
    assert len(data) == 100, "Должно генерироваться заданное количество точек"
    assert isinstance(data, np.ndarray), "Должен возвращаться numpy массив"

# Тесты для Monte Carlo экспериментов
def test_monte_carlo_experiment():
    """Тестирование базового Monte Carlo эксперимента."""
    params = {
        'distribution': 'h0',
        'n': 50,
        'lambda': 1.0,
        'graph_type': 'knn',
        'x': 3
    }
    result = monte_carlo_experiment(params, n_samples=10)
    assert isinstance(result, (float, list)), "Должен возвращаться float или list"

def test_monte_carlo_experiment_for_several_characteristics():
    """Тестирование расширенного Monte Carlo эксперимента."""
    params = {
        'distribution': 'h1',
        'n': 50,
        'lambda': 1.0,
        'x': 0.5
    }
    result = monte_carlo_experiment_for_several_characteristics(params, n_samples=10)
    assert isinstance(result, dict), "Должен возвращаться словарь"
    assert 'cromatic_number' in result or 'metrics_cromatic_number' in result, "Должны быть результаты по хроматическому числу"

# Тесты для граничных случаев
def test_empty_graph():
    """Тестирование обработки пустого графа."""
    G = nx.Graph()
    assert calculate_connected_components(G) == 0, "Для пустого графа должно быть 0 компонент"
    assert calculate_chromatic_number(G) == 0, "Для пустого графа хроматическое число должно быть 0"

def test_single_node_graph():
    """Тестирование графа с одной вершиной."""
    G = nx.Graph()
    G.add_node(0)
    assert calculate_connected_components(G) == 1, "Должна быть одна компонента"
    assert calculate_chromatic_number(G) == 1, "Хроматическое число должно быть 1"

# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])