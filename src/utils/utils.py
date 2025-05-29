import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def build_knn_nx(data, k):
    """Строит граф на основе k ближайших соседей для каждой точки.
    
    Параметры:
        data : array-like
            Одномерный массив числовых данных
        k : int
            Количество ближайших соседей для соединения
            
    Возвращает:
        nx.Graph
            Граф, где каждая точка соединена со своими k ближайшими соседями
    """
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
    """Строит граф, соединяя точки, находящиеся на расстоянии <= d.
    
    Параметры:
        data : array-like
            Одномерный массив числовых данных
        d : float
            Пороговое расстояние для соединения точек
            
    Возвращает:
        nx.Graph
            Граф, где точки соединены, если расстояние между ними ≤ d
    """
    G = nx.Graph()
    n = len(data)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if abs(data[i] - data[j]) <= d:
                G.add_edge(i, j)
                
    return G


def calculate_connected_components(G):
    """Вычисляет количество связных компонент графа.
    
    Параметры:
        G : nx.Graph
            Входной граф
            
    Возвращает:
        int
            Количество связных компонент графа
    """
    return nx.number_connected_components(G)


def calculate_chromatic_number(G):
    """Вычисляет хроматическое число графа (минимальное число цветов для раскраски).
    
    Использует жадный алгоритм раскраски с стратегией 'largest_first'.
    
    Параметры:
        G : nx.Graph
            Входной граф
            
    Возвращает:
        int
            Хроматическое число графа
    """
    if len(G.nodes) == 0:  # Проверка на пустой граф
        return 0
    coloring = nx.greedy_color(G, strategy='largest_first')
    return max(coloring.values()) + 1


def calculate_clique_number(G):
    """Вычисляет число клики (размер максимальной клики в графе).
    
    Параметры:
        G : nx.Graph
            Входной граф
            
    Возвращает:
        int
            Размер максимальной клики
    """
    if len(G.nodes) == 0:  # Проверка на пустой граф
        return 0
    return max(len(c) for c in nx.find_cliques(G))


def calculate_size_maximal_independent_set(G):
    """Вычисляет размер максимального независимого множества.
    
    Независимое множество - множество вершин, никакие две из которых не смежны.
    
    Параметры:
        G : nx.Graph
            Входной граф
            
    Возвращает:
        int
            Размер максимального независимого множества
    """
    if len(G.nodes) == 0:  # Проверка на пустой граф
        return 0
    return len(nx.maximal_independent_set(G))

def calculate_size_dom_set(G):
    """Вычисляет размер доминирующего множества.
    
    Доминирующее множество - множество вершин, где каждая вершина вне множества
    смежна хотя бы с одной вершиной из множества.
    
    Параметры:
        G : nx.Graph
            Входной граф
            
    Возвращает:
        int
            Размер минимального доминирующего множества
    """
    if len(G.nodes) == 0:  # Проверка на пустой граф
        return 0
    return len(nx.dominating_set(G))




class DataGenerator:
    """Генератор случайных данных для тестирования гипотез H0 и H1.
    
    Класс предоставляет методы для генерации данных:
    - Для нулевой гипотезы H0: данные генерируются из экспоненциального распределения
    - Для альтернативной гипотезы H1: данные генерируются из распределения Вейбулла
    
    Параметры инициализации:
        lambda0_exp : float, default=1.0
            Параметр масштаба (λ) для экспоненциального распределения (H0)
        lambda0_weibull : float, default=1/sqrt(10)
            Параметр масштаба для распределения Вейбулла (H1)
        shape_weibull : float, default=0.5
            Параметр формы (k) для распределения Вейбулла (H1)
    """
    def __init__(self, lambda0_exp=1.0, lambda0_weibull=1/np.sqrt(10), shape_weibull=0.5):
        self.lambda0_exp = lambda0_exp
        self.lambda0_weibull = lambda0_weibull
        self.shape_weibull = shape_weibull
        
    def generate_h0(self, n, lambda_param):
        """Генерирует данные для нулевой гипотезы H0 (экспоненциальное распределение).
        
        Параметры:
            n : int
                Количество генерируемых точек данных
            lambda_param : float
                Параметр масштаба экспоненциального распределения
                
        Возвращает:
            np.ndarray
                Массив из n случайных величин, распределенных экспоненциально
        """
        return np.random.exponential(scale=1/lambda_param, size=n)
    
    def generate_h1(self, n, lambda_param):
        """Генерирует данные для альтернативной гипотезы H1 (распределение Вейбулла).
        
        Параметры:
            n : int
                Количество генерируемых точек данных
            lambda_param : float
                Параметр масштаба распределения Вейбулла
                
        Возвращает:
            np.ndarray
                Массив из n случайных величин, распределенных по Вейбуллу
        """
        return np.random.weibull(a=self.shape_weibull, size=n) * lambda_param
    
    def generate_h0_lambda0(self, n):
        """Генерирует данные для H0 с параметром lambda0, заданным при инициализации.
        
        Параметры:
            n : int
                Количество генерируемых точек данных
                
        Возвращает:
            np.ndarray
                Массив из n экспоненциально распределенных величин
        """
        return np.random.exponential(scale=1/self.lambda0_exp, size=n)
    
    def generate_h1_lambda0(self, n):
        """Генерирует данные для H1 с параметрами lambda0 и shape, заданными при инициализации.
        
        Параметры:
            n : int
                Количество генерируемых точек данных
                
        Возвращает:
            np.ndarray
                Массив из n величин, распределенных по Вейбуллу
        """
        return np.random.weibull(a=self.shape_weibull, size=n) * self.lambda0_weibull


def monte_carlo_experiment(params, n_samples=1000, return_average=True):
    """Проводит Монте-Карло эксперимент для оценки статистических свойств графов.
    
    Генерирует множественные выборки данных и для каждой строит граф, вычисляя заданную
    метрику (число компонент связности или хроматическое число).

    Параметры:
        params : dict
            Словарь с параметрами эксперимента:
            - distribution : str ('h0' или 'h1')
                Тип распределения для генерации данных:
                'h0' - экспоненциальное, 'h1' - Вейбулла
            - n : int
                Размер генерируемой выборки данных
            - lambda : float
                Параметр масштаба распределения
            - graph_type : str ('knn' или 'dist')
                Тип построения графа:
                'knn' - по k ближайшим соседям, 'dist' - по порогу расстояния
            - x : float/int
                Параметр для построения графа:
                для 'knn' - число соседей, для 'dist' - пороговое расстояние
        n_samples : int, optional (default=1000)
            Количество повторений эксперимента (размер Монте-Карло выборки)
        return_average : bool, optional (default=True)
            Если True, возвращает среднее значение метрик,
            иначе - список всех значений метрик

    Возвращает:
        float или list
            Среднее значение метрик (если return_average=True)
            или список всех значений метрик (если return_average=False)
    """
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



def monte_carlo_experiment_for_several_characteristics(params, n_samples=1000, return_average=True):
    """Проводит Монте-Карло эксперимент для оценки нескольких характеристик графа.
    
    Для каждой сгенерированной выборки данных строит граф по заданному расстоянию и вычисляет:
    - хроматическое число
    - число клики
    - размер максимального независимого множества
    - размер доминирующего множества

    Параметры:
        params : dict
            Словарь параметров эксперимента:
            - distribution : str ('h0' или 'h1')
                Тип распределения данных:
                'h0' - экспоненциальное, 'h1' - распределение Вейбулла
            - n : int
                Количество точек в выборке
            - lambda : float
                Параметр масштаба распределения
            - x : float
                Пороговое расстояние для построения графа
        n_samples : int, optional (default=1000)
            Количество повторений эксперимента
        return_average : bool, optional (default=True)
            Если True, возвращает средние значения характеристик,
            иначе - все значения для каждого эксперимента

    Возвращает:
        dict
            Если return_average=True:
            {
                "cromatic_number": среднее хроматическое число,
                "clique_number": среднее число клики,
                "size_maximal_independent_set": средний размер макс. независимого множества,
                "size_dom_set": средний размер доминирующего множества
            }
            Если return_average=False:
            {
                "metrics_cromatic_number": список хроматических чисел,
                "metrics_clique_number": список чисел клики,
                "metrics_size_maximal_independent_set": список размеров независимых множеств,
                "metrics_size_dom_set": список размеров доминирующих множеств
            }
    """
    gen = DataGenerator()
    metrics_cromatic_number = []
    metrics_clique_number = []
    metrics_size_maximal_independent_set = []
    metrics_size_dom_set = []
    for _ in range(n_samples):
        data = []
        if params['distribution'] == 'h0':
            data = gen.generate_h0(params['n'], params['lambda'])
        else:
            data = gen.generate_h1(params['n'], params['lambda'])

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
        }
    return {
        "metrics_cromatic_number": metrics_cromatic_number,
        "metrics_clique_number": metrics_clique_number,
        "metrics_size_maximal_independent_set": metrics_size_maximal_independent_set,
        "metrics_size_dom_set": metrics_size_dom_set,
    }