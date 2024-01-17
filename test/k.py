from clustering_utils import gaussian_cluster, draw_clusters, distance
#from sklearn.datasets import load_iris
# iris = load_iris()
from typing import Union, List
import numpy as np
import random


class KMeans:
    """
    Метод К-средних соседей.
    Этапы алгоритма:
    1. Выбирается число кластеров k.
    2. Из исходного множества данных случайным образом выбираются k наблюдений,
       которые будут служить начальными центрами кластеров.
    3. Для каждого наблюдения исходного множества определяется ближайший к нему центр кластера
       (расстояния измеряются в метрике Евклида). При этом записи,
        «притянутые» определенным центром, образуют начальные кластеры
    4. Вычисляются центроиды — центры тяжести кластеров. Каждый центроид — это вектор, элементы которого
       представляют собой средние значения соответствующих признаков, вычисленные по всем записям кластера.
    5. Центр кластера смещается в его центроид, после чего центроид становится центром нового кластера.
    6. 3-й и 4-й шаги итеративно повторяются. Очевидно, что на каждой итерации происходит изменение границ
       кластеров и смещение их центров. В результате минимизируется расстояние между элементами внутри
       кластеров и увеличиваются между-кластерные расстояния.
    Примечание: Ниже описана функциональная структура, которую использовал я. Вы можете модифицировать или вовсе
                отойти в сторону от неё. Главное, что требуется это реализация пунктов 1-6.
    """
    def __init__(self):
        """
        Метод к-средних соседей.
        """
        """
        Количество кластеров, которые ожидаем обнаружить.
        """
        self._n_clusters: int = 0
        """
        Данные для кластеризации. Поле инициализируется при вызове метода fit.
        Например, если данные представляют собой координаты на плоскости,
        каждая отельная строка - это точка этой плоскости.
        """
        self._data: Union[np.ndarray, None] = None
        """
        Центры кластеров на текущем этапе кластеризации.
        """
        self._clusters_centers: Union[List[np.ndarray], None] = None
        """
        Список индексов строк из "_data", которые соответствуют определённому кластеру.
        Список списков индексов.
        """
        self._clusters_points_indices: Union[List[List[int]], None] = None
        """
        Расстояние между центроидом кластера на текущем шаге и предыдущем при котором завершается кластеризация.
        """
        self._distance_threshold: float = 0.0001

    @property
    def distance_threshold(self) -> float:
        """
        Просто геттер для "_distance_threshold".
        """
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value: float) -> None:
        """
        Сеттер для "_distance_threshold".
        1. Должен осуществлять проверку типа.
        2. Проверку на не отрицательность.
        """
        ...
        assert isinstance(value, float)
        self._distance_threshold = value

    @property
    def n_clusters(self) -> int:
        """
        Геттер для числа кластеров, которые ожидаем обнаружить.
        """
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int) -> None:
        """
        Сеттер для числа кластеров, которые ожидаем обнаружить.
        1. Должен осуществлять проверку типа.
        2. Количество кластеров не менее двух.
        """
        assert isinstance(value, int)
        self._n_clusters = value

    @property
    def n_samples(self) -> int:
        """
        Количество записей в массиве данных. Например, количество {x, y} координат на плоскости.
        """
        return 0 if self._data is None else self._data.shape[0]

    @property
    def n_features(self) -> int:
        """
        Количество особенностей каждой записи в массив денных. Например,
        две координаты "x" и "y" в случе точек на плоскости.
        """
        return 0 if self._data is None else self._data.shape[1]

    @property
    def clusters(self) -> List[np.ndarray]:
        """
        Создаёт список из np.ndarray. Каждый такой массив - это все точки определённого кластера.
        Индексы точек соответствующих кластеру хранятся в "_clusters_points_indices"
        """
        clusters = []
        for ids in self._clusters_points_indices:
            cluster_points = np.zeros((len(ids), self.n_features), dtype=float)
            for index, cluster_point_index in enumerate(ids):
                cluster_points[index, :] = self._data[cluster_point_index, :]
            clusters.append(cluster_points)
        return clusters

    def _clear_current_clusters(self) -> None:
        """
        Очищает центры кластеров на текущем этапе кластеризации.
        Очищает список индексов строк из "_data", которые соответствуют определённому кластеру.
        Реализует "ленивую" инициализацию полей "_clusters_points_indices" и "_clusters_centers".
        """
        if self._clusters_points_indices is None:
            self._clusters_points_indices = []
            self._clusters_centers = []
        self._clusters_points_indices.clear()
        self._clusters_centers.clear()

    def _create_start_clusters_centers(self) -> None:
        """
        Случайным образом выбирает n_clusters точек из переданных данных в качестве начальных центроидов кластеров.
        Точки нужно выбрать таким образом, что бы они не повторялись
        """
        """
        Очищаем информацию о центрах кластеров и о индексах точек, соответствующих кластеру. 
        """

        self._clear_current_clusters()
        """
        Далее есть смысл использовать что-то вроде "set", что бы индексы случайно выбранных начальных
        центров кластеров не повторялись.
        """
        clusters_ids = set()  # Проверка, что мы не воткнём две одинаковые точки, как центр кластера.
        while len(self._clusters_points_indices) != self._n_clusters:
            cluster_center_index = random.randint(0, self.n_samples - 1)
            if cluster_center_index in clusters_ids:
                continue
            clusters_ids.update({cluster_center_index})
            self._clusters_centers.append(self._data[cluster_center_index, :])
            self._clusters_points_indices.append([])

    def _get_closest_cluster_center(self, sample: np.ndarray) -> int:
        """
        Определяет ближайший центр кластера для точки из переданного набора данных.
        Hint: для ускорения кода используйте min с генератором.
        """
        # min_index = -1
        # min_dist = 1e32
        m = min([(distance(cluster_center, sample), cluster_center_index) for cluster_center_index, cluster_center in enumerate(self._clusters_centers)], key=lambda x: x[0])
        return m[1]
        # for cluster_center_index, cluster_center in enumerate(self._clusters_centers):
        #     dist = distance(cluster_center, sample)
        #     []
        #     if dist > min_dist:
        #         continue
        #     min_index = cluster_center_index
        #     min_dist = dis

    def _clusterize_step(self) -> List[np.ndarray]:
        """
        Определяет списки индексов точек из "_data", наиболее близких для того или иного кластера.
        На основе этих список вычисляются новые центры кластеров.
        """
        """
        Поиск ближайшего центра кластера для конкретной точки:
        """
        """
        Расчёт центроидов:
        Hint: используйте sum с генератором
        """
        for cluster in self._clusters_points_indices:
            cluster.clear()
        for sample_index, sample in enumerate(self._data):
            cluster_index = self._get_closest_cluster_center(sample)
            self._clusters_points_indices[cluster_index].append(sample_index)
        centroids = [np.mean(self._data[sample_indices:], axis = 0)
            for sample_indices in self._clusters_points_indices if sample_indices]
        return centroids

    def fit(self, data: np.ndarray, target_clusters: int = 5) -> None:
        """
        Выполняет кластеризацию данных в "data".
        1. Необходима проверка, что "data" - экземпляр класса "np.ndarray".
        2. Необходима проверка, что "data" - двумерный массив.
        Этапы работы метода:
        1. Проверки передаваемых аргументов
        2. Присваивание аргументов внутренним полям класса.
        3. Построение начальных центроидов кластеров "_create_start_clusters_centers"
        4. Цикл уточнения положения центроидов. Выполнять пока расстояние между текущим центроидом
           кластера и предыдущим больше, чем "distance_threshold"
        """
        assert isinstance(data, np.ndarray)
        assert (data.shape[0] != 2)

        self._n_clusters = target_clusters
        self._data = data
        self._create_start_clusters_centers()

        prev_centroids = self._clusters_centers

        while True:
            curr_centroids = self._clusterize_step()
            if all(distance(left, right) < self._distance_threshold for left, right in
                   zip(prev_centroids, curr_centroids)):
                break
            prev_centroids, self._clusters_centers = self._clusters_centers, curr_centroids

    def show(self):
        """
        Выводит результат кластеризации в графическом виде
        """
        draw_clusters(self.clusters, cluster_centers=self._clusters_centers, title="K-means clustering")


def separated_clusters():
    """
    Пример с пятью разрозненными распределениями точек на плоскости.
    """
    k_means = KMeans()
    clusters_data = np.vstack((gaussian_cluster(cx=0.5, n_points=512),
                               gaussian_cluster(cx=1.0, n_points=512),
                               gaussian_cluster(cx=1.5, n_points=512),
                               gaussian_cluster(cx=2.0, n_points=512),
                               gaussian_cluster(cx=2.5, n_points=512)))
    k_means.fit(clusters_data)
    k_means.show()


def merged_clusters():
    """
    Пример с кластеризацией пятна.
    """
    k_means = KMeans()
    k_means.fit(gaussian_cluster(n_points=512*5))
    k_means.show()


if __name__ == "__main__":
    """
    Сюрприз-сюрприз! Вызов функций "merged_clusters" и "separated_clusters".
    """
    merged_clusters()
    separated_clusters()



