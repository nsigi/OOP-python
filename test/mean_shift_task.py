import time

from clustering_utils import gaussian_cluster, draw_clusters, distance, gauss_core
from typing import Union, List, Tuple
import numpy as np


class MShift:
    def __init__(self):
        """
        Метод среднего сдвига.
        Этапы алгоритма:
        1. Копируем переданные данные для кластеризации "data" в "data_shifted"
        2. Для каждой точки из набора данных "data" с индексом "point_index" считаем среднее значение вокруг нее.
        3. Полученное значение на среднего сравниваем со значением под индексом "point_index" из "data_shifted"
           Если расстояние между точками меньше, чем некоторый порог "distance_threshold", то говорим, что смещать
           точку далее нет смысла и помечаем ее, как неподвижную.
           Для новой неподвижной точки пытаемся найти ближайший к ней центр кластера и добавить индекс этой точки
           к индексам точек кластера. Кластер считается ближайшим, если расстояние между точкой и центроидом
           меньше, чем "window_size".
           Если такового нет, то точка считается новым центром кластера.
        4. Пункты 2-3 повторяются до тех пор, пока все точки не будут помечены, как неподвижные.
        """
        """
        Количество кластеров, которые ожидаем обнаружить.
        """
        self._n_clusters: int = -1
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
        self._distance_threshold: float = 1e-3
        """
        Ширина ядра функции усреднения.
        """
        self._window_size: float = 0.15

    @property
    def window_size(self) -> float:
        """
        Просто геттер для ширины ядра функции усреднения ("_window_size").
        """
        return self._window_size

    @window_size.setter
    def window_size(self, value: float) -> None:
        """
        Сеттер для ширины ядра функции усреднения ("_window_size").
        1. Должен осуществлять проверку типа.
        2. Проверку на не отрицательность.
        """
        assert isinstance(value, float)
        self._window_size = max(0.0, value)

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
        assert isinstance(value, float)
        self._distance_threshold = max(0.0, value)

    @property
    def n_clusters(self) -> int:
        """
        Геттер для числа кластеров, которые обнаружили.
        """
        return 0 if self._clusters_centers is None else len(self._clusters_centers)

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
        Реализует "ленивую" инициализацию полей "_clusters" и "_clusters_centers".
        """
        if self._clusters_points_indices is None:
            self._clusters_points_indices = []
            self._clusters_centers = []
        self._clusters_points_indices.clear()
        self._clusters_centers.clear()

    def _shift_cluster_point(self, point: np.ndarray) -> np.ndarray:
        """
        Функция, которая считает средне-взвешенное (если, например, используется Гауссово ядро) внутри круглого окна
        с радиусом "window_size" вокруг точки point.
        Возвращает массив равный по размеру "point".
        """
        dists = np.linalg.norm(self._data - point, axis=1)
        weights = gauss_core(dists, self._window_size)
        return np.dot(weights, self._data) / weights.sum()

    def _update_clusters_centers(self, sample_index, sample: np.ndarray):
        """
        Функция ищет ближайший центр кластера для точки "sample".
        Если не находит, то считает, что "sample" - новый центр кластера.
        Если находит, то добавляет к индексам точек кластера "sample_index"
        """
        if self.n_clusters == 0:
            self._clusters_centers.append(sample)
            self._clusters_points_indices.append([sample_index])
            return
        min_index, min_dist = self._get_closest_cluster_center(sample)
        if self.window_size > min_dist:
            self._clusters_points_indices[min_index].append(sample_index)
        else:
            self._clusters_centers.append(sample)
            self._clusters_points_indices.append([sample_index])

    def _shift_cluster_points(self) -> None:
        """
        Выполняет итеративный сдвиг всех точек к их среднему значению.
        Т.е. для каждой точки вызывается функция _shift_cluster_point()
        Выполняется до тех пор, пока все точки не будут помечены, как неподвижные.
        """
        shift_points = np.array(self._data)
        fixed_count = 0
        fixed_points = set()
        while fixed_count != self.n_samples:
            for ind, point in enumerate(shift_points):
                if ind in fixed_points:
                    continue
                shift_point = self._shift_cluster_point(point)
                dist = distance(shift_point, point)
                shift_points[ind] = shift_point
                if dist < self.distance_threshold:
                    fixed_points.add(ind)
                    fixed_count += 1
                    self._update_clusters_centers(ind, point)

    def _get_closest_cluster_center(self, sample: np.ndarray) -> Tuple[int, float]:
        """
        Определяет ближайший центр кластера для точки из переданного набора данных и расстояние до него.
        Hint: для ускорения кода используйте min с генератором.
        """
        min_index = -1
        min_dist = np.inf
        for cluster_center_index, cluster_center in enumerate(self._clusters_centers):
            dist = distance(cluster_center, sample)
            if dist > min_dist:
                continue
            min_index = cluster_center_index
            min_dist = dist
        return min_index, min_dist

    def fit(self, data: np.ndarray) -> None:
        """
        Выполняет кластеризацию данных в "data".
        1. Необходима проверка, что "data" - экземпляр класса "np.ndarray".
        2. Необходима проверка, что "data" - двумерный массив.
        Этапы работы метода:
        # 1. Проверки передаваемых аргументов
        # 2. Присваивание аргументов внутренним полям класса.
        # 3. Сдвиг точек в направлении средних значений вокруг них ("_shift_cluster_points").
        """
        self._data = data
        self._clear_current_clusters()
        self._shift_cluster_points()

    def show(self):
        draw_clusters(self.clusters, cluster_centers=self._clusters_centers, title="Mean shift clustering")


def separated_clusters():
    """
    Пример с пятью разрозненными распределениями точек на плоскости.
    """
    n = 1024
    m_means = MShift()
    clusters_data = np.vstack((gaussian_cluster(cx=0.5, n_points=n),
                               gaussian_cluster(cx=1.0, n_points=n),
                               gaussian_cluster(cx=1.5, n_points=n),
                               gaussian_cluster(cx=2.0, n_points=n),
                               gaussian_cluster(cx=2.5, n_points=n)))
    m_means.fit(clusters_data)
    m_means.show()


def merged_clusters():
    """
    Пример с кластеризацией пятна.
    """
    m_means = MShift()
    m_means.fit(gaussian_cluster(n_points=100))
    m_means.show()


if __name__ == "__main__":
    """
    Сюрприз-сюрприз! Вызов функций "merged_clusters" и "separated_clusters".
    """
    t = time.perf_counter()
    merged_clusters()
    print(f'TIME0 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
    t = time.perf_counter()
    separated_clusters()
    print(f'TIME1 {time.perf_counter() - t}')
