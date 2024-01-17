from typing import Tuple, Callable, Union, List
import matplotlib.pyplot as plt
import numpy as np
import random

_debug_mode = True
_accuracy = 1e-6
Vector2Int = Tuple[int, int]
Vector2 = Tuple[float, float]
Section = Tuple[Vector2, Vector2]
EmptyArray = np.ndarray([])


def march_squares_2d(field: Callable[[float, float], float],
                     min_bound: Vector2 = (-5.0, -5.0),
                     max_bound: Vector2 = (5.0, 5.0),
                     march_resolution: Vector2Int = (128, 128),
                     threshold: float = 0.5) -> List[Section]:
    """
    Эта функция рисует неявнозаданную функцию вида f(x,y) = 0. Есть аналог в matplotlib
    :param field: неявно заданное поле, для которого ищем контур
    :param min_bound: границы прямоугольной области, в которой выполняется алгоритм
    :param max_bound:
    :param march_resolution: количество шагов по осям при выполнении алгоритма
    :param threshold: значение, где начинается и заканчивается контур
    :return:
    """

    def lin_interp(_a: float, _b: float, t) -> float:
        return _a + (_b - _a) * t

    rows, cols = max(march_resolution[0], 3), max(march_resolution[1], 3)
    cols_ = cols - 1
    rows_ = cols - 1
    # шаг по x и y
    dx = (max_bound[0] - min_bound[0]) / cols_
    dy = (max_bound[1] - min_bound[1]) / rows_
    row: float
    col: float
    state: int
    p1: float
    p2: float
    p3: float
    p4: float
    t_val: float
    d_t: float
    shape: List[Section] = []
    # идем по всем точкам сетки
    for i in range(cols_ * rows_):
        state = 0
        # вычисление координат i-го узла сетки
        row = (i // cols_) * dy + min_bound[1]
        col = (i % cols_) * dx + min_bound[0]

        a_val = field(col, row)
        b_val = field(col + dx, row)
        c_val = field(col + dx, row + dy)
        d_val = field(col, row + dy)

        state += 8 if a_val >= threshold else 0
        state += 4 if b_val >= threshold else 0
        state += 2 if c_val >= threshold else 0
        state += 1 if d_val >= threshold else 0

        if state == 0 or state == 15:
            continue
        # без интерполяции
        # a = (col + dx * 0.5, row           )
        # b = (col + dx,       row + dy * 0.5)
        # c = (col + dx * 0.5, row + dy      )
        # d = (col,            row + dy * 0.5)

        d_t = b_val - a_val
        if np.abs(d_t) < _accuracy:
            a = (lin_interp(col, col + dx, np.sign(threshold - a_val)), row)
        else:
            t_val = (threshold - a_val) / d_t
            a = (lin_interp(col, col + dx, t_val), row)

        d_t = c_val - b_val
        if np.abs(d_t) < _accuracy:
            b = (col + dx, lin_interp(row, row + dy, np.sign(threshold - b_val)))
        else:
            t_val = (threshold - b_val) / d_t
            b = (col + dx, lin_interp(row, row + dy, t_val))

        d_t = c_val - d_val
        if np.abs(d_t) < _accuracy:
            c = (lin_interp(col, col + dx, np.sign(threshold - d_val)), row + dy)
        else:
            t_val = (threshold - d_val) / d_t
            c = (lin_interp(col, col + dx, t_val), row + dy)

        d_t = d_val - a_val
        if np.abs(d_t) < _accuracy:
            d = (col, lin_interp(row, row + dy, np.sign(threshold - a_val)))
        else:
            t_val = (threshold - a_val) / d_t
            d = (col, lin_interp(row, row + dy, t_val))

        while True:
            if state == 1:
                shape.append((c, d))
                break
            if state == 2:
                shape.append((b, c))
                break
            if state == 3:
                shape.append((b, d))
                break
            if state == 4:
                shape.append((a, b))
                break
            if state == 5:
                shape.append((a, d))
                shape.append((b, c))
                break
            if state == 6:
                shape.append((a, c))
                break
            if state == 7:
                shape.append((a, d))
                break
            if state == 8:
                shape.append((a, d))
                break
            if state == 9:
                shape.append((a, c))
                break
            if state == 10:
                shape.append((a, b))
                shape.append((c, d))
                break
            if state == 11:
                shape.append((a, b))
                break
            if state == 12:
                shape.append((b, d))
                break
            if state == 13:
                shape.append((b, c))
                break
            if state == 14:
                shape.append((c, d))
                break
            break
    return shape


def rand_in_range(rand_range: Union[float, Tuple[float, float]] = 1.0) -> float:
    if isinstance(rand_range, float):
        return random.uniform(-0.5 * rand_range, 0.5 * rand_range)
    if isinstance(rand_range, tuple):
        return random.uniform(rand_range[0], rand_range[1])
    return random.uniform(-0.5, 0.5)


def ellipsoid(x: float, y: float, params: Tuple[float, float, float, float, float]) -> float:
    """
    уравнение эллипсойда
    :param x: координата по х
    :param y: координата по y
    :param params: расстояние от точки (x,y) до эллипсойда f(x, y), f(x, y) = 0 <- точка принадлежит эллипсойду
    :return:
    """
    return x * params[0] + y * params[1] + x * y * params[2] + x * x * params[3] + y * y * params[4] - 1


def log_reg_ellipsoid_test_data(params: Tuple[float, float, float, float, float],
                                arg_range: float = 5.0, rand_range: float = 1.0,
                                n_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генератор тестовых данных для логистической регрессии. что бы понять что тут просходит, просто  _debug_mode = True
    :param params:
    :param arg_range:
    :param rand_range:
    :param n_points:
    :return:
    """
    if _debug_mode:
        print(f"logistic regression f(x,y) = {params[0]:1.3}x + {params[1]:1.3}y + {params[2]:1.3}xy +"
              f" {params[3]:1.3}x^2 + {params[4]:1.3}y^2 - 1,\n"
              f" arg_range =  [{-arg_range * 0.5:1.3}, {arg_range * 0.5:1.3}],\n"
              f" rand_range = [{-rand_range * 0.5:1.3}, {rand_range * 0.5:1.3}]")
    features = np.zeros((n_points, 5), dtype=float)
    features[:, 0] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 1] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 2] = features[:, 0] * features[:, 1]
    features[:, 3] = features[:, 0] * features[:, 0]
    features[:, 4] = features[:, 1] * features[:, 1]
    groups = np.array([np.sign(ellipsoid(features[i, 0], features[i, 1], params)) * 0.5 + 0.5 for i in range(n_points)])
    return features, groups


def log_reg_test_data(k: float = -1.5, b: float = 0.1, arg_range: float = 1.0,
                      rand_range: float = 0.0, n_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генератор тестовых данных для логистической регрессии. что бы понять что тут просходит, просто  _debug_mode = True
    :param k:
    :param b:
    :param arg_range:
    :param rand_range: параметр рандомизации
    :param n_points: кол-во точек
    :return:
    """
    if _debug_mode:
        print(f"logistic regression test data b = {b:1.3}, k = {k:1.3},\n"
              f" arg_range = [{-arg_range * 0.5:1.3}, {arg_range * 0.5:1.3}],\n"
              f" rand_range = [{-rand_range * 0.5:1.3}, {rand_range * 0.5:1.3}]")
    features = np.zeros((n_points, 2), dtype=float)
    features[:, 0] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 1] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    groups = np.array(
        [1 if features[i, 0] * k + b > features[i, 1] + rand_in_range(rand_range) else 0.0 for i in range(n_points)])
    return features, groups


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    :param x:
    :return:
    """
    return np.array(1.0 / (1.0 + np.exp(-x)))


def loss(groups_probs, groups):
    """
    Функция потерь
    :param groups_probs:
    :param groups:
    :return:
    """
    epsilon = 1e-15
    groups_probs = np.clip(groups_probs, epsilon, 1 - epsilon)
    return (-groups * np.log(groups_probs) -
            (1.0 - groups) * np.log(1.0 - groups_probs)).mean()


def draw_logistic_data(features: np.ndarray, groups: np.ndarray, theta: np.ndarray = None) -> None:
    """
    Рисует результат вычисления логистической регрессии
    :param features:
    :param groups:
    :param theta:
    :return:
    """
    [plt.plot(features[i, 0], features[i, 1], '+b') if groups[i] == 0
     else plt.plot(features[i, 0], features[i, 1], '*r') for i in range(features.shape[0] // 2)]

    if theta is None:
        plt.show()
        return

    b = theta[0] / np.abs(theta[2])
    k = theta[1] / np.abs(theta[2])

    x_0, x_1 = features[:, 0].min(), features[:, 0].max()
    y_0, y_1 = features[:, 1].min(), features[:, 1].max()

    y_1 = (y_1 - b) / k
    y_0 = (y_0 - b) / k

    if y_0 < y_1:
        x_1 = min(x_1, y_1)
        x_0 = max(x_0, y_0)
    else:
        x_1 = min(x_1, y_0)
        x_0 = max(x_0, y_1)

    x = [x_0, x_1]
    y = [b + x_0 * k, b + x_1 * k]
    plt.plot(x, y, 'k')
    plt.show()


class LogisticRegression:
    def __init__(self, learning_rate: float = 1.0,
                 max_iters: int = 1000, accuracy: float = 1e-2):
        # максимальное количество шагов градиентным спуском
        self._max_train_iters: int = 0
        # длина шага вдоль направления градиента
        self._learning_rate: float = 0
        # точность к которой мы стремимся
        self._learning_accuracy: float = 0
        # количество признаков одной группы
        self._group_features_count = 0
        # параметры тетта (подробное описание в pdf файле)
        self._thetas: Union[np.ndarray, None] = None
        # текущее знаение функции потерь
        self._losses: float = 0.0

        self.max_train_iters = max_iters
        self.learning_rate = learning_rate
        self.learning_accuracy = accuracy

    def __str__(self):
        """
        JSON совместимая строка, состоящая из:
        group_features_count - целое число;,
        max_train_iters - целое число;
        learning_rate - число с плавающей точкой;
        learning_accuracy - число с плавающей точкой;
        thetas - массив значений с плавающей точкой;
        losses - число с плавающей точкой.
        :return:
        """
        sep = ', '
        return f"\t\t{{\n" \
               f"\t\t\t\"group_features_count\":   {self._group_features_count},\n" \
               f"\t\t\t\"max_train_iters\": {self._max_train_iters},\n" \
               f"\t\t\t\"learning_rate\":   {self._learning_rate},\n" \
               f"\t\t\t\"learning_accuracy\":   {self._learning_accuracy},\n" \
               f"\t\t\t\"thetas\":   [{sep.join(str(v)for v in self._thetas)}],\n" \
               f"\t\t\t\"losses\":   {self._losses}\n" \
               f"\t\t}}"

    @property
    def group_features_count(self) -> int:
        return self._group_features_count

    @property
    def max_train_iters(self) -> int:
        return self._max_train_iters

    @max_train_iters.setter
    def max_train_iters(self, value: int) -> None:
        if 100 <= value <= 1000 and isinstance(value, int):
            self._max_train_iters = value
        else:
            raise ValueError(f"Invalid args in max_train_iters setter:\n\"value\" {value}")

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if 0.01 <= value <= 1.0 and isinstance(value, float):
            self._learning_rate = value
        else:
            raise ValueError(f"Invalid args in learning_rate setter:\n\"value\" {value}")

    @property
    def learning_accuracy(self) -> float:
        return self._learning_accuracy

    @learning_accuracy.setter
    def learning_accuracy(self, value: float) -> None:
        if 0.01 <= value <= 1.0 and isinstance(value, float):
            self._learning_rate = value
        else:
            raise ValueError(f"Invalid args in learning_accuracy setter:\n\"value\" {value}")

    @property
    def thetas(self) -> np.ndarray:
        return self._thetas

    @property
    def losses(self) -> float:
        return self._losses

    def predict(self, features: np.ndarray) -> np.ndarray:
        # проверка размерности - количество принаков группы == количество элементов в столбце
        if features.shape[1] != self.thetas.size - 1:
            raise ValueError(f"Неверное количество признаков")

        return np.array(sigmoid(np.dot(features, self._thetas[1::]) + self.thetas[0]))

    def train(self, features: np.ndarray, groups: np.ndarray) -> None:
        """
        :param features: - признаки групп, записанные в виде столбцов                    (x)
        :param groups: - вектор столбец принадлежности групп (0-первая , 1-вторая) (y)
        :return:
        """
        # проверка размерности - количество признаков группы == количество элементов в столбце
        # реализация градиентного спуска для обучения логистической регрессии.

        if features.shape[0] != groups.shape[0] or features.ndim != 2:
            raise ValueError(f"Неверное количество признаков")

        # проверка размерности -  количество принаков группы == количество элементов в толбце
        # формула thetas(i) = thetas(i - 1) - learning_rate * (X^T * sigmoid(X *  thetas(i - 1)) - groups)
        # чем больше learning_rate, тем быстрее происходит обучение, но падает точность
        # количество признаков у группы
        self._group_features_count = features.shape[1]
        # Инициализация весов thetas рандомными значениями
        self._thetas = np.array([rand_in_range(1000) for _ in range(self._group_features_count + 1)])
        # _thetas - столбец из трех признаков
        # theta[0] * 1 + theta[1] * x + theta[2] * y = 0

        # Добавили слева столбец единиц к точкам (типо b = 1)
        # Добавляем новый признак для учета свободного члена, чтобы соблидать размерность с thetas
        # x = np.hstack((np.ones((features.shape[0], 1), dtype=float), features))
        # обучаем модель,обновляя параметры модели, используя градиентный спуск

        for iteration in range(self.max_train_iters):
            thetas_cp = self.thetas.copy()
            error = sigmoid(np.dot(features, thetas_cp[1:]) + thetas_cp[0]) - groups
            self._thetas -= self._learning_rate * np.hstack((np.sum(error), np.dot(features.T, error)))
            # смотрим на разницу между последними значениями, чтобы они удовлетворяли заданной точности
            if np.linalg.norm(thetas_cp - self._thetas) <= self._learning_accuracy:
                break

        self._losses = loss(self.predict(features), groups)


def lin_reg_test():
    # features - массив точек, group - массив классов (1 - красная, 0 - синяя)
    # Точку характеризует x y на i-ой позиции в features и её класс на i-ой позиции в group
    features, group = log_reg_test_data()
    lg = LogisticRegression()
    lg.train(features, group)
    draw_logistic_data(features, group, lg.thetas)


def non_lin_reg_test():
    features, group = log_reg_ellipsoid_test_data((0.0,0.0,0.0, 1.0, 1.0))
    lg = LogisticRegression()
    lg.train(features, group)
    print(lg)

    def _ellipsoid(x: float, y: float) -> float:
        return lg.thetas[0] + x * lg.thetas[1] + y * lg.thetas[2] + x * y * lg.thetas[3] + x * x * lg.thetas[4] + y * y * lg.thetas[5]

    for arc in march_squares_2d(_ellipsoid):
        p_0, p_1 = arc
        plt.plot([p_0[0], p_1[0]], [p_0[1], p_1[1]], 'k')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.gca().axis('equal')
    print(lg.thetas / np.abs(lg.thetas[0]))
    draw_logistic_data(features, group)


if __name__ == "__main__":
    lin_reg_test()
    non_lin_reg_test()
