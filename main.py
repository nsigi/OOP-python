from random import random
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')


def test_data(k: float = 1.0, b: float = 0.1, half_disp: float = 0.05, n: int = 100, x_step: float = 0.01) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Генерируюет линию вида y = k*x + b + dy, где dy - аддитивный шум с амплитудой half_disp
    :param k: наклон линии
    :param b: смещение по y
    :param half_disp: амплитуда разброса данных
    :param n: количество точек
    :param x_step: шаг между соседними точками
    :return: кортеж значенией по x и y
    """
    import random
    return np.array([i * x_step for i in range(n + 1)]), \
           np.array([i * x_step * k + b + random.uniform(-half_disp, half_disp) for i in range(n + 1)])


def n_linear_regression(data_rows: np.ndarray) -> np.ndarray:

    """
    H_ij = Σx_i * x_j, i in [0, rows - 1] , j in [0, rows - 1]
    H_ij = Σx_i, j = rows i in [rows, :]
    H_ij = Σx_j, j in [:, rows], i = rows

           | Σkx * xi^2    + Σky * xi * yi + b * Σxi - Σzi * xi|\n
    grad = | Σkx * xi * yi + Σky * yi^2    + b * Σyi - Σzi * yi|\n
           | Σyi * ky      + Σxi * kx                - Σzi     |\n

    x_0 = [1,...1, 0] =>

           | Σ xi^2    + Σ xi * yi - Σzi * xi|\n
    grad = | Σ xi * yi + Σ yi^2    - Σzi * yi|\n
           | Σxi       + Σ yi      - Σzi     |\n

    :param data_rows:  состоит из строк вида: [x_0,x_1,...,x_n, f(x_0,x_1,...,x_n)]
    :return:
    """
    s_rows, s_cols = data_rows.shape

    hessian = np.zeros((s_cols, s_cols,), dtype=float)

    grad = np.zeros((s_cols,), dtype=float)

    x_0 = np.zeros((s_cols,), dtype=float)

    for row in range(s_cols - 1):
        x_0[row] = 1.0
        for col in range(row + 1):
            value = np.sum(data_rows[:, row] @ data_rows[:, col])
            hessian[row, col] = value
            hessian[col, row] = value

    for i in range(s_cols):
        value = np.sum(data_rows[:, i])
        hessian[i, s_cols - 1] = value
        hessian[s_cols - 1, i] = value

    hessian[s_cols - 1, s_cols - 1] = data_rows.shape[0]

    for row in range(s_cols - 1):
        grad[row] = np.sum(hessian[row, 0: s_cols - 1]) - np.dot(data_rows[:, s_cols - 1], data_rows[:, row])

    grad[s_cols - 1] = np.sum(hessian[s_cols - 1, 0 : s_cols - 1]) - np.sum(data_rows[:, s_cols - 1])

    return x_0 - np.linalg.inv(hessian) @ grad


def test_date_nd(surf_params: np.ndarray = np.array([1, 2, 3, 4, 5, 6, 10000]),
                 arg_range: float = 10, rand_range: float = 0.05, n_points: int = 100) -> np.ndarray:
    data = np.zeros((n_points, surf_params.size))
    import random
    for i in range(surf_params.size-1):
        data[:, i] = np.array([random.uniform(-0.5*arg_range, 0.5*arg_range) for _ in range(n_points)])
        data[:, surf_params.size-1] += data[:, i] * surf_params[i]
    data[:, surf_params.size-1] += \
        np.array([surf_params[surf_params.size-1] + random.uniform(-0.5*rand_range, 0.5*rand_range) for _ in range(n_points)])
    return data


def n_linear_reg_test(x_step: float=0.01,n:int=100):
    """
    Функция проверки работы метода n-мерной линейной регрессии:
    1) Посчитать тестовыe x, y и z используя функцию test_data_2d
    2) Получить с помошью bi_linear_regression значения kx, ky и b
    3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить
       регрессионную плоскость вида z = kx*x + ky*y + b
    :return:
    """
    from matplotlib import cm
    nd = test_date_nd()
    print(n_linear_regression(nd))

def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, half_disp: float = 1.01, n: int = 100,
                 x_step: float = 0.01, y_step: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует плоскость вида z = kx*x + ky*x + b + dz, где dz - аддитивный шум с амплитудой half_disp
    :param kx: наклон плоскости по x
    :param ky: наклон плоскости по y
    :param b: смещение по z
    :param half_disp: амплитуда разброса данных
    :param n: количество точек
    :param x_step: шаг между соседними точками по х
    :param y_step: шаг между соседними точками по y
    :returns: кортеж значенией по x, y и z
    """
    import random
    x = np.array([random.uniform(0.0, n * x_step) for i in range(n)])
    y = np.array([random.uniform(0.0, n * y_step) for i in range(n)])
    dz = np.array([b + random.uniform(-half_disp, half_disp) for i in range(n)])
    return x, y, x * kx + y * ky + dz


def distance_sum(x: np.ndarray, y: np.ndarray, k: float, b: float) -> float:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b при фиксированных k и b
    по формуле: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: значение параметра k (наклон)
    :param b: значение параметра b (смещение)
    :returns: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5
    """
    return np.sqrt(np.sum((y - x * k + b) ** 2))


def distance_field(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b, где k и b являются диапазонами
    значений. Формула расстояния для j-ого значения из набора k и k-ого значения из набора b:
    F(k_j, b_k) = (Σ(yi -(k_j * xi + b_k))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: массив значений параметра k (наклоны)
    :param b: массив значений параметра b (смещения)
    :returns: поле расстояний вида F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    """
    return np.array([[distance_sum(x, y, k_i, b_i) for k_i in k.flat] for b_i in b.flat])


def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Линейная регрессия.\n
    Основные формулы:\n
    yi - xi*k - b = ei\n
    yi - (xi*k + b) = ei\n
    (yi - (xi*k + b))^2 = yi^2 - 2*yi*(xi*k + b) + (xi*k + b)^2 = ei^2\n
    yi^2 - 2*(yi*xi*k + yi*b) + (xi^2 * k^2 + 2 * xi * k * b + b^2) = ei^2\n
    yi^2 - 2*yi*xi*k - 2*yi*b + xi^2 * k^2 + 2 * xi * k * b + b^2 = ei^2\n
    d ei^2 /dk = - 2*yi*xi + 2 * xi^2 * k + 2 * xi * b = 0\n
    d ei^2 /db = - 2*yi + 2 * xi * k + 2 * b = 0\n
    ====================================================================================================================\n
    d ei^2 /dk = (yi - xi * k - b) * xi = 0\n
    d ei^2 /db =  yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ(yi - xi * k) = n * b\n
    ====================================================================================================================\n
    Σyi - k * Σxi = n * b\n
    Σxi*yi - xi^2 * k - xi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*(Σyi - k * Σxi) / n = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*Σyi / n + k * (Σxi)^2 / n = 0\n
    Σxi*yi - Σxi*Σyi / n + k * ((Σxi)^2 / n - Σxi^2)  = 0\n
    Σxi*yi - Σxi*Σyi / n = -k * ((Σxi)^2 / n - Σxi^2)\n
    (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n) = k\n
    окончательно:\n
    k = (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n)\n
    b = (Σyi - k * Σxi) /n\n
    :param x: массив значений по x
    :param y: массив значений по y
    :returns: возвращает пару (k, b), которая является решением задачи (Σ(yi -(k * xi + b))^2)->min
    """
    sum_x = x.sum()
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    n = x.size
    k = (sum_xy - sum_x * sum_y / n) / (sum_xx - sum_x * sum_x / n)
    return k, (sum_y - k * sum_x) / n


def bi_linear_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """
    Билинейная регрессия.\n
    Основные формулы:\n
    zi - (yi * ky + xi * kx + b) = ei\n
    zi^2 - 2*zi*(yi * ky + xi * kx + b) + (yi * ky + xi * kx + b)^2 = ei^2\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + ((yi*ky)^2 + 2 * (xi*kx*yi*ky + b*yi*ky) + (xi*kx + b)^2)\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx + b)^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b + b^2\n
    ====================================================================================================================\n
    d Σei^2 /dkx = Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b = 0\n
    d Σei^2 /dky = Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi = 0\n
    d Σei^2 /db  = Σ-zi + yi*ky + xi*kx = 0\n
    ====================================================================================================================\n
    d Σei^2 /dkx / dkx = Σ xi^2\n
    d Σei^2 /dkx / dky = Σ xi*yi\n
    d Σei^2 /dkx / db  = Σ xi\n
    ====================================================================================================================\n
    d Σei^2 /dky / dkx = Σ xi*yi\n
    d Σei^2 /dky / dky = Σ yi^2\n
    d Σei^2 /dky / db  = Σ yi\n
    ====================================================================================================================\n
    d Σei^2 /db / dkx = Σ xi\n
    d Σei^2 /db / dky = Σ yi\n
    d Σei^2 /db / db  = n\n
    ====================================================================================================================\n
    Hesse matrix:\n
    || d Σei^2 /dkx / dkx;  d Σei^2 /dkx / dky;  d Σei^2 /dkx / db ||\n
    || d Σei^2 /dky / dkx;  d Σei^2 /dky / dky;  d Σei^2 /dky / db ||\n
    || d Σei^2 /db  / dkx;  d Σei^2 /db  / dky;  d Σei^2 /db  / db ||\n
    ====================================================================================================================\n
    Hesse matrix:\n
                   | Σ xi^2;  Σ xi*yi; Σ xi |\n
    H(kx, ky, b) = | Σ xi*yi; Σ yi^2;  Σ yi |\n
                   | Σ xi;    Σ yi;    n    |\n
    ====================================================================================================================\n
                      | Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b |\n
    grad(kx, ky, b) = | Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi |\n
                      | Σ-zi + yi*ky + xi*kx                |\n
    ====================================================================================================================\n
    Окончательно решение:\n
    |kx|   |1|\n
    |ky| = |1| -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
    | b|   |0|\n

    :param x: массив значений по x
    :param y: массив значений по y
    :param z: массив значений по z
    :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
    """
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_z = np.sum(z)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    sum_yy = np.sum(y * y)
    sum_zy = np.sum(z * y)
    sum_zx = np.sum(z * x)
    n = x.size

    hesse = np.array([[sum_xx, sum_xy, sum_x],
                      [sum_xy, sum_yy, sum_y],
                      [sum_x, sum_y, n]])
    hesse = np.linalg.inv(hesse)

    dkx = sum_xy + sum_xx - sum_zx
    dky = sum_yy + sum_xy - sum_zy
    db = sum_y + sum_x - sum_z

    return 1.0 - (hesse[0][0] * dkx + hesse[0][1] * dky + hesse[0][2] * db), \
           1.0 - (hesse[1][0] * dkx + hesse[1][1] * dky + hesse[1][2] * db), \
           - (hesse[2][0] * dkx + hesse[2][1] * dky + hesse[2][2] * db)


def poly_regression(x: np.ndarray, y: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Полином: y = Σ_j x^j * bj
    Отклонение: ei =  yi - Σ_j xi^j * bj
    Минимизируем: Σ_i(yi - Σ_j xi^j * bj)^2 -> min
    Σ_i(yi - Σ_j xi^j * bj)^2 = Σ_iyi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2
    условие минимума: d/dbj Σ_i ei = d/dbj (Σ_i yi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2) = 0
    :param x: массив значений по x
    :param y: массив значений по y
    :param order: порядок полинома
    :return: набор коэффициентов bi полинома y = Σx^i*bi
    """
    a_m = np.zeros((order, order,), dtype=float)
    c_m = np.zeros((order,), dtype=float)
    n = x.size
    copy_x = 1
    for row in range(order):
        c_m[row] = np.sum(y * (copy_x)) / n
        copy_copy_x = copy_x
        for col in range(row + 1):
            a_m[row][col] = np.sum(copy_copy_x) / n
            a_m[col][row] = a_m[row][col]
            copy_copy_x = copy_copy_x * x
        copy_x= copy_x * x
    return np.linalg.inv(a_m) @ c_m


def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    :param x: массив значений по x
    :param b: массив коэффициентов полинома
    :returns: возвращает полином yi = Σxi^j*bj
    """
    result = b[0] + b[1] * x
    copy_x = x
    for i in range(2, b.size):
        copy_x = copy_x * x
        result += b[i] * copy_x
    return result


def distance_field_test():
    """
    Функция проверки поля расстояний:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Задать интересующие нас диапазоны k и b (np.linspace...)
    3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.
    4) Проанализировать результат (смысл этой картинки в чём...)
    :return:
    """
    x, y = test_data()
    k_, b_ = linear_regression(x, y)
    print(f"y(x) = {k_:1.5} * x + {b_:1.5}")
    k = np.linspace(-2.0, 2.0, 128, dtype=float)
    b = np.linspace(-2.0, 2.0, 128, dtype=float)
    z = distance_field(x, y, k, b)
    plt.imshow(z, extent=[k.min(), k.max(), b.min(), b.max()])
    plt.plot(k_, b_, 'r*')
    plt.xlabel("k")
    plt.ylabel("b")
    plt.grid(True)
    plt.show()


def linear_reg_test():
    """
    Функция проверки работы метода линейной регрессии:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Получить с помошью linear_regression значения k и b
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную прямую вида y = k*x + b
    :return:
    """
    x, y = test_data()
    k, b = linear_regression(x, y)
    print(f"y(x) = {k:1.5} * x + {b:1.5}")
    plt.plot([0, 1.0], [b, k + b], 'g')
    plt.plot(x, y, 'r.')
    plt.show()


def bi_linear_reg_test():
    """
    Функция проверки работы метода билинейной регрессии:
    1) Посчитать тестовыe x, y и z используя функцию test_data_2d
    2) Получить с помошью bi_linear_regression значения kx, ky и b
    3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить
       регрессионную плоскость вида z = kx*x + ky*y + b
    :return:
    """
    from matplotlib import cm
    x, y, z = test_data_2d()
    kx, ky, b = bi_linear_regression(x, y, z)
    print(f"z(x, y) = {kx:1.5} * x + {ky:1.5} * y + {b:1.5}")
    x_, y_ = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    z_ = kx * x_ + y_ * ky + b
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(x, y, z, 'r.')
    surf = ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm, linewidth=0, antialiased=False, edgecolor='none', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def poly_reg_test():
    """
    Функция проверки работы метода полиномиальной регрессии:
    1) Посчитать тестовыe x, y используя функцию test_data
    2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную кривую. Для построения кривой использовать метод polynom
    :return:
    """
    x, y = test_data()
    coefficients = poly_regression(x, y)
    y_ = polynom(x, coefficients)
    print(f"y(x) = {' + '.join(f'{coefficients[i]:.4} * x^{i}' for i in range(coefficients.size))}")
    plt.plot(x, y_, 'g')
    plt.plot(x, y, 'r.')
    plt.show()


import random
def test_square_data_2d(surf_param: Tuple[float, float, float, float, float, float] = (1.0, 2.0, 3.0, 1.0, 2.0, 3.0),
                        args_range: float = 1.0, rand_range: float = 0.10, n: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([random.uniform(0.0, args_range) for _ in range(n)])
    y = np.array([random.uniform(0.0, args_range) for _ in range(n)])
    dz = np.array([random.uniform(-rand_range, rand_range) for _ in range(n)])
    Fe = x * x * surf_param[0] + x * y * surf_param[1] + y * y * surf_param[2] + \
         x * surf_param[3] + y * surf_param[4] + surf_param[5]
    Des= dz + Fe
    return x, y, Des


def square_regression_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    b = [x * x, x * y, y * y, x, y, np.array([1])]
    m = np.zeros((len(b), len(b)), dtype=float)
    d = np.zeros((6,), dtype=float)
    for rows in range(6):
        d[rows] = (b[rows] * z).sum()
        for cols in range(rows + 1):
            m[rows, cols] = (b[rows] * b[cols]).sum()
            m[cols, rows] = m[rows, cols]
    m[5, 5] = x.size
    return np.linalg.inv(m) @ d


def square_linear_reg_test():
    from matplotlib import cm
    x, y, z = test_square_data_2d()
    x1, y1 = np.meshgrid(np.linspace(np.min(x), np.max(x), 200), np.linspace(np.min(y), np.max(y), 200))#100
    surf_param = square_regression_2d(x, y, z)
    print(f"z(x, y) = {surf_param[0]:1.5} x^2 + {surf_param[1]:1.5} xy + {surf_param[2]:1.5} y^2 +"
          f" {surf_param[3]:1.5} x + {surf_param[4]:1.5} y + {surf_param[5]:1.5}")
    z1 = surf_param[0] * x1 ** 2 + surf_param[1] * x1 * y1 + surf_param[2] * y1 ** 2 + surf_param[3] * x1 + \
         surf_param[4] * y1 + surf_param[5]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(x, y, z, 'r.')
    surfs = ax.plot_surface(x1, y1, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False, edgecolor='none',alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(surfs, shrink=0.5, aspect=5)
    plt.show()


if __name__ == "__main__":
    #distance_field_test()
    #linear_reg_test()
    #bi_linear_reg_test()
    #poly_reg_test()
    n_linear_reg_test()
    #square_linear_reg_test()
