from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def pack_color_code(red: int, green: int, blue: int) -> str:
    """
    Создаёт цветовую кодировку, совместимую с параметром matplotlib color
    """
    return f"#{''.join('{:02X}'.format(clamp(a, 0, 255)) for a in (red, green, blue))}"


def unpack_color_code(code: str) -> Tuple[int, int, int]:
    return int(code[-6:-4], 16), int(code[-4:-2], 16), int(code[-2:], 16)


def gaussian_cluster(cx: float = 0.0, cy: float = 0.0, sigma_x: float = 0.1, sigma_y: float = 0.1, n_points: int = 1024):
    """
    Двумерный кластер точек, распределённых нормально с центром в
    точке с координатами cx, cy и разбросом sigma_x, sigma_y.
    """
    return np.hstack((np.random.normal(cx, sigma_x, n_points).reshape((n_points, 1)),
                      np.random.normal(cy, sigma_y, n_points).reshape((n_points, 1))))


def color_map_nonlinear(map_amount: int = 3) -> List[str]:
    """
    Генератор цветов для кластеров
    """
    colors = []
    dx = 1.0 / (map_amount - 1) if map_amount > 1 else 1.0
    for i in range(map_amount):
        xi = i * dx
        colors.append(pack_color_code(int(255.0 * max(1.0 - (2.0 * xi - 1.0) ** 2, 0.0)),
                                      int(255.0 * max(1.0 - (2.0 * xi - 2.0) ** 2, 0.0)),
                                      int(255.0 * max(1.0 - (2.0 * xi - 0.0) ** 2, 0.0))))
    return colors


def draw_cluster(cluster: np.ndarray, rgb: Union[Tuple[int, int, int], str] = (255, 0, 0), axes=None, show=False):
    axes = plt.gca() if axes is None else axes
    if isinstance(rgb, str):
        axes.plot(cluster[:, 0], cluster[:, 1], '.', color=rgb)
    if isinstance(rgb, tuple):
        axes.plot(cluster[:, 0], cluster[:, 1], '.', color=pack_color_code(*rgb))
    if show:
        plt.show()


def draw_clusters(clusters: List[np.ndarray],
                  colors: Union[List[Tuple[int, int, int]], List[str], None] = None,
                  cluster_centers: Union[List[np.ndarray], None] = None,
                  title="figure"):
    colors = color_map_nonlinear(len(clusters))if colors is None else colors
    fig, axes = plt.subplots(1)
    legend = []
    for cluster_index, (cluster, color) in enumerate(zip(clusters, colors)):
        draw_cluster(cluster, color, axes=axes)
        legend.append(f"cluster {cluster_index}")
    if cluster_centers is not None:
        # цвета маркеров сделаем чуть-чуть темнее, что бы они выделялись
        marker_colors = tuple(pack_color_code(*tuple(int(v * 0.5) for v in unpack_color_code(code))) for code in colors)
        for center, color in zip(cluster_centers, marker_colors):
            axes.plot(center[0], center[1], marker='*', markersize=10, color=color)
    axes.set_title(title)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend(legend, loc=1)
    axes.axis('equal')
    axes.grid(True)
    plt.show()


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return np.linalg.norm(right - left)  # , axis=1)


def gauss_core(value: Union[np.ndarray, float], sigma: float = 0.5) -> Union[np.ndarray, float]:
    return np.exp(-value * 0.5 / (sigma * sigma))


def flat_core(value: Union[np.ndarray, float], sigma: float = 0.5) -> np.ndarray:
    return np.exp(-value * 0.5 / (sigma * sigma))
