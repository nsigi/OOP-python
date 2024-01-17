import numpy as np
import matplotlib.pyplot as plt


def gaussian_cluster(cx: float = 0.0, cy: float = 0.0, sigma_x: float = 0.1, sigma_y: float = 0.1,
                     n_points: int = 1024):
    """
    Двумерный кластер точек, распределённых нормально с центром в
    точке с координатами cx, cy и разбросом sigma_x, sigma_y.
    """
    return np.hstack((np.random.normal(cx, sigma_x, n_points).reshape((n_points, 1)),
                      np.random.normal(cy, sigma_y, n_points).reshape((n_points, 1))))


def k_means(data, k, max_iterations=100):
    # init центров кластеров
    centers = data[np.random.choice(data.shape[0], size=k, replace=False)]

    for _ in range(max_iterations):
        # Определение ближайшего центра кластера для каждого наблюдения
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=-1)
        labels = np.argmin(distances, axis=-1)

        # calc новых центров кластеров
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # check на сходимость
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return labels, centers


def plot_clusters(data, centers, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-means Clustering')
    plt.show()


def separated_clusters():
    """
    Пример с пятью разрозненными распределениями точек на плоскости.
    """
    # Пример данных
    data = np.vstack((
        gaussian_cluster(cx=0.5, n_points=512),
        gaussian_cluster(cx=1.0, n_points=512),
        gaussian_cluster(cx=1.5, n_points=512),
        gaussian_cluster(cx=2.0, n_points=512),
        gaussian_cluster(cx=2.5, n_points=512)))
    k = 5
    labels, centers = k_means(data, k)

    # Визуализация кластеров
    plot_clusters(data, centers, labels)


def merged_clusters():
    """
    Пример с кластеризацией пятна.
    """
    # Пример данных
    data = gaussian_cluster(n_points=512 * 5)
    k = 5
    labels, centers = k_means(data, k)

    # Визуализация кластеров
    plot_clusters(data, centers, labels)


if __name__ == "__main__":
    """
    Сюрприз-сюрприз! Лучший сюрприз - это автомат. Вызов функций "merged_clusters" и "separated_clusters".
    """
    merged_clusters()
    separated_clusters()