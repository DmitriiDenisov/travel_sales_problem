import random
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def generate_points(n_points=30):
    points = np.random.rand(n_points, 2) * 100
    points = np.round(points, 2)
    return points

def calculate_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = np.linalg.norm(points[i] - points[j])
            distances[j, i] = distances[i, j]
    return np.round(distances)


def distort_distances(distances):
    n = len(distances)
    distorted = np.zeros_like(distances)
    for i in range(n):
        for j in range(i + 1, n):
            t = np.random.normal()
            distorted[i, j] = distances[i, j] * (1 + abs(t))
            distorted[j, i] = distorted[i, j]
    return distorted

# Шаг 4-8: Размещение точек на плоскости с минимизацией ошибок
def place_points(distorted):
    n = len(distorted)
    placed_points = np.zeros((n, 2))
    placed_points[1, 0] = distorted[0, 1]

    for i in range(2, n):
        possible_positions = []
        pairs = []
        for j in range(i):
            for k in range(j + 1, i):
                pairs.append((j, k))

        if len(pairs) > 20:
            pairs = random.sample(pairs, 20)

        for (j, k) in pairs:
            d_ij = distorted[i, j]
            d_ik = distorted[i, k]
            d_jk = np.linalg.norm(placed_points[j] - placed_points[k])

            if d_jk == 0:
                continue

            # Находим координаты пересечения двух окружностей
            a = (d_ij**2 - d_ik**2 + d_jk**2) / (2 * d_jk)
            h_squared = d_ij**2 - a**2
            if h_squared < 0:
                # Окружности не пересекаются, выбираем ближайшую точку на границе
                h = np.sqrt(abs(h_squared))
                mid_point = placed_points[j] + a * (placed_points[k] - placed_points[j]) / d_jk
                offset = h * np.array([-(placed_points[k, 1] - placed_points[j, 1]) / d_jk, (placed_points[k, 0] - placed_points[j, 0]) / d_jk])
                candidate1 = mid_point + offset
                candidate2 = mid_point - offset
                chosen_candidate = random.choice([candidate1, candidate2])
                possible_positions.append(chosen_candidate)
            else:
                h = np.sqrt(h_squared)
                mid_point = placed_points[j] + a * (placed_points[k] - placed_points[j]) / d_jk
                offset = h * np.array([-(placed_points[k, 1] - placed_points[j, 1]) / d_jk, (placed_points[k, 0] - placed_points[j, 0]) / d_jk])
                candidate1 = mid_point + offset
                candidate2 = mid_point - offset
                chosen_candidate = random.choice([candidate1, candidate2])
                possible_positions.append(chosen_candidate)

        if possible_positions:
            placed_points[i] = np.mean(possible_positions, axis=0)
        else:
            placed_points[i] = np.array([np.nan, np.nan])

    return placed_points

# Вычисление невязки
def calculate_discrepancy(original_distances, new_distances):
    discrepancy = np.sqrt(np.sum((original_distances - new_distances)**2))
    return discrepancy

# Основная функция
def main(n):
    points = generate_points(n)
    distances = calculate_distances(points)
    distorted = distort_distances(distances)
    placed_points = place_points(distorted)
    new_distances = calculate_distances(placed_points)
    discrepancy = calculate_discrepancy(distances, new_distances)

    return points, placed_points, discrepancy

def plot():
    # Генерация и отображение результатов
    n = 10
    original_points, placed_points, discrepancy = main(n)

    print(f'Discrepancy: {discrepancy}')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(original_points[:, 0], original_points[:, 1], c='blue', label='Original Points')
    plt.title('Original Points')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(placed_points[:, 0], placed_points[:, 1], c='red', label='Placed Points')
    plt.title('Placed Points')
    plt.legend()

    plt.show()
