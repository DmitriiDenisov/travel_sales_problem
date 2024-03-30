import numpy as np
import sys


def generate_cities(N):
    cities = np.random.rand(N, 2) * 100
    distance_matrix = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))
    return distance_matrix

def reduce_matrix(matrix):
    total_reduction = 0
    for i in range(matrix.shape[0]):
        if not np.all(np.isinf(matrix[i])):
            row_min = np.min(matrix[i][np.isfinite(matrix[i])])
            matrix[i, :] = np.where(np.isfinite(matrix[i, :]), matrix[i, :] - row_min, matrix[i, :])
            total_reduction += row_min
    for j in range(matrix.shape[1]):
        if not np.all(np.isinf(matrix[:, j])):
            col_min = np.min(matrix[:, j][np.isfinite(matrix[:, j])])
            matrix[:, j] = np.where(np.isfinite(matrix[:, j]), matrix[:, j] - col_min, matrix[:, j])
            total_reduction += col_min
    return matrix, total_reduction

def calculate_bound(matrix, current_cost):
    reduced_matrix, reduction_cost = reduce_matrix(matrix.copy())
    return current_cost + reduction_cost

def branch_and_bound(distance_matrix):
    N = distance_matrix.shape[0]
    initial_matrix, initial_cost = reduce_matrix(distance_matrix.copy())
    nodes = [(initial_cost, [0], initial_matrix, 0)]
    best_cost = sys.maxsize
    best_path = None

    while nodes:
        current_cost, path, matrix, last_city = nodes.pop(0)

        if len(path) == N + 1:  # Ensure path is complete including return to start
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = path
            continue

        for next_city in range(N):
            if next_city not in path:
                new_matrix = matrix.copy()
                new_matrix[last_city, :] = np.inf
                new_matrix[:, next_city] = np.inf
                new_matrix[next_city, 0] = np.inf
                new_path = path + [next_city]
                if len(new_path) < N:
                    new_cost = current_cost + distance_matrix[last_city][next_city]
                else:  # Add cost to return to the start for the final path
                    new_cost = current_cost + distance_matrix[last_city][next_city] + distance_matrix[next_city][0]
                bound = calculate_bound(new_matrix, new_cost)

                if bound < best_cost:
                    nodes.append((bound, new_path + [0] if len(new_path) == N else new_path, new_matrix, next_city))
        nodes.sort(key=lambda x: x[0])

    return best_cost, best_path[:-1]  # Remove the duplicated start city for the final path presentation


# Assuming you have already defined distance_matrix_3_cities_np with dtype=float
# distance_matrix = distance_matrix_3_cities_np
N = 5
# distance_matrix = generate_cities(N)
distance_matrix_3_cities_np = np.array([
    [0, 10, 20],
    [10, 0, 25],
    [20, 25, 0]
], dtype=float)

best_cost, best_path = branch_and_bound(distance_matrix_3_cities_np)

print(f"Min distance: {best_cost}")
print("Path:", "->".join(map(str, [x+1 for x in best_path])))  # Adjust for 1-indexed city representation

