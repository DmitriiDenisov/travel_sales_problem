import numpy as np


def greedy_solver(points):
    n = len(points)
    unvisited = list(range(n))
    current_city = unvisited.pop(0)
    tour = [current_city]
    total_cost = 0

    while unvisited:
        next_city = min(unvisited, key=lambda city: np.linalg.norm(points[current_city] - points[city]))
        total_cost += np.linalg.norm(points[current_city] - points[next_city])
        current_city = next_city
        tour.append(current_city)
        unvisited.remove(current_city)

    total_cost += np.linalg.norm(points[current_city] - points[tour[0]])
    return total_cost, tour
