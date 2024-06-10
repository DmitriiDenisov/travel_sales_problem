import numpy as np
import random

from generation_utils import calculate_distances


class OptimizedGraph:
    def __init__(self, distance, default_pheromone_level=None):
        self.distance = np.array(distance, dtype=float)
        self.nodes = self.distance.shape[0]
        if default_pheromone_level is not None:
            self.pheromone = np.full(self.distance.shape, default_pheromone_level, dtype=float)
        else:
            self.pheromone = np.full(self.distance.shape, self.distance.mean() * 10, dtype=float)
        self.visibility = 1 / (self.distance + np.finfo(float).eps)  # Avoid division by zero

def rotate_tour_to_start_with_zero(tour):
    # Find the index of 0 in the tour
    zero_index = tour.index(0)
    # Rotate the tour to start from 0
    return tour[zero_index:] + tour[:zero_index]


def ant_tour(graph, alpha, beta):
    tour = [random.randint(0, graph.nodes - 1)]
    visited = np.zeros(graph.nodes, dtype=bool)
    visited[tour[0]] = True

    for _ in range(graph.nodes - 1):
        current_city = tour[-1]
        probs = (graph.pheromone[current_city] ** alpha) * (graph.visibility[current_city] ** beta)
        probs[visited] = 0  # Set visited cities' probabilities to 0
        next_city = np.random.choice(graph.nodes, 1, p=probs / probs.sum())[0]
        tour.append(next_city)
        visited[next_city] = True

    return tour


def update_pheromones(graph, tours, Q, rho):
    delta_pheromone = np.zeros_like(graph.pheromone)
    for tour in tours:
        tour_length = sum(graph.distance[tour[i], tour[(i + 1) % graph.nodes]] for i in range(graph.nodes))

        for i in range(graph.nodes):
            delta_pheromone[tour[i], tour[(i + 1) % graph.nodes]] += Q / tour_length
    graph.pheromone = graph.pheromone * (1 - rho) + delta_pheromone


def ant_colony_solver(points, iterations=100, n_ants=10, alpha=1, beta=2, rho=0.5, Q=100):
    distance_matrix = calculate_distances(points)
    graph = OptimizedGraph(distance_matrix)
    best_tour = None
    best_length = float('inf')

    for _ in range(iterations):
        tours = [ant_tour(graph, alpha, beta) for _ in range(n_ants)]
        lengths = [sum(graph.distance[tour[i], tour[(i + 1) % graph.nodes]] for i in range(graph.nodes)) for tour in
                   tours]

        # Update the best tour if found
        for length, tour in zip(lengths, tours):
            if length < best_length:
                best_length = length
                best_tour = tour

        update_pheromones(graph, tours, Q, rho)

    # Ensure the best tour starts from 0 before returning
    if best_tour is not None:
        best_tour = rotate_tour_to_start_with_zero(best_tour) + [0]

    return best_length, best_tour

