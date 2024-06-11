import heapq
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation_utils import calculate_distances


def branch_and_bound_solver(points):
    matrix = calculate_distances(points)
    n = len(matrix)  # Number of cities

    # Function to calculate the lower bound for a given path
    def calculate_lower_bound(matrix, path):
        lower_bound = 0
        visited = set(path)
        # Calculate the lower bound for the current path
        for i in range(len(path) - 1):
            lower_bound += matrix[path[i]][path[i + 1]]

        if len(path) < n:
            # Add minimum edges going out from the last visited city and returning to the starting city
            last_city = path[-1]
            min_outgoing = min(matrix[last_city][j] for j in range(n) if j not in visited)
            min_incoming = min(matrix[i][path[0]] for i in range(n) if i not in visited)
            lower_bound += min_outgoing + min_incoming

        return lower_bound

    # Priority queue to store live nodes of search tree
    pq = []
    initial_path = [0]  # Start from the first city
    lower_bound = calculate_lower_bound(matrix, initial_path)
    heapq.heappush(pq, (lower_bound, initial_path))

    min_distance = float('inf')
    min_path = []

    while pq:
        current_bound, current_path = heapq.heappop(pq)
        if len(current_path) == n:
            current_distance = current_bound + matrix[current_path[-1]][current_path[0]]
            if current_distance < min_distance:
                min_distance = current_distance
                min_path = current_path + [current_path[0]]
        else:
            for i in range(n):
                if i not in current_path:
                    new_path = current_path + [i]
                    new_bound = calculate_lower_bound(matrix, new_path)
                    if new_bound < min_distance:
                        heapq.heappush(pq, (new_bound, new_path))

    return min_distance, min_path
