import math


def solve_tsp_with_bnb(distance_matrix):
    city_count = len(distance_matrix)
    optimal_route = [None] * (city_count + 1)
    has_visited = [False] * city_count
    best_cost = float('infinity')

    def finalize_route(route_so_far):
        optimal_route[:city_count + 1] = route_so_far[:]
        optimal_route[city_count] = route_so_far[0]

    def minimum_edge(distance_matrix, node):
        min_distance = float('infinity')
        for k in range(city_count):
            if distance_matrix[node][k] < min_distance and node != k:
                min_distance = distance_matrix[node][k]
        return min_distance

    def next_minimum_edge(distance_matrix, node):
        min_one, min_two = float('infinity'), float('infinity')
        for j in range(city_count):
            if node == j:
                continue
            if distance_matrix[node][j] <= min_one:
                min_two = min_one
                min_one = distance_matrix[node][j]
            elif distance_matrix[node][j] <= min_two and distance_matrix[node][j] != min_one:
                min_two = distance_matrix[node][j]
        return min_two

    def explore_routes(distance_matrix, current_bound, current_cost, level, route_so_far, has_visited):
        nonlocal best_cost
        if level == city_count:
            if distance_matrix[route_so_far[level - 1]][route_so_far[0]] != 0:
                current_result = current_cost + distance_matrix[route_so_far[level - 1]][route_so_far[0]]
                if current_result < best_cost:
                    finalize_route(route_so_far)
                    best_cost = current_result
            return

        for i in range(city_count):
            if distance_matrix[route_so_far[level - 1]][i] != 0 and not has_visited[i]:
                temp_bound = current_bound
                current_cost += distance_matrix[route_so_far[level - 1]][i]

                if level == 1:
                    current_bound -= ((minimum_edge(distance_matrix, route_so_far[level - 1]) + minimum_edge(
                        distance_matrix, i)) / 2)
                else:
                    current_bound -= ((next_minimum_edge(distance_matrix, route_so_far[level - 1]) + minimum_edge(
                        distance_matrix, i)) / 2)

                if current_bound + current_cost < best_cost:
                    route_so_far[level] = i
                    has_visited[i] = True
                    explore_routes(distance_matrix, current_bound, current_cost, level + 1, route_so_far, has_visited)

                current_cost -= distance_matrix[route_so_far[level - 1]][i]
                current_bound = temp_bound

                has_visited = [False] * len(has_visited)
                for j in range(level):
                    if route_so_far[j] != -1:
                        has_visited[route_so_far[j]] = True

    initial_bound = 0
    route_so_far = [-1] * (city_count + 1)
    has_visited = [False] * city_count

    for i in range(city_count):
        initial_bound += (minimum_edge(distance_matrix, i) + next_minimum_edge(distance_matrix, i))

    initial_bound = math.ceil(initial_bound / 2)

    has_visited[0] = True
    route_so_far[0] = 0
    # def explore_routes(distance_matrix, current_bound, current_cost, level, route_so_far, has_visited):
    explore_routes(distance_matrix, initial_bound, 0, 1, route_so_far, has_visited)
    return best_cost, optimal_route
