# Импорт необходимых модулей
from branch_and_bound import branch_and_bound_solver
from ant_colony_tsp import ant_colony_solver
from convex_hull import convex_hull_solver


def solve_with_branch_and_bound(distance_matrix):
    """
    Решение задачи коммивояжера методом ветвей и границ.
    """
    return branch_and_bound_solver(distance_matrix)


def solve_with_ant_colony(distance_matrix, num_ants=50, num_iterations=100):
    """
    Решение задачи коммивояжера муравьиным алгоритмом.
    """
    return ant_colony_solver(distance_matrix, num_ants, num_iterations)


def solve_with_convex_hull(distance_matrix):
    """
    Решение задачи коммивояжера опоясывающим методом.
    """
    return convex_hull_solver(distance_matrix)


# Пример использования функций
distance_matrix = [[0, 10, 15, 20],
                   [10, 0, 35, 25],
                   [15, 35, 0, 30],
                   [20, 25, 30, 0]]

print(solve_with_branch_and_bound(distance_matrix))
print(solve_with_ant_colony(distance_matrix))
print(solve_with_convex_hull(distance_matrix))
