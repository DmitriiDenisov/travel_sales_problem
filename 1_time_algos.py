import time
import matplotlib.pyplot as plt

# Функция для измерения времени выполнения
from ant_colony_tsp.aco import ant_colony_solver
from branch_and_bound.bnb import branch_and_bound_solver
from convex_hull.convex_hull_main import convex_hull_solver
from generation_utils import generate_points


def measure_time(algorithm, points):
    start_time = time.time()
    algorithm(points)
    end_time = time.time()
    return end_time - start_time


# Список размеров задач (количество городов)
n_values = [5, 7, 10, 13]

# Списки для хранения времени выполнения
times_branch_and_bound = []
times_ant_colony = []
times_convex_hull = []

# Измерение времени выполнения для каждого значения n
for n in n_values:
    points = generate_points(n)

    print('BNB')
    time_branch_and_bound = measure_time(branch_and_bound_solver, points)
    times_branch_and_bound.append(time_branch_and_bound)

    print('Ant')
    time_ant_colony = measure_time(ant_colony_solver, points)
    times_ant_colony.append(time_ant_colony)

    print('Convex Hull')
    time_convex_hull = measure_time(convex_hull_solver, points)
    times_convex_hull.append(time_convex_hull)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(n_values, times_branch_and_bound, label='Branch and Bound')
plt.plot(n_values, times_ant_colony, label='Ant Colony')
plt.plot(n_values, times_convex_hull, label='Convex Hull')
plt.xlabel('Number of Cities (n)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of Cities')
plt.legend()
plt.grid(True)
plt.show()
