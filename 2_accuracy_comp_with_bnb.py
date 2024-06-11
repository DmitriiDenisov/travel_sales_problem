import pickle
import time
import os
import plotly.graph_objects as go
import numpy as np

from ant_colony_tsp.aco import ant_colony_solver
from branch_and_bound.bnb import branch_and_bound_solver
from convex_hull.convex_hull_main import convex_hull_solver
from generation_utils import generate_points
from greedy_algo.greedy_algo import greedy_solver


def measure_time_and_result(algorithm, points):
    start_time = time.time()
    result = algorithm(points)
    end_time = time.time()
    return end_time - start_time, result

results_file = 'tsp_results_with_deviation.pkl'


if os.path.exists(results_file):
    with open(results_file, 'rb') as file:
        results_dict = pickle.load(file)
else:
    results_dict = {
        'branch_and_bound': {'times': [], 'results': []},
        'ant_colony': {'times': [], 'results': []},
        'convex_hull': {'times': [], 'results': []},
        'greedy': {'times': [], 'results': []},
        'n_values': [],
        'deviations_ant_colony': [],
        'deviations_convex_hull': [],
        'deviations_greedy': []
    }

n_values = list(range(5, 17))
start_index = len(results_dict['n_values'])

for n in n_values[start_index:]:
    print(n)
    points = generate_points(n)

    # Выполнение и сохранение результатов метода ветвей и границ
    time_bnb, result_bnb = measure_time_and_result(branch_and_bound_solver, points)
    results_dict['branch_and_bound']['times'].append(time_bnb)
    results_dict['branch_and_bound']['results'].append(result_bnb)

    # Выполнение и сохранение результатов муравьиного алгоритма
    time_ant, result_ant = measure_time_and_result(ant_colony_solver, points)
    results_dict['ant_colony']['times'].append(time_ant)
    results_dict['ant_colony']['results'].append(result_ant)

    # Выполнение и сохранение результатов опоясывающего метода
    time_hull, result_hull = measure_time_and_result(convex_hull_solver, points)
    results_dict['convex_hull']['times'].append(time_hull)
    results_dict['convex_hull']['results'].append(result_hull)

    # Выполнение и сохранение результатов жадного алгоритма
    time_greedy, result_greedy = measure_time_and_result(greedy_solver, points)
    results_dict['greedy']['times'].append(time_greedy)
    results_dict['greedy']['results'].append(result_greedy)

    # Расчет отклонений от оптимального решения (метод ветвей и границ)
    deviation_ant_colony = abs(result_ant[0] - result_bnb[0])
    deviation_convex_hull = abs(result_hull[0] - result_bnb[0])
    deviation_greedy = abs(result_greedy[0] - result_bnb[0])

    results_dict['deviations_ant_colony'].append(deviation_ant_colony)
    results_dict['deviations_convex_hull'].append(deviation_convex_hull)
    results_dict['deviations_greedy'].append(deviation_greedy)

    results_dict['n_values'].append(n)

    # Сохранение данных после каждой итерации
    with open(results_file, 'wb') as file:
        pickle.dump(results_dict, file)

# Построение графиков с использованием Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=results_dict['n_values'], y=results_dict['deviations_ant_colony'], mode='lines+markers', name='Ant Colony Deviation', line=dict(width=3)))
fig.add_trace(go.Scatter(x=results_dict['n_values'], y=results_dict['deviations_convex_hull'], mode='lines+markers', name='Convex Hull Deviation', line=dict(width=3)))
fig.add_trace(go.Scatter(x=results_dict['n_values'], y=results_dict['deviations_greedy'], mode='lines+markers', name='Greedy Deviation', line=dict(width=3)))

fig.update_layout(
    title=dict(text='Deviation from Optimal Solution vs Number of Cities', font=dict(size=25)),
    xaxis_title='Number of Cities (n)',
    yaxis_title='Deviation from Optimal Solution',
    legend_title='Algorithm',
    template='plotly_white',
    xaxis=dict(
        title=dict(font=dict(size=20)),
        tickfont=dict(size=15)
    ),
    yaxis=dict(
        title=dict(font=dict(size=20)),
        tickfont=dict(size=15)
    ),
    legend=dict(font=dict(size=15))
)

fig.show()
