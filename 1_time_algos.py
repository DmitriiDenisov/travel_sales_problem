import pickle
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os


from ant_colony_tsp.aco import ant_colony_solver
from branch_and_bound.bnb import branch_and_bound_solver
from convex_hull.convex_hull_main import convex_hull_solver
from generation_utils import generate_points


def measure_time(algorithm, points):
    start_time = time.time()
    algorithm(points)
    end_time = time.time()
    return end_time - start_time

results_file = '1_time_tsp_results.pkl'
results_file = '3_times_tsp_results.pkl'

# Инициализация или загрузка данных
if os.path.exists(results_file):
    with open(results_file, 'rb') as file:
        times_dict = pickle.load(file)
else:
    times_dict = {
        'branch_and_bound': [],
        'ant_colony': [],
        'convex_hull': [],
        'n_values': []
    }

n_values = list(range(5, 1000))
start_index = len(times_dict['n_values'])


for n in n_values[start_index:]:
    print(n)
    points = generate_points(n)

    # times_bnb = [measure_time(branch_and_bound_solver, points) for _ in range(3)]
    # avg_time_bnb = sum(times_bnb) / len(times_bnb)
    # times_dict['branch_and_bound'].append(avg_time_bnb)

    # times_ant = [measure_time(ant_colony_solver, points) for _ in range(3)]
    # avg_time_ant = sum(times_ant) / len(times_ant)
    # times_dict['ant_colony'].append(avg_time_ant)

    times_hull = [measure_time(convex_hull_solver, points) for _ in range(3)]
    avg_time_hull = sum(times_hull) / len(times_hull)
    times_dict['convex_hull'].append(avg_time_hull)

    times_dict['n_values'].append(n)

    # Сохранение данных после каждой итерации
    with open(results_file, 'wb') as file:
        pickle.dump(times_dict, file)


# Построение графика с использованием Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=times_dict['n_values'], y=times_dict['branch_and_bound'], mode='lines+markers', name='Branch and Bound', line=dict(width=3)))
fig.add_trace(go.Scatter(x=times_dict['n_values'], y=times_dict['ant_colony'], mode='lines+markers', name='Ant Colony', line=dict(width=3)))
fig.add_trace(go.Scatter(x=times_dict['n_values'], y=times_dict['convex_hull'], mode='lines+markers', name='Convex Hull', line=dict(width=3)))

fig.update_layout(
    title=dict(text='Execution Time vs Number of Cities', font=dict(size=25)),
    xaxis_title='Number of Cities (n)',
    yaxis_title='Execution Time (seconds)',
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
