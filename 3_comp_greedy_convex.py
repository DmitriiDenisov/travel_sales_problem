import pickle
import time
import os
import plotly.graph_objects as go
import numpy as np

from convex_hull.convex_hull_main import convex_hull_solver
from generation_utils import generate_points
from greedy_algo.greedy_algo import greedy_solver


def measure_result(algorithm, points):
    result = algorithm(points)
    return result[0]


n_values = list(range(5, 200))
greedy_distances = []
convex_hull_distances = []

for n in n_values:
    print(f"Calculating for n = {n}")
    points = generate_points(n)

    greedy_distance = measure_result(greedy_solver, points)
    greedy_distances.append(greedy_distance)

    convex_hull_distance = measure_result(convex_hull_solver, points)
    convex_hull_distances.append(convex_hull_distance)

# Построение графика с использованием Plotly
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=n_values, y=greedy_distances, mode='lines+markers', name='Greedy Algorithm', line=dict(width=3)))
fig.add_trace(go.Scatter(x=n_values, y=convex_hull_distances, mode='lines+markers', name='Convex Hull Algorithm',
                         line=dict(width=3)))

fig.update_layout(
    title=dict(text='Distance Found by Greedy and Convex Hull Algorithms vs Size of Matrix', font=dict(size=25)),
    xaxis_title='Size of Matrix (n)',
    yaxis_title='Distance Found',
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
