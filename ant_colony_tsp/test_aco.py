import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aco import ant_colony_solver


def test_3_city_example():
    points = np.array([
        [0, 0],
        [10, 0],
        [-6.25, 18.99]
    ])
    expected_cost = 55
    expected_path = [0, 1, 2, 0]
    reversed_path = list(reversed(expected_path))
    cost, path = ant_colony_solver(points, iterations=200, n_ants=10, alpha=1, beta=2, rho=0.5, Q=100)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path in [expected_path, reversed_path], f"Expected path {expected_path}, got {path}"

def test_4_city_example():
    points = np.array([
        [  3.9476268 ,  -2.23942679],  # Point A
        [ 13.58425129,   7.06799258],  # Point B
        [ -6.97628382, -17.13267757],  # Point C
        [-10.55559427,  12.30411178]   # Point D
    ])
    expected_cost = 86  # Update this based on the actual expected cost
    expected_path = [0, 1, 3, 2, 0]  # Update this based on the actual expected path
    reversed_path = list(reversed(expected_path))
    cost, path = ant_colony_solver(points, iterations=200, n_ants=10, alpha=1, beta=2, rho=0.5, Q=100)
    print(cost)
    print(path)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path in [expected_path, reversed_path] , f"Expected path {expected_path}, got {path}"

def test_5_city_example():
    points = np.array([
        [51, 92],  # Point 0
        [14, 71],  # Point 1
        [60, 20],  # Point 2
        [82, 86],  # Point 3
        [74, 74]   # Point 4
    ])

    expected_cost = 214
    expected_path = [0, 1, 2, 4, 3, 0]
    reversed_path = list(reversed(expected_path))
    cost, path = ant_colony_solver(points, iterations=200, n_ants=10, alpha=1, beta=2, rho=0.5, Q=100)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path in [expected_path, reversed_path] , f"Expected path {expected_path}, got {path}"

def test_6_city_example():
    points = np.array([
        [51, 92],  # Point 0
        [14, 71],  # Point 1
        [60, 20],  # Point 2
        [82, 86],  # Point 3
        [74, 74],  # Point 4
        [87, 99]   # Point 5
    ])
    expected_cost = 233
    expected_path = [0, 1, 2, 4, 3, 5, 0]
    reversed_path = list(reversed(expected_path))
    cost, path = ant_colony_solver(points, iterations=400, n_ants=10, alpha=1, beta=2, rho=0.5, Q=100)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path in [expected_path, reversed_path], f"Expected path {expected_path}, got {path}"

def test_7_city_example():
    points = np.array([
        [51, 92],  # Point 0
        [14, 71],  # Point 1
        [60, 20],  # Point 2
        [82, 86],  # Point 3
        [74, 74],  # Point 4
        [87, 99],  # Point 5
        [23,  2]   # Point 6
    ])
    expected_cost = 275

    cost, path = ant_colony_solver(points, iterations=400, n_ants=10, alpha=1, beta=2, rho=0.5, Q=100)
    print(cost)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    # don't check the exact path because there are too many possible paths
    # assert path in [expected_path], f"Expected path {expected_path}, got {path}"
