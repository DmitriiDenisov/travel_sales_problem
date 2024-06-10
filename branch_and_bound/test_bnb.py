import numpy as np

from bnb import branch_and_bound_solver

def test_3_city_example():
    points = np.array([
        [0, 0],
        [10, 0],
        [-6.25, 18.99]
    ])
    expected_cost = 55
    expected_path = [0, 1, 2, 0]
    cost, path = branch_and_bound_solver(points)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"

def test_4_city_example():
    points = np.array([
        [  3.9476268 ,  -2.23942679],  # Point A
        [ 13.58425129,   7.06799258],  # Point B
        [ -6.97628382, -17.13267757],  # Point C
        [-10.55559427,  12.30411178]   # Point D
    ])
    expected_cost = 86  # Update this based on the actual expected cost
    expected_path = [0, 1, 3, 2, 0]  # Update this based on the actual expected path
    cost, path = branch_and_bound_solver(points)
    print(cost)
    print(path)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"

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
    cost, path = branch_and_bound_solver(points)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"

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
    cost, path = branch_and_bound_solver(points)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"

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

    cost, path = branch_and_bound_solver(points)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
