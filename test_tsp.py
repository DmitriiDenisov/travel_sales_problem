from tsp import solve_tsp_with_bnb

def test_3_city_example():
    adj = [
        [0, 10, 20],
        [10, 0, 25],
        [20, 25, 0]
    ]
    expected_cost = 55
    expected_path = [0, 1, 2, 0]
    cost, path = solve_tsp_with_bnb(adj)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"

def test_4_city_example():
    adj = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    expected_cost = 80  # Update this based on the actual expected cost
    expected_path = [0, 1, 3, 2, 0]  # Update this based on the actual expected path
    cost, path = solve_tsp_with_bnb(adj)
    print(cost)
    print(path)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"

def test_refined_5_city_example():
    adj = [
        [0, 60, 20, 10, 100],
        [60, 0, 15, 100, 20],
        [20, 15, 0, 10, 30],
        [10, 100, 10, 0, 25],
        [100, 20, 30, 25, 0]
    ]
    expected_cost = 90
    expected_path = [0, 2, 1, 4, 3, 0]
    cost, path = solve_tsp_with_bnb(adj)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"

def test_refined_6_city_example():
    adj = [
        [0, 40, 20, 55, 35, 25],
        [40, 0, 30, 25, 60, 75],
        [20, 30, 0, 15, 50, 65],
        [55, 25, 15, 0, 20, 30],
        [35, 60, 50, 20, 0, 10],
        [25, 75, 65, 30, 10, 0]
    ]
    expected_cost = 130
    expected_path = [0, 2, 1, 3, 4, 5, 0]
    cost, path = solve_tsp_with_bnb(adj)
    assert cost == expected_cost, f"Expected cost {expected_cost}, got {cost}"
    assert path == expected_path, f"Expected path {expected_path}, got {path}"
