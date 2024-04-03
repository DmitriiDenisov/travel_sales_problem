from scipy.spatial import ConvexHull, distance_matrix
import numpy as np
import matplotlib.pyplot as plt

def generate_points(n_points=30):
    points = np.random.rand(n_points, 2) * 100
    points = np.round(points, 2)
    return points

def find_nearest_points_between_hulls(hull_1_points, hull_2_points):
    min_distance = np.inf
    best_pair_indices = (0, 0)
    for i, p1 in enumerate(hull_1_points):
        for j, p2 in enumerate(hull_2_points):
            dist = np.linalg.norm(p1 - p2)
            if dist < min_distance:
                min_distance = dist
                best_pair_indices = (i, j)
    return best_pair_indices

def find_optimal_neighbor_pairs(hull_1_points, hull_2_points, excluded_edges):
    min_distance = np.inf
    best_pair = (0, 0, 0, 0)  # Initialize with dummy indices

    for i in range(len(hull_1_points)):
        p1_h1 = hull_1_points[i]
        p2_h1 = hull_1_points[(i + 1) % len(hull_1_points)]

        # Skip this pair if it's in the excluded edges
        if (tuple(p1_h1), tuple(p2_h1)) in excluded_edges or (tuple(p2_h1), tuple(p1_h1)) in excluded_edges:
            continue

        for j in range(len(hull_2_points)):
            p1_h2 = hull_2_points[j]
            p2_h2 = hull_2_points[(j + 1) % len(hull_2_points)]

            # Similarly, skip if this pair is in the excluded edges
            #if (tuple(p1_h2), tuple(p2_h2)) in excluded_edges or (tuple(p2_h2), tuple(p1_h2)) in excluded_edges:
            #    continue

            # Calculate distance between pairs and update the best pair if necessary
            dist = np.linalg.norm(p1_h1 - p1_h2) + np.linalg.norm(p2_h1 - p2_h2)
            if dist < min_distance:
                min_distance = dist
                best_pair = (i, (i + 1) % len(hull_1_points), j, (j + 1) % len(hull_2_points))

    return best_pair

def connect_hulls_optimized(points):
    remaining_points = points.copy()
    hulls = []
    all_connections = []
    excluded_edges = []  # Track edges to be excluded as tuples of scalars
    used_points = set() # to avoid situation when same points are used to connect outside hull and internal hull


    while len(remaining_points) > 2:
        hull = ConvexHull(remaining_points)
        hulls.append(remaining_points[hull.vertices])
        remaining_points = np.delete(remaining_points, hull.vertices, axis=0)

    for i in range(len(hulls) - 1):
        best_pair_indices = find_optimal_neighbor_pairs(hulls[i], hulls[i + 1], excluded_edges)

        # Ensuring we're working with scalars for excluded_edges
        p1_h1, p2_h1 = tuple(hulls[i][best_pair_indices[0]]), tuple(hulls[i][best_pair_indices[1]])
        p1_h2, p2_h2 = tuple(hulls[i + 1][best_pair_indices[2]]), tuple(hulls[i + 1][best_pair_indices[3]])

        # Add connections
        all_connections.append((p1_h1, p1_h2))
        all_connections.append((p2_h1, p2_h2))

        # Track excluded edges as tuples of scalars
        excluded_edges.append((p1_h1, p2_h1))
        excluded_edges.append((p1_h2, p2_h2))

    if len(remaining_points) > 0:
        # Check if we have an existing hull to connect these points to
        if len(hulls) > 0:
            # Treat the remaining points as a separate "hull" for connection purposes
            remaining_hull = np.array(remaining_points)

            # Find the optimal connection between this "hull" and the last actual hull
            indecies = find_optimal_neighbor_pairs(hulls[-1], remaining_hull, excluded_edges)
            indices_in_last_hull, indices_in_remaining_hull = (indecies[0], indecies[1]), (indecies[2], indecies[3])

            # Identify the points in the last hull and remaining points based on found indices
            p1_h1, p2_h1 = hulls[-1][indices_in_last_hull[0]], hulls[-1][indices_in_last_hull[1]]
            p1_h2, p2_h2 = remaining_hull[indices_in_remaining_hull[0]], remaining_hull[indices_in_remaining_hull[1]]

            # Add connections and update excluded_edges based on these optimal pairs
            all_connections.append((tuple(p1_h1), tuple(p1_h2)))
            all_connections.append((tuple(p2_h1), tuple(p2_h2)))
            # Add connection between the left two points
            if len(remaining_points) > 1:
                all_connections.append((tuple(remaining_points[0]), tuple(remaining_points[1])))
            excluded_edges.append((tuple(p1_h1), tuple(p2_h1)))

    return hulls, all_connections, excluded_edges


def plot_hulls_and_connections(points, hulls, connections, excluded_edges):
    plt.figure(figsize=(12, 8))  # Adjust figure size for better visualization

    # Plot all points
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=8, label='Points')

    for hull_points in hulls:
        hull_points_array = np.array(hull_points)
        hull = ConvexHull(hull_points_array)
        for simplex in hull.simplices:
            start_point = tuple(hull_points_array[simplex[0]].tolist())
            end_point = tuple(hull_points_array[simplex[1]].tolist())

            # Check if the edge should be excluded and render accordingly


            if (start_point, end_point) in excluded_edges or (end_point, start_point) in excluded_edges:
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k:', lw=1, label='Excluded Edges')
            else:
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-', lw=1, label='Hull Edges')

    # Highlight connections between hulls
    for connection in connections:
        plt.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], 'r--', lw=2, label='Optimized Connections')

    # Improve layout
    plt.title("Convex Hulls and Optimized Connections")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)
    plt.axis('equal')  # Ensure equal aspect ratio for X and Y axes to maintain scale

    # Handling legend - avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')  # Place legend in upper right corner

    plt.tight_layout()  # Adjust layout to ensure everything fits without overlap
    plt.show()


def calculate_total_distance(hulls, connections):
    total_distance = 0

    # Calculate distance within each hull
    for hull_points in hulls:
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]  # Wrap around to the first point
            total_distance += np.linalg.norm(np.array(p1) - np.array(p2))

    # Calculate distance for the optimized connections between hulls
    for connection in connections:
        total_distance += np.linalg.norm(np.array(connection[0]) - np.array(connection[1]))

    return total_distance

# Example usage
points = np.array([[50.11, 58.12],
       [ 5.5 , 35.19],
       [76.74, 80.09],
       [48.32, 68.08],
       [53.15, 67.21],
       [38.38, 35.  ],
       [16.37, 85.81],
       [13.2 , 42.74],
       [42.11, 56.2 ],
       [32.55, 53.97]])# generate_points(10)
hulls, connections, excluded_edges = connect_hulls_optimized(points)
total_distance = calculate_total_distance(hulls, connections)

# Assume remaining_points is populated with any points not included in hulls by connect_hulls_optimized
# integrate_remaining_points(hulls, remaining_points)


print(f"Total distance traveled by the path: {total_distance:.2f}")
plot_hulls_and_connections(points, hulls, connections, excluded_edges)
