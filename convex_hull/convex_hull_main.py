from scipy.spatial import ConvexHull
import numpy as np

from convex_hull.plot_utils import plot_hulls_and_connections_plotly
from convex_hull.utils import calculate_total_distance
from generation_utils import generate_points, calculate_distances


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
            # if (tuple(p1_h2), tuple(p2_h2)) in excluded_edges or (tuple(p2_h2), tuple(p1_h2)) in excluded_edges:
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


def convex_hull_solver(points):
    hulls, connections, excluded_edges = connect_hulls_optimized(points)
    total_distance = calculate_total_distance(hulls, connections, excluded_edges)
    return total_distance, hulls, connections, excluded_edges


# Example usage
points = calculate_distances(generate_points(15))

total_distance, hulls, connections, excluded_edges = convex_hull_solver(points)
print(f"Total distance traveled by the path: {total_distance:.2f}")
plot_hulls_and_connections_plotly(points, hulls, connections, excluded_edges)
# plot_hulls_and_connections_seaborn(points, hulls, connections, excluded_edges)
