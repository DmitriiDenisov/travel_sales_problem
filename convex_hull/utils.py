import numpy as np
from scipy.spatial import ConvexHull


# Function to calculate distance between points
def calc_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def calculate_total_distance(hulls, connections, excluded_edges):
    total_distance = 0

    # Calculate distance within each hull, excluding the excluded edges
    for hull_points in hulls:
        hull = ConvexHull(hull_points)
        for simplex in hull.simplices:
            p1, p2 = hull_points[simplex[0]], hull_points[simplex[1]]
            if (tuple(p1), tuple(p2)) not in excluded_edges and (tuple(p2), tuple(p1)) not in excluded_edges:
                # Add distance if edge is not excluded
                total_distance += calc_distance(p1, p2)

    # Calculate distance for the optimized connections between hulls
    for connection in connections:
        p1, p2 = connection
        # Connections are not subjected to exclusion, so add directly
        total_distance += calc_distance(p1, p2)

    return total_distance

def generate_points(n_points=30):
    points = np.random.rand(n_points, 2) * 100
    points = np.round(points, 2)
    return points
