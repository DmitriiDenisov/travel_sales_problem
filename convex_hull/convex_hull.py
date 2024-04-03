from scipy.spatial import ConvexHull, distance_matrix
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("darkgrid")

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


def plot_hulls_and_connections_seaborn(points, hulls, connections, excluded_edges):
    plt.figure(figsize=(12, 8))

    # Use Seaborn's context setting to make fonts larger.
    sns.set_context('talk')

    # Plotting all points with Seaborn's scatterplot for better styling
    sns.scatterplot(x=points[:, 0], y=points[:, 1], color='blue', s=100, edgecolor='none', alpha=0.7, label='Points')

    for hull_points in hulls:
        hull_points_array = np.array(hull_points)
        hull = ConvexHull(hull_points_array)
        for simplex in hull.simplices:
            start_point = tuple(hull_points_array[simplex[0]].tolist())
            end_point = tuple(hull_points_array[simplex[1]].tolist())

            # Draw excluded edges with a dotted line
            if (start_point, end_point) in excluded_edges or (end_point, start_point) in excluded_edges:
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k:', lw=2, label='Excluded Edges')
            else:
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-', lw=2, label='Hull Edges')

    # Highlight connections between hulls
    for connection in connections:
        plt.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], color='red', linestyle='--', lw=3, label='Optimized Connections')

    # Improve layout with Seaborn's despine to remove the top and right borders for a cleaner look
    sns.despine()

    plt.title("Convex Hulls and Optimized Connections")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")

    # Handling legend - avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_hulls_and_connections_plotly(points, hulls, connections, excluded_edges):
    fig = go.Figure()

    # Define colors
    hull_edge_color = 'Black'  # Hull Edges
    connection_color = 'Red'  # Optimized Connections
    excluded_edge_color = 'Black'  # Excluded Edges, will be dotted
    background_color = 'rgba(240, 240, 240, 1)'  # Light background

    # Add points
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', name='Points', marker=dict(color='RoyalBlue', size=8, opacity=0.8)))


    # Initialize flags to control legend entry
    added_hull_edge_legend = False
    added_connection_legend = False
    added_excluded_edge_legend = False

    # Lift annotations a bit above the line
    annotation_lift = 0.02 * (max(points[:, 1]) - min(points[:, 1]))  # Dynamic lift based on y-axis range


    # Function to calculate distance between points
    def calc_distance(p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    # Process hull edges and excluded edges
    for hull_points in hulls:
        hull = ConvexHull(hull_points)
        for simplex in hull.simplices:
            p1, p2 = hull_points[simplex[0]], hull_points[simplex[1]]
            distance = calc_distance(p1, p2)
            mid_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + annotation_lift]  # Apply lift to y-coordinate


            # 1. EXCLUDED EDGES
            if (tuple(p1), tuple(p2)) in excluded_edges or (tuple(p2), tuple(p1)) in excluded_edges:
                dash_style = 'dot'

                fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines',
                                     line=dict(color=hull_edge_color if dash_style == 'solid' else excluded_edge_color, dash=dash_style),
                                     name='Excluded Edges',
                                     showlegend=not added_excluded_edge_legend))
                added_excluded_edge_legend = True

            # 2. HULL EDGES
            else:
                dash_style = 'solid'
                show_legend = True  # Show legend only for the first hull edge
                # Add annotation for the hull edge distance
                fig.add_annotation(x=mid_point[0], y=mid_point[1], text=f"{distance:.2f}", showarrow=False, font=dict(size=10))

                fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines',
                                        line=dict(color=hull_edge_color if dash_style == 'solid' else excluded_edge_color, dash=dash_style),
                                        name='Hull Edges',
                                        showlegend=not added_hull_edge_legend))
                added_hull_edge_legend = True

    # 3. CONNECTION EDGES
    for connection in connections:
        p1, p2 = connection
        distance = calc_distance(np.array(p1), np.array(p2))
        mid_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        # Add annotation for the connection edge distance
        fig.add_annotation(x=mid_point[0], y=mid_point[1], text=f"{distance:.2f}", showarrow=False, font=dict(size=10, color=connection_color))

        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines',
                                 line=dict(color=connection_color, width=2),
                                 name='Connections edges',
                                 showlegend=not added_connection_legend))
        added_connection_legend = True

    # Set plot layout
    fig.update_layout(
        title='Convex Hulls and Optimized Connections',
        xaxis_title='X-coordinate',
        yaxis_title='Y-coordinate',
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=dict(color='Black'),
        legend=dict(bgcolor='rgba(255, 255, 255, 0.5)')
    )

    fig.show()



def calculate_total_distance(hulls, connections, excluded_edges):
    total_distance = 0

    # Function to calculate distance between points
    def calc_distance(p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

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


# Example usage
points = generate_points(13)
hulls, connections, excluded_edges = connect_hulls_optimized(points)
total_distance = calculate_total_distance(hulls, connections, excluded_edges)

# Assume remaining_points is populated with any points not included in hulls by connect_hulls_optimized
# integrate_remaining_points(hulls, remaining_points)


print(f"Total distance traveled by the path: {total_distance:.2f}")
plot_hulls_and_connections_plotly(points, hulls, connections, excluded_edges)
# plot_hulls_and_connections_seaborn(points, hulls, connections, excluded_edges)
