import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
from scipy.spatial import ConvexHull

sns.set_style("darkgrid")


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
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k:', lw=2,
                         label='Excluded Edges')
            else:
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-', lw=2, label='Hull Edges')

    # Highlight connections between hulls
    for connection in connections:
        plt.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], color='red',
                 linestyle='--', lw=3, label='Optimized Connections')

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
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', name='Points',
                             marker=dict(color='RoyalBlue', size=8, opacity=0.8)))

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
                                         line=dict(
                                             color=hull_edge_color if dash_style == 'solid' else excluded_edge_color,
                                             dash=dash_style),
                                         name='Excluded Edges',
                                         showlegend=not added_excluded_edge_legend))
                added_excluded_edge_legend = True

            # 2. HULL EDGES
            else:
                dash_style = 'solid'
                show_legend = True  # Show legend only for the first hull edge
                # Add annotation for the hull edge distance
                fig.add_annotation(x=mid_point[0], y=mid_point[1], text=f"{distance:.2f}", showarrow=False,
                                   font=dict(size=10))

                fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines',
                                         line=dict(
                                             color=hull_edge_color if dash_style == 'solid' else excluded_edge_color,
                                             dash=dash_style),
                                         name='Hull Edges',
                                         showlegend=not added_hull_edge_legend))
                added_hull_edge_legend = True

    # 3. CONNECTION EDGES
    for connection in connections:
        p1, p2 = connection
        distance = calc_distance(np.array(p1), np.array(p2))
        mid_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        # Add annotation for the connection edge distance
        fig.add_annotation(x=mid_point[0], y=mid_point[1], text=f"{distance:.2f}", showarrow=False,
                           font=dict(size=10, color=connection_color))

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
