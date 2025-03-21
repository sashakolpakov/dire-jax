# dire_utils.py

"""
Utilities for plotting, benchmarking, and miscellany
"""

#
# Imports
#

from collections import defaultdict
from time import perf_counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.express as px
from jax import random

from .hpmetrics import (compute_local_metrics,
                        compute_global_metrics,
                        compute_context_measures,
                        compute_quality_measures)

#
# Auxiliary functions for plotting, benchmarking, and miscellany
# that are not included in other packages and classes
#


#
# Display a given layout with labels and with given point size
#
def display_layout(layout, labels, point_size=2):
    """
    Parameters
    ----------
    layout: (numpy.ndarray) Layout to display, must have dimension 2 or 3.
    labels: (numpy.ndarray) Labels to generate color and legend.
    point_size: (int) Point size for plotting.

    Returns
    -------
    object:
        Plot of the layout if the latter has dimension 2 or 3 (using Plotly Express).
        For higher-dimensional data no plot is provided, and the function returns `None`.
    """
    #
    dimension = layout.shape[1]
    #
    if dimension == 2:

        data_df = pd.DataFrame(layout, columns=['x', 'y'])

        if labels is not None:
            data_df['label'] = labels
            figure = px.scatter(data_df, x='x', y='y', color='label',
                                title=f'Layout in dimension {dimension}')
        else:
            figure = px.scatter(data_df, x='x', y='y',
                                title=f'Layout in dimension {dimension}')

        figure.update_traces(marker={"size": point_size})
        return figure
    #
    if dimension == 3:

        data_df = pd.DataFrame(layout, columns=['x', 'y', 'z'])

        if labels is not None:
            data_df['label'] = labels
            figure = px.scatter_3d(data_df, x='x', y='y', z='z', color='label',
                                   title=f'Layout in dimension {dimension}')
        else:
            figure = px.scatter_3d(data_df, x='x', y='y', z='z',
                                   title=f'Layout in dimension {dimension}')

        figure.update_traces(marker={"size": point_size})
        return figure
    #
    return None


#
# Print out local metrics for a given high-dimensional data, low-dimensional layout pair
#
def do_local_analysis(data, layout, n_neighbors):
    """
    Parameters
    ----------
    data: (numpy.ndarray) High-dimensional data.
    layout: (numpy.ndarray) Low-dimensional embedding.
    n_neighbors: (int) Number of neighbors in the kNN graph.

    Returns
    -------
    None: Prints out the local metrics (embedding stress, neighborhood preservation score, etc.)
    """

    # Compute local metrics
    print("Computing kNN graphs and local metrics")
    local_metrics = compute_local_metrics(data, layout, n_neighbors)
    print("done")

    # Printing local metrics
    print(f"Embedding stress (scaling adjusted): {local_metrics['stress']}")
    print(f"Neighborhood preservation score (mean, std): {local_metrics['neighbor']}")


#
# Create a plotly visualization of persistence diagrams
#
def visualize_persistence_diagram(diagram, dimension, title):
    """
    Create a visualization of a persistence diagram.

    Parameters
    ----------
    diagram : list of tuples
        Persistence diagram represented as a list of (birth, death) tuples
    dimension : int
        Homology dimension of the diagram
    title : str
        Title for the visualization

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive visualization of the persistence diagram
    """
    # Filter out points with infinite death time for visualization
    filtered_diagram = [(b, d) for b, d in diagram if not np.isinf(d)]

    if not filtered_diagram:  # Handle empty diagrams
        return None

    # Create dataframe for plotting
    df = pd.DataFrame(filtered_diagram, columns=['birth', 'death'])

    # Calculate the persistence (death - birth) for each point
    df['persistence'] = df['death'] - df['birth']

    # Calculate the diagonal line range for reference
    max_value = max(df['death'].max(), df['birth'].max()) * 1.05
    diagonal = pd.DataFrame({
        'x': [0, max_value],
        'y': [0, max_value]
    })

    # Create the scatter plot
    fig = px.scatter(
        df, 
        x='birth', 
        y='death', 
        size='persistence',
        color='persistence',
        color_continuous_scale='Viridis',
        opacity=0.8,
        labels={
            'birth': 'Birth',
            'death': 'Death',
            'persistence': 'Persistence'
        },
        title=f"{title} (Dimension {dimension})"
    )

    # Add the diagonal line
    fig.add_scatter(
        x=diagonal['x'],
        y=diagonal['y'],
        mode='lines',
        line={"dash": 'dash', "color": 'gray'},
        name='Diagonal',
        showlegend=True
    )

    # Configure layout
    fig.update_layout(
        xaxis_title='Birth',
        yaxis_title='Death',
        xaxis={"range": [0, max_value]},
        yaxis={"range": [0, max_value]},
        height=600,
        width=600
    )

    return fig


#
# Do persistence analysis: compute persistence diagrams and Betti curves, and find various distances between them.
# Visualise the Betti curves and persistence diagrams.
#
def do_persistence_analysis(data, layout, dimension, subsample_threshold, rng_key, n_steps=100):
    """
    Perform a comprehensive persistence analysis by subsampling data, computing persistence diagrams, and
    calculating distances between Betti curves of high-dimensional and low-dimensional data. This analysis
    includes computing distances such as Dynamic Time Warp (DTW), Time Warp Edit Distance (TWED), and
    Earth Mover Distance.

    Parameters
    ----------
    data : numpy.ndarray
        High-dimensional data.
    layout : numpy.ndarray
        Low-dimensional embedding.
    dimension : int
        The dimension up to which persistence diagrams are computed.
    subsample_threshold : float
        The threshold used for subsampling the data points.
    rng_key : jax.random.PRNGKey
        Random key used for generating random numbers for subsampling, ensuring reproducibility.
    n_steps : int, optional
        The number of steps or points in the filtration range for computing Betti curves, default 100.

    Returns
    -------
    None
        This function primarily visualizes results and prints metric values.
    """
    print(f"Computing persistence up to dimension {dimension} and global metrics")
    gdict = compute_global_metrics(data,
                                   layout,
                                   dimension,
                                   subsample_threshold,
                                   rng_key,
                                   n_steps=n_steps,
                                   metrics_only=False)
    print("done")

    global_metrics = gdict['metrics']
    diags = gdict['diags']  # Persistence diagrams
    bettis = gdict['bettis'] # Betti curves

    for dim in range(dimension + 1):
        # Visualize persistence diagrams
        hd_diagram_fig = visualize_persistence_diagram(
            diags['data'][dim], 
            dim, 
            "High-dimensional Data Persistence Diagram"
        )
        ld_diagram_fig = visualize_persistence_diagram(
            diags['layout'][dim], 
            dim, 
            "Low-dimensional Layout Persistence Diagram"
        )

        if hd_diagram_fig is not None:
            hd_diagram_fig.show()
        if ld_diagram_fig is not None:
            ld_diagram_fig.show()

        # Extracting and visualizing Betti curves
        axis_x_hd, axis_y_hd = bettis['data'][dim]
        axis_x_ld, axis_y_ld = bettis['layout'][dim]

        # Normalising Betti curves for plotting
        # This ensures comparable scales regardless of filtration range
        axis_x_hd_, axis_y_hd_ = axis_x_hd / np.max(axis_x_hd), axis_y_hd
        axis_x_ld_, axis_y_ld_ = axis_x_ld / np.max(axis_x_ld), axis_y_ld

        # Plotting normalized Betti curves
        df_hd = pd.DataFrame({
            'x': axis_x_hd_,
            'y': axis_y_hd_,
            'Type': f'Betti {dim} High-dimensional'
        })
        df_ld = pd.DataFrame({
            'x': axis_x_ld_,
            'y': axis_y_ld_,
            'Type': f'Betti {dim} Low-dimensional'
        })
        df = pd.concat([df_hd, df_ld])

        figure = px.line(
            df, 
            x='x', 
            y='y', 
            color='Type', 
            labels={'x': 'Filtration Level', 'y': 'Rank'},
            title=f'Betti {dim} High-dimensional vs Low-dimensional'
        )
        figure.show()

        # Printing global metrics
        print(f"Distance `DTW` for dimension {dim}: {global_metrics['dtw'][dim]}")
        print(f"Distance `TWED` for dimension {dim}: {global_metrics['twed'][dim]}")
        print(f"Distance `EMD` for dimension {dim}: {global_metrics['emd'][dim]}")
        print(f"Distance `Wasserstein` for dimension {dim}: {global_metrics['wass'][dim]}")
        print(f"Distance `bottleneck` for dimension {dim}: {global_metrics['bott'][dim]}")


#
# Do quality and context analysis
#
def do_context_analysis(data, layout, labels, subsample_threshold, n_neighbors, rng_key, **kwargs):
    """
    Parameters
    ----------
    data: (numpy.ndarray) High-dimensional data.
    layout: (numpy.ndarray) Low-dimensional embedding.
    labels: (numpy.ndarray) Data labels, required for context preservation analysis.
    subsample_threshold: (float) Subsample thresholds.
    n_neighbors: (int) Number of nearest neighbors for the kNN graph of data.
    rng_key: Random key used for generating random numbers for subsampling, ensuring reproducibility.
    kwargs: Keyword arguments for kNN and SVM score procedure, and similar.

    Returns
    -------
    None: This function prints out context preservation measures.
    """

    assert labels is not None, "Context analysis needs labelled data"

    print('Computing context measures')
    context_measures = compute_context_measures(data,
                                                layout,
                                                labels,
                                                subsample_threshold,
                                                n_neighbors,
                                                rng_key,
                                                **kwargs)

    print("done")

    print(f"Context preservation score (SVM): {context_measures['svm']}")
    print(f"Context preservation score (kNN): {context_measures['knn']}")


#
# Timing a given block of code; adding the measurement to a given dictionary of metrics
#
@contextmanager
def block_timing():
    """
    Returns
    -------
    float: elapsed runtime (in seconds) for a given block of code
    """
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()


#
# Visualising a single benchmark on a given reducer with (data, labels)
#
def viz_benchmark(reducer, data, **kwargs):
    """
    Run a benchmarking process for dimensionality reduction using provided reducer.

    Parameters
    ----------
    reducer: (object) Dimensionality reduction model with a `fit_transform` method.
                      It should also have an `n_neighbors` attribute for computing neighborhood scores.
    data: (numpy.ndarray) High-dimensional data to be reduced.
    kwargs: Keyword arguments for benchmark's metrics (such as `labels` if using labeled data, maximum `dimension` for
            persistence homology computation, threshold `subsample_threshold` for subsampling, etc.)

    Returns
    -------
    `None`: This function does not return anything. It performs the embedding, displays the layout,
            conducts persistence analysis, prints the embedding stress and neighborhood preservation score,
            and times the embedding process.
    """
    with block_timing() as bt:
        layout = reducer.fit_transform(data)

    print(f'Embedding time: {bt():.4f} seconds')

    labels = kwargs.pop('labels', None)
    dimension = kwargs.pop('dimension', 1)
    subsample_threshold = kwargs.pop('subsample_threshold', 0.1)
    rng_key = kwargs.pop('rng_key', random.PRNGKey(42))
    point_size = kwargs.pop('point_size', 4)
    n_neighbors = kwargs.pop('n_neighbors', reducer.n_neighbors)

    display_layout(layout,
                   labels=labels.astype('str') if labels is not None else None,
                   point_size=point_size).show()

    do_persistence_analysis(data,
                            layout,
                            dimension=dimension,
                            subsample_threshold=subsample_threshold,
                            rng_key=rng_key,
                            )

    do_local_analysis(data,
                      layout,
                      n_neighbors=n_neighbors,
                      )

    if labels is not None:
        do_context_analysis(data,
                            layout,
                            labels=labels,
                            subsample_threshold=subsample_threshold,
                            n_neighbors=n_neighbors,
                            rng_key=rng_key,
                            )
    else:
        print("Data has no labels: no context analysis performed")


#
# Computing local and global metrics, and context preservation measures.
#
def do_metrics(reducer, data, **kwargs):
    """
    Compute local and global metrics, and context preservation measures.

    Parameters
    ----------
    reducer: (object) The dimensionality reduction model with a `fit_transform` method.
                      It should also have an `n_neighbors` attribute for computing neighborhood scores.
    data: (numpy.ndarray) The high-dimensional data to be reduced.
    kwargs: Keyword arguments to be passed to `compute_local_metrics`, `compute_global_metrics`,
            and `compute_context_measures`.

    Returns
    -------
    dict: A dictionary of local and global metrics, and context preservation measures.
    """

    mdict = {}

    with block_timing() as bt:
        layout = reducer.fit_transform(data)
    mdict['timing'] = bt()

    n_neighbors = kwargs.pop('n_neighbors', reducer.n_neighbors)
    ldict = compute_local_metrics(data, layout, n_neighbors)
    mdict.update(ldict)

    dimension = kwargs.pop('dimension', 1)
    subsample_threshold = kwargs.pop('subsample_threshold', 0.1)
    rng_key = kwargs.pop('rng_key', random.PRNGKey(42))
    n_steps = kwargs.pop('n_steps', 100)
    gdict = compute_global_metrics(data,
                                   layout,
                                   dimension,
                                   subsample_threshold,
                                   rng_key,
                                   n_steps,
                                   metrics_only=True)
    mdict.update(gdict['metrics'])

    # Add quality measures for the embedding
    qdict = compute_quality_measures(data, layout)
    mdict.update(qdict)

    # Add context measures if labels are provided
    labels = kwargs.pop('labels', None)
    if labels is not None:
        cdict = compute_context_measures(data,
                                         layout,
                                         labels=labels,
                                         subsample_threshold=subsample_threshold,
                                         n_neighbors=n_neighbors,
                                         rng_key=rng_key,
                                         **kwargs)
        mdict.update(cdict)

    return mdict


#
# Benchmark a reducer on given data
#
def run_benchmark(reducer, data, *, num_trials=100, only_stats=True, **kwargs):
    """
    Benchmark a reducer on given data.

    Parameters
    ----------
    reducer: (object) The dimensionality reduction model with a `fit_transform` method.
                      It should also have an `n_neighbors` attribute for computing neighborhood scores.
    data: (numpy.ndarray) The high-dimensional data to be reduced as a benchmark.
    num_trials: (int) The number of runs to collect stats.
    only_stats: (bool) If True, only the tuple (mean, std) for each metrics is returned.
                       If False, both stats and values for all runs are returned.
    kwargs: Keyword arguments to be passed to do_metrics.

    Returns
    -------
    dict: If only_stats is True, a dictionary with stats of all metrics available.
    dict, dict: If only_stats is False, a dictionary with stats and a dictionary with the initial values of all metrics.
    """

    benchmark_vals = defaultdict(list)
    benchmark_stat = {}

    rng_key = kwargs.pop('rng_key', random.PRNGKey(42))
    keys = random.split(rng_key, num_trials)

    for key in keys:

        mdict = do_metrics(reducer, data, rng_key=key, **kwargs)
        for name, val in mdict.items():
            benchmark_vals[name].append(val)

    for name, vals in benchmark_vals.items():
        mean_vals = np.mean(vals, axis=0)
        std_vals = np.std(vals, axis=0)
        benchmark_stat[name] = [mean_vals, std_vals]

    if only_stats:
        return benchmark_stat
    return benchmark_stat, benchmark_vals
