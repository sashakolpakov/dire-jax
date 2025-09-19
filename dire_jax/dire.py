# dire.py

"""
Provides the main JAX-based class for dimensionality reduction.

The DiRe (Dimensionality Reduction) class implements a modern approach to
dimensionality reduction, leveraging JAX for efficient computation. It uses 
force-directed layout techniques combined with k-nearest neighbor graph 
construction to generate meaningful low-dimensional embeddings of 
high-dimensional data.

This JAX implementation features:
- Fully vectorized force computation (no chunking for optimal performance)
- JIT compilation for mathematical operations
- Optimized for small to medium datasets (<50K points)
- Excellent CPU performance and research-friendly design
"""

#
# Imports
#

import functools
import gc
import os
import sys
from random import randint

# JAX-related imports
import jax
import jax.numpy as jnp
# Scientific and numerical libraries
import numpy as np
# Data structures and visualization
import pandas as pd
import plotly.express as px
from jax import device_get, device_put, jit, lax, random, vmap
from loguru import logger
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
# Utilities
from tqdm import tqdm

# Nearest neighbor search
from .hpindex import HPIndex

#
# Double precision support
#
jax.config.update("jax_enable_x64", True)


#
# Main class for Dimensionality Reduction
#
class DiRe(TransformerMixin):
    """
    Dimension Reduction (DiRe) is a class designed to reduce the dimensionality of high-dimensional
    data using various embedding techniques and optimization algorithms. It supports embedding
    initialization methods such as random and spectral embeddings and utilizes k-nearest neighbors
    for manifold learning.

    Parameters
    ----------
    n_components: (int) Embedding dimension, default 2.
    n_neighbors: (int) Number of nearest neighbors to consider for each point, default 16.
    init: (str)
        Method to initialize the embedding; choices are:
         - 'random' for random projection based on the Johnson-Lindenstrauss lemma;
         - 'spectral' for spectral embedding (with sim_kernel as similarity kernel);
         - 'pca' for PCA embedding (classical, no kernel).

         By default, 'random'.
    sim_kernel: (callable)
        A similarity kernel function that transforms a distance metric to a similarity score.
        The function should have the form `lambda distance: float -> similarity: float`; default `None`.
    max_iter_layout: (int)
        Maximum number of iterations to run the layout optimization, default 128.
    min_dist: (float)
        Minimum distance scale for distribution kernel, default 0.01.
    spread: (float)
        Spread of the distribution kernel, default 1.0.
    cutoff: (float)
        Cutoff for clipping forces during layout optimization, default 42.0.
    n_sample_dirs: (int)
        Number of directions to sample in random sampling, default 8.
    sample_size: (int)
        Number of samples per direction in random sampling, default 16.
    neg_ratio: (int)
        Ratio of negative to positive samples in random sampling, default 8.
    my_logger: (logger.Logger or `None`)
        Custom logger for logging events; if None, a default logger is created, default `None`.
    verbose: (bool)
        Flag to enable verbose output, default `True`.
    random_state: (int)
        Random seed to make stochastic computations reproducible.

    Attributes
    ----------
    n_components: int
        Target dimensionality of the output space.
    n_neighbors: int
        Number of neighbors to consider in the k-nearest neighbors graph.
    init: str
        Chosen method for initial embedding.
    sim_kernel: callable
        Similarity kernel function to be used if 'init' is 'spectral', by default `None`.
    pca_kernel: callable
        Kernel function to be used if 'init' is 'pca', by default `None`.
    max_iter_layout: int
        Maximum iterations for optimizing the layout.
    min_dist: float
        Minimum distance for repulsion used in the distribution kernel.
    spread: float
        Spread between the data points used in the distribution kernel.
    cutoff: float
        Maximum cutoff for forces during optimization.
    n_sample_dirs: int
        Number of random directions sampled.
    sample_size: int or 'auto'
        Number of samples per random direction, unless chosen automatically with 'auto'.
    batch_size : int or None, optional
        Number of samples to process at once. If None, a suitable value
        will be automatically determined based on dataset size.
    neg_ratio: int
        Ratio of negative to positive samples in the sampling process.
    logger: logger.Logger or `None`
        Logger used for logging informational and warning messages.
    verbose: bool
        Logger output flag (True = output logger messages, False = flush to null)
    memm: dictionary or `None`
        Memory manager: a dictionary with the batch / memory tile size for different
        hardware architectures. Accepts 'tpu', 'gpu' and 'other' as keys. Values must
        be positive integers.
    mpa: bool
        Mixed Precision Arithmetic flag (True = use MPA, False = always use float64)

    Methods
    -------
    fit_transform(data)
        A convenience method that fits the model and then transforms the data.
        The separate `fit` and `transform` methods can only be used one after
        another because dimension reduction is applied to the dataset as a whole.
    visualize(labels=None, point_size=2)
        Visualizes the transformed data, optionally using labels to color the points.
    """

    def __init__(
        self,
        n_components=2,
        n_neighbors=16,
        init="random",
        sim_kernel=None,
        pca_kernel=None,
        max_iter_layout=128,
        min_dist=1e-2,
        spread=1.0,
        cutoff=42.0,
        n_sample_dirs=8,
        sample_size=16,
        batch_size=None,
        neg_ratio=8,
        my_logger=None,
        verbose=True,
        memm=None,
        mpa=True,
        random_state=None,
    ):
        """
        Class constructor
        """

        #
        self.n_components = n_components
        """ Embedding dimension """
        self.n_neighbors = n_neighbors
        """ Number of neighbors for kNN computations"""
        self.init = init
        """ Type of the initial embedding (PCA, random, spectral) """
        self.sim_kernel = sim_kernel
        """ Similarity kernel """
        self.pca_kernel = pca_kernel
        """ PCA kernel """
        self.max_iter_layout = max_iter_layout
        """ Max iterations for the force layout """
        self.min_dist = min_dist
        """ Min distance between points in layout """
        self.spread = spread
        """ Layout spread """
        self.cutoff = cutoff
        """ Cutoff for layout displacement """
        self.n_sample_dirs = n_sample_dirs
        """ Number of sampling directions for layout"""
        self.sample_size = sample_size
        """ Sample size for attraction """
        self.neg_ratio = neg_ratio
        """ Ratio for repulsion sample size """
        self._init_embedding = None
        """ Initial embedding """
        self._layout = None
        """ Layout output """
        self._a = None
        """ Probability kernel parameter """
        self._b = None
        """ Probability kernel parameter """
        self._data = None
        """ Higher-dimensional data """
        self._n_samples = None
        """ Number of data points """
        self._data_dim = None
        """ Dimension of data """
        self._distances_np = None
        self._distances_jax = None
        """ Distances in the kNN graph """
        self._indices_np = None
        self._indices_jax = None
        """ Neighbor indices in the kNN graph """
        self._nearest_neighbor_distances = None
        """ Neighbor indices in the kNN graph, excluding the point itself """
        self._row_idx = None
        """ Row indices for nearest neighbors """
        self._col_idx = None
        """ Column indices for nearest neighbors """
        self._adjacency = None
        """ kNN adjacency matrix """
        self.random_state = random_state
        self.batch_size = batch_size
        #
        if my_logger is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, "w", encoding="utf-8")
            logger.add(sink, level="INFO")
            self.logger = logger
            """ System logger """
        else:
            self.logger = my_logger
        # Memory manager to be adjusted for each particular type of hardware
        # Below are some minimalist settings that may give less than satisfactory performance
        self.memm = {"gpu": 16384, "tpu": 8192, "other": 8192} if memm is None else memm
        # Using Mixed Precision Arithmetic flag (True = MPA, False = no MPA)
        self.mpa = mpa

    #
    # Fitting the distribution kernel with given min_dist and spread
    #

    def find_ab_params(self, min_dist=0.01, spread=1.0):
        """
        Rational function approximation to the probabilistic t-kernel
        """

        #
        self.logger.info("find_ab_params ...")

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        #
        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, _ = curve_fit(curve, xv, yv)
        #
        self.logger.info(f"a = {params[0]}, b = {params[1]}")
        self.logger.info("find_ab_params done ...")
        #
        return params[0], params[1]

    #
    # Fitting on data
    #

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the model to data: create the kNN graph and fit the probability kernel to force layout parameters.

        Parameters
        ----------
        X: (numpy.ndarray)
            High-dimensional data to fit the model. Shape (n_samples, n_features).
        y: None
            Ignored, exists for sklearn compatibility.

        Returns
        -------
        self: The DiRe instance fitted to data.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit the model to data and transform it into a low-dimensional layout.

        This is a convenience method that combines the fitting and transformation
        steps. It first builds the kNN graph and then creates the optimized
        layout in a single operation.

        Parameters
        ----------
        X: numpy.ndarray
            High-dimensional data to fit and transform.
            Shape (n_samples, n_features)
        y: None
            Ignored, exists for sklearn API compatibility.

        Returns
        -------
        numpy.ndarray
            The lower-dimensional embedding of the data.
            Shape (n_samples, dimension)

        Notes
        -----
        This method is more memory-efficient than calling fit() and transform()
        separately, as it avoids storing intermediate results.
        """
        #
        self.logger.info("fit ...")
        #
        self._data = X

        self._n_samples = self._data.shape[0]
        self._data_dim = self._data.shape[1]

        self.logger.info(
            f"Dimension {self._data_dim}, number of samples {self._n_samples}"
        )

        self.make_knn_adjacency(batch_size=self.batch_size)

        self._a, self._b = self.find_ab_params(self.min_dist, self.spread)
        #
        self.logger.info("fit done ...")
        #
        # Store the data and perform fitting (build kNN graph)
        self._data = X
        self.logger.info("transform ...")

        # Create initial embedding based on specified initialization method
        if self.init == "random":
            self.do_random_embedding()
        elif self.init == "spectral":
            self.do_spectral_embedding()
        elif self.init == "pca":
            self.do_pca_embedding()
        else:
            error_msg = f'Unsupported embedding method: "{self.init}"'
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Optimize the layout using force-directed placement
        self.do_layout()

        self.logger.info("transform done ...")

        return self._layout

    #
    # Computing the kNN adjacency matrix (sparse)
    #

    def make_knn_adjacency(self, batch_size=None):
        """
        Internal routine building the adjacency matrix for the kNN graph.

        This method computes the k-nearest neighbors for each point in the dataset
        and constructs a sparse adjacency matrix representing the kNN graph.
        It attempts to use GPU acceleration if available, with a fallback to CPU.

        For large datasets, it uses batching to limit memory usage.

        Parameters
        ----------
        batch_size : int or None, optional
            Number of samples to process at once. If None, a suitable value
            will be automatically determined based on dataset size.

        The method sets the following instance attributes:
        - distances: Distances to the k nearest neighbors (including self)
        - indices: Indices of the k nearest neighbors (including self)
        - nearest_neighbor_distances: Distances to the nearest neighbors (excluding self)
        - row_idx, col_idx: Indices for constructing the sparse adjacency matrix
        - adjacency: Sparse adjacency matrix of the kNN graph
        """
        self.logger.info("make_knn_adjacency ...")

        # Ensure data is in the right format for HPIndex
        self._data = np.ascontiguousarray(self._data.astype(np.float64))
        n_neighbors = self.n_neighbors + 1  # Including the point itself

        # Determine appropriate batch size for memory efficiency
        if batch_size is None:
            # Process in chunks to reduce peak memory usage
            if jax.devices()[0].platform == "tpu":
                batch_size = min(self.memm["tpu"], self._n_samples)
            elif jax.devices()[0].platform == "gpu":
                batch_size = min(self.memm["gpu"], self._n_samples)
            else:
                batch_size = min(self.memm["other"], self._n_samples)

        self.logger.info(f"Using batch size: {batch_size}")
        self.logger.debug(
            f"[KNN] Using precision: {'float32' if self.mpa else 'float64'}"
        )

        if self.mpa:
            self._indices_jax, self._distances_jax = HPIndex.knn_tiled(
                self._data,
                self._data,
                n_neighbors,
                batch_size,
                batch_size,
                dtype=jnp.float32,
            )
        else:
            self._indices_jax, self._distances_jax = HPIndex.knn_tiled(
                self._data,
                self._data,
                n_neighbors,
                batch_size,
                batch_size,
                dtype=jnp.float64,
            )

        # Wait until ready
        self._indices_jax.block_until_ready()
        self._distances_jax.block_until_ready()

        # Store results in numpy
        self._indices_np = device_get(self._indices_jax).astype(np.int64)
        self._distances_np = device_get(self._distances_jax).astype(np.float64)

        # Extract nearest neighbor distances (excluding self)
        self._nearest_neighbor_distances = self._distances_np[:, 1:]

        # Create indices for sparse matrix construction
        self._row_idx = np.repeat(np.arange(self._n_samples), n_neighbors)
        self._col_idx = self._indices_np.ravel()

        # Create sparse adjacency matrix (memory efficient)
        data_values = self._distances_np.ravel()
        self._adjacency = csr_matrix(
            (data_values, (self._row_idx, self._col_idx)),
            shape=(self._n_samples, self._n_samples),
        )

        # Clean up resources
        gc.collect()

        self.logger.info("make_knn_adjacency done ...")

    #
    # Initialize embedding using different techniques
    #

    def do_pca_embedding(self):
        """
        Initialize embedding using Principal Component Analysis (PCA).

        This method creates an initial embedding of the data using PCA, which finds
        a linear projection of the high-dimensional data into a lower-dimensional space
        that maximizes the variance. If a kernel is specified, Kernel PCA is used instead,
        which can capture nonlinear relationships.

        Sets the init_embedding attribute with the PCA projection of the data.
        """
        self.logger.info("do_pca_embedding ...")

        if self.pca_kernel is not None:
            # Use Kernel PCA for nonlinear dimensionality reduction
            self.logger.info("Using kernelized PCA embedding...")
            pca = KernelPCA(n_components=self.n_components, kernel=self.pca_kernel)
            self._init_embedding = pca.fit_transform(self._data)
        else:
            # Use standard PCA for linear dimensionality reduction
            self.logger.info("Using standard PCA embedding...")
            pca = PCA(n_components=self.n_components)
            self._init_embedding = pca.fit_transform(self._data)

        self.logger.info("do_pca_embedding done ...")

    def do_spectral_embedding(self):
        """
        Initialize embedding using Spectral Embedding.

        This method creates an initial embedding of the data using spectral embedding,
        which is based on the eigenvectors of the graph Laplacian. It relies on the
        kNN graph structure to find a lower-dimensional representation that preserves
        local relationships.

        If a similarity kernel is specified, it is applied to transform the distances
        in the adjacency matrix before computing the Laplacian.

        Sets the init_embedding attribute with the spectral embedding of the data.
        """
        self.logger.info("do_spectral_embedding ...")

        # Apply similarity kernel if provided
        if self.sim_kernel is not None:
            self.logger.info("Applying similarity kernel to adjacency matrix...")
            # Transform distances using the similarity kernel
            data_values = self.sim_kernel(self._adjacency.data)
            # Create a new adjacency matrix with transformed values
            adj_mat = csr_matrix(
                (data_values, (self._row_idx, self._col_idx)),
                shape=(self._n_samples, self._n_samples),
            )
        else:
            adj_mat = self._adjacency

        # Make the adjacency matrix symmetric by adding it to its transpose
        symmetric_adj = adj_mat + adj_mat.T

        # Compute the normalized Laplacian
        lap = laplacian(symmetric_adj, normed=True)

        # Find the k smallest eigenvectors (k = dimension + 1)
        k = self.n_components + 1
        _, eigenvectors = eigsh(lap, k, which="SM")

        # Skip the first eigenvector (corresponds to constant function)
        self._init_embedding = eigenvectors[:, 1:k]

        self.logger.info("do_spectral_embedding done ...")

    def do_random_embedding(self):
        """
        Initialize embedding using Random Projection.

        This method creates an initial embedding of the data using random projection,
        which is a simple and computationally efficient technique for dimensionality
        reduction. It projects the data onto a randomly generated basis, providing
        a good starting point for further optimization.

        Random projection is supported by the Johnson-Lindenstrauss lemma, which
        guarantees that the distances between points are approximately preserved
        under certain conditions.

        Sets the init_embedding attribute with the random projection of the data.
        """
        self.logger.info("do_random_embedding ...")

        # Create a random projection matrix
        if self.random_state is None:
            key = random.PRNGKey(randint(0, 1000))
        else:
            key = random.PRNGKey(self.random_state)
        
        # Use appropriate dtype for MPA
        compute_dtype = jnp.float32 if self.mpa else jnp.float64
        rand_basis = random.normal(key, (self.n_components, self._data_dim), dtype=compute_dtype)

        # Move data and projection matrix to device memory with consistent dtype
        data_matrix = device_put(self._data.astype(compute_dtype))
        rand_basis = device_put(rand_basis)

        # Project data onto random basis with explicit precision control
        self._init_embedding = jnp.dot(
            data_matrix, rand_basis.T, 
            precision=jax.lax.Precision.DEFAULT if self.mpa else jax.lax.Precision.HIGHEST
        )

        self.logger.info("do_random_embedding done ...")

    #
    # Efficient sampling for force-directed layout
    #

    def do_rand_sampling(self, key, arr, n_samples, n_dirs, neg_ratio):
        """
        Sample points for force calculation using cached sampling kernel.
        Uses a cached kernel to avoid recompilation issues with dynamic vmap.
        """
        self.logger.info("do_rand_sampling ...")

        arr_len = len(arr)
        compute_dtype = jnp.float32 if hasattr(self, 'mpa') and self.mpa else jnp.float64

        # Use cached sampling kernel
        sampling_kernel = _get_sampling_kernel(
            self.n_components, n_samples, n_dirs, neg_ratio, compute_dtype
        )

        sampled_indices = sampling_kernel(key, arr)

        self.logger.info("do_rand_sampling done ...")

        return sampled_indices

    #
    # Create layout using force-directed optimization
    #

    def do_layout(self, large_dataset_mode=None, force_cpu=False):
        """
        Optimize the layout using force-directed placement with JAX kernels.

        This method takes the initial embedding and iteratively refines it using
        attractive and repulsive forces to create a meaningful low-dimensional
        representation of the high-dimensional data. The algorithm applies:

        1. Attraction forces between points that are neighbors in the high-dimensional space
        2. Repulsion forces between randomly sampled points in the low-dimensional space
        3. Gradual cooling (decreasing force impact) as iterations progress

        The final layout is normalized to have zero mean and unit standard deviation.

        Parameters
        ----------
        large_dataset_mode : bool or None, optional
            If True, use memory-efficient techniques for large datasets.
            If None, automatically determine based on dataset size.
        force_cpu : bool, optional
            If True, force computations on CPU instead of GPU, which can
            be helpful for large datasets that exceed GPU memory.
        """
        self.logger.info("do_layout ...")

        # Setup parameters with appropriate dtype
        compute_dtype = jnp.float32 if self.mpa else jnp.float64

        # Handle automatic batch size calculation if needed
        sample_size = self.sample_size
        num_iterations = self.max_iter_layout

        if sample_size == "auto":
            # Scale batch size based on dataset size and neighborhood size
            sample_size = int(self.n_neighbors * np.log(self._n_samples))
            # Ensure batch size is reasonable (not too small or large)
            sample_size = max(min(512, sample_size), 32)

        # Determine if we should use memory-efficient mode for large datasets
        if large_dataset_mode is None:
            large_dataset_mode = (self._n_samples > 65536) or (
                jax.devices()[0].platform == "tpu"
            )

        # Other parameters
        n_dirs = self.n_sample_dirs
        neg_ratio = self.neg_ratio

        # Debug initial embedding precision
        self.logger.debug(
            f"[LAYOUT] Initial embedding precision: {self._init_embedding.dtype}"
        )

        # Initialize and normalize positions with appropriate dtype
        if force_cpu:
            self.logger.info("Forcing computations on CPU")
            cpu_device = jax.devices("cpu")[0]
            init_pos_jax = device_put(self._init_embedding.astype(compute_dtype), device=cpu_device)
            neighbor_indices_jax = device_put(self._indices_np, device=cpu_device)
        else:
            init_pos_jax = device_put(self._init_embedding.astype(compute_dtype))
            neighbor_indices_jax = device_put(self._indices_jax)

        # Use dtype-consistent operations for normalization
        mean_pos = init_pos_jax.mean(axis=0, keepdims=True)
        init_pos_jax = init_pos_jax - mean_pos  # Center positions
        std_pos = init_pos_jax.std(axis=0, keepdims=True)
        # Avoid division by zero with appropriate epsilon for dtype
        eps = jnp.array(1e-7 if self.mpa else 1e-15, dtype=compute_dtype)
        init_pos_jax = init_pos_jax / jnp.maximum(std_pos, eps)  # Normalize variance

        # Set random seed for reproducibility
        if self.random_state is None:
            key = random.PRNGKey(randint(0, 1000))
        else:
            key = random.PRNGKey(self.random_state)

        # Use cached layout optimization kernel - this replaces the entire Python loop
        # with a single JIT-compiled operation that avoids all recompilation issues
        self.logger.info(f"Using cached layout optimization kernel for {num_iterations} iterations")

        # Use a reasonable max_iterations to allow caching across different iteration counts
        # This enables a single kernel to handle 8, 16, 32, 64, 128 iterations efficiently
        max_iterations = max(128, num_iterations)

        # Calculate chunk size for large dataset mode
        chunk_size = None
        if large_dataset_mode:
            platform = jax.devices()[0].platform
            base_chunk_size = self.memm.get(platform, self.memm.get("other", 8192))
            # Reduce chunk size significantly for large dataset mode
            chunk_size = max(base_chunk_size // 4, 1024)  # At least 1024, but 1/4 of normal batch size
            self.logger.info(f"Using large dataset mode with chunk size: {chunk_size} (platform: {platform})")
        else:
            self.logger.info("Using standard dataset mode (no chunking)")

        layout_kernel = _get_layout_optimization_kernel(
            max_iterations=max_iterations,
            sample_size=sample_size,
            n_dirs=n_dirs,
            neg_ratio=neg_ratio,
            cutoff=self.cutoff,
            a=self._a,
            b=self._b,
            use_float32=self.mpa,
            large_dataset_mode=large_dataset_mode,
            chunk_size=chunk_size
        )

        # Run the complete optimization in a single JIT-compiled call with dynamic iterations
        final_positions = layout_kernel(init_pos_jax, neighbor_indices_jax, key, num_iterations)

        # Store final layout
        self._layout = np.asarray(final_positions)

        # Clear any cached values to free memory
        gc.collect()

        self.logger.info("do_layout done ...")

    # Modified _compute_forces method to use the kernel
    def _compute_forces(
        self, positions, chunk_indices, neighbor_indices, sample_indices, alpha=1.0
    ):
        """
        Compute attractive and repulsive forces for points using JAX kernels.

        This method uses JAX-optimized kernels to efficiently compute forces
        between points during layout optimization.

        Parameters
        ----------
        positions : jax.numpy.ndarray
            Current point positions
        chunk_indices : jax.numpy.ndarray
            Current batch indices
        neighbor_indices : jax.numpy.ndarray
            Indices of neighbors for attractive forces
        sample_indices : jax.numpy.ndarray
            Indices of points for repulsive forces
        alpha : float
            Cooling factor that scales force magnitude

        Returns
        -------
        jax.numpy.ndarray
            Net force vectors for each point
        """

        self.logger.debug(f"[FORCE] Computing forces on device: {positions.device}")
        self.logger.debug(f"[FORCE] Using MPA: {self.mpa}")

        # Call the JAX-optimized kernel with MPA flag
        return compute_forces_kernel(
            positions,
            chunk_indices,
            neighbor_indices,
            sample_indices,
            alpha,
            self._a,
            self._b,
            use_float32=self.mpa,
        )

    #
    # Visualize the layout
    #

    def visualize(
        self,
        labels=None,
        point_size=2,
        title=None,
        colormap=None,
        width=800,
        height=600,
        opacity=0.7,
    ):
        """
        Generate an interactive visualization of the data in the transformed space.

        This method creates a scatter plot visualization of the embedded data, supporting
        both 2D and 3D visualizations depending on the specified dimension. Points can be
        colored by provided labels for clearer visualization of clusters or categories.

        Parameters
        ----------
        labels : numpy.ndarray or None, optional
            Labels for each data point to color the points in the visualization.
            If None, all points will have the same color. Default is None.
        point_size : int or float, optional
            Size of points in the scatter plot. Default is 2.
        title : str or None, optional
            Title for the visualization. If None, a default title will be used. Default is None.
        colormap : str or None, optional
            Name of the colormap to use for labels (e.g., 'viridis', 'plasma').
            If None, the default Plotly colormap will be used. Default is None.
        width : int, optional
            Width of the figure in pixels. Default is 800.
        height : int, optional
            Height of the figure in pixels. Default is 600.
        opacity : float, optional
            Opacity of the points (0.0 to 1.0). Default is 0.7.

        Returns
        -------
        plotly.graph_objs._figure.Figure or None
            A Plotly figure object if the visualization is successful;
            None if no layout is available or dimension > 3.

        Notes
        -----
        For 3D visualizations, you can rotate, zoom, and pan the plot interactively.
        For both 2D and 3D, hover over points to see their coordinates and labels.
        """
        # Check if layout is available
        if self._layout is None:
            self.logger.warning("visualize ERROR: no layout available")
            return None

        # Set default title if not provided
        if title is None:
            title = (
                f"{self.init.capitalize()} Initialized {self.n_components}D Embedding"
            )

        # Common visualization parameters
        vis_params = {
            "color": "label" if labels is not None else None,
            "color_continuous_scale": colormap,
            "opacity": opacity,
            "title": title,
            "hover_data": ["label"] if labels is not None else None,
        }

        # Create 2D visualization
        if self.n_components == 2:
            self.logger.info("visualize: 2D ...")

            # Create dataframe for plotting
            datadf = pd.DataFrame(self._layout, columns=["x", "y"])

            # Add labels if provided
            if labels is not None:
                datadf["label"] = labels

            # Create scatter plot
            fig = px.scatter(datadf, x="x", y="y", **vis_params)

            # Update layout
            fig.update_layout(
                width=width,
                height=height,
                xaxis_title="x",
                yaxis_title="y",
            )

        # Create 3D visualization
        elif self.n_components == 3:
            self.logger.info("visualize: 3D ...")

            # Create dataframe for plotting
            datadf = pd.DataFrame(self._layout, columns=["x", "y", "z"])

            # Add labels if provided
            if labels is not None:
                datadf["label"] = labels

            # Create 3D scatter plot
            fig = px.scatter_3d(datadf, x="x", y="y", z="z", **vis_params)

            # Update layout
            fig.update_layout(
                width=width,
                height=height,
                scene={
                    "xaxis_title": "x",
                    "yaxis_title": "y",
                    "zaxis_title": "z",
                },
            )

        # Return None for higher dimensions
        else:
            self.logger.warning("visualize ERROR: dimension > 3")
            return None

        # Update marker properties
        fig.update_traces(marker={"size": point_size})

        return fig


#
# Kernel for force-directed layout
#


# Cache force computation kernels to avoid recompilation
@functools.lru_cache(maxsize=8)
def _get_force_computation_kernel(a, b, use_float32):
    """Get or create a cached force computation kernel for the given parameters."""

    @jax.jit
    def force_kernel(positions, chunk_indices, neighbor_indices, sample_indices, alpha):
        return _compute_forces_impl(positions, chunk_indices, neighbor_indices, sample_indices, alpha, a, b, use_float32)

    return force_kernel

def compute_forces_kernel(
    positions, chunk_indices, neighbor_indices, sample_indices, alpha, a, b, use_float32=True
):
    """
    Cached wrapper for force computation to avoid recompilation.
    Gets or creates a cached kernel for the given a, b, use_float32 configuration.
    """
    kernel = _get_force_computation_kernel(a, b, use_float32)
    return kernel(positions, chunk_indices, neighbor_indices, sample_indices, alpha)

def _compute_forces_impl(
    positions, chunk_indices, neighbor_indices, sample_indices, alpha, a, b, use_float32
):
    """
    Fully vectorized JAX implementation of force computation.
    This is the core computation that gets JIT-compiled and cached.
    """
    
    # Convert to appropriate dtype at the beginning
    compute_dtype = jnp.float32 if use_float32 else jnp.float64
    positions = positions.astype(compute_dtype)
    alpha = jnp.array(alpha, dtype=compute_dtype)
    a = jnp.array(a, dtype=compute_dtype)
    b = jnp.array(b, dtype=compute_dtype)
    
    # Get current positions for processing
    current_positions = positions[chunk_indices]  # (N, D)
    forces = jnp.zeros_like(current_positions, dtype=compute_dtype)

    # ===== ATTRACTION FORCES (k-NN only) =====
    # Get neighbor positions efficiently using advanced indexing
    neighbor_positions = positions[neighbor_indices]  # (N, k, D)
    
    # Broadcast current positions for vectorized computation
    current_pos_expanded = current_positions[:, None, :]  # (N, 1, D)
    
    # Compute differences and distances
    att_diff = neighbor_positions - current_pos_expanded  # (N, k, D)
    # Use more stable distance computation for MPA
    att_dist_sq = jnp.sum(att_diff * att_diff, axis=2, keepdims=True)  # (N, k, 1)
    att_dist = jnp.sqrt(att_dist_sq + jnp.array(1e-10, dtype=compute_dtype))  # (N, k, 1)
    
    # Attraction coefficient: 1 / (1 + a * (1/d)^(2b))
    # Use more numerically stable computation
    inv_dist = jnp.array(1.0, dtype=compute_dtype) / att_dist
    att_coeff = jnp.array(1.0, dtype=compute_dtype) / (jnp.array(1.0, dtype=compute_dtype) + a * jnp.power(inv_dist, 2.0 * b))  # (N, k, 1)
    
    # Compute attraction forces and sum over neighbors
    att_forces = jnp.sum(att_coeff * att_diff / att_dist, axis=1)  # (N, D)
    forces += att_forces

    # ===== REPULSION FORCES (Random Sampling) =====
    if sample_indices.size > 0:
        # Get negative sample positions
        sample_positions = positions[sample_indices]  # (N, n_neg, D)
        
        # Compute differences and distances
        rep_diff = sample_positions - current_pos_expanded  # (N, n_neg, D)
        # Use more stable distance computation for MPA
        rep_dist_sq = jnp.sum(rep_diff * rep_diff, axis=2, keepdims=True)  # (N, n_neg, 1)
        rep_dist = jnp.sqrt(rep_dist_sq + jnp.array(1e-10, dtype=compute_dtype))  # (N, n_neg, 1)
        
        # Repulsion coefficient: -1 / (1 + a * d^(2b))
        rep_coeff = jnp.array(-1.0, dtype=compute_dtype) / (jnp.array(1.0, dtype=compute_dtype) + a * jnp.power(rep_dist, 2.0 * b))  # (N, n_neg, 1)
        
        # Compute repulsion forces and sum over negative samples
        rep_forces = jnp.sum(rep_coeff * rep_diff / rep_dist, axis=1)  # (N, D)
        forces += rep_forces

    # Apply cooling factor
    return alpha * forces


#
# Auxiliary functions for force-directed layout
#


# Cache coefficient kernels to avoid recompilation
@functools.lru_cache(maxsize=8)
def _get_distribution_kernel(a, b):
    """Get or create a cached distribution kernel for the given parameters."""

    @jax.jit
    def dist_kernel(dist):
        return 1.0 / (1.0 + a * dist ** (2 * b))

    return dist_kernel

def distribution_kernel(dist, a, b):
    """
    Probability kernel that maps distances to similarity scores using cached kernels.

    This is a rational function approximation of a t-distribution.

    Parameters
    ----------
    dist : jax.numpy.ndarray
        Distance values
    a : float
        Scale parameter that controls the steepness of the distribution
    b : float
        Shape parameter that controls the tail behavior

    Returns
    -------
    float or jax.numpy.ndarray
        Similarity score(s) between 0 and 1
    """
    kernel = _get_distribution_kernel(a, b)
    return kernel(dist)

# Cache coefficient kernels
@functools.lru_cache(maxsize=8)
def _get_coeff_att_kernel(a, b):
    """Get or create a cached attraction coefficient kernel."""

    @jax.jit
    def att_kernel(dist):
        return 1.0 / (1.0 + a * (1.0 / dist) ** (2 * b))

    return att_kernel

@functools.lru_cache(maxsize=8)
def _get_coeff_rep_kernel(a, b):
    """Get or create a cached repulsion coefficient kernel."""

    @jax.jit
    def rep_kernel(dist):
        return -1.0 / (1.0 + a * dist ** (2 * b))

    return rep_kernel

def jax_coeff_att(dist, a, b):
    """JAX-optimized attraction coefficient function with caching."""
    kernel = _get_coeff_att_kernel(a, b)
    return kernel(dist)

def jax_coeff_rep(dist, a, b):
    """JAX-optimized repulsion coefficient function with caching."""
    kernel = _get_coeff_rep_kernel(a, b)
    return kernel(dist)


# Cache random direction kernels to avoid recompilation
@functools.lru_cache(maxsize=8)
def _get_rand_directions_kernel(dim, num, dtype):
    """Get or create a cached random directions kernel for the given configuration."""

    @jax.jit
    def rand_dir_kernel(key):
        points = random.normal(key, (num, dim), dtype=dtype)
        norms = jnp.sqrt(jnp.sum(points * points, axis=-1, keepdims=True))
        # Add small epsilon to avoid division by zero
        eps = jnp.array(1e-7 if dtype == jnp.float32 else 1e-15, dtype=dtype)
        return points / jnp.maximum(norms, eps)

    return rand_dir_kernel

def rand_directions(key, dim=2, num=100, dtype=jnp.float64):
    """
    Sample unit vectors in random directions using cached kernels.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key
    dim : int
        Dimensionality of the vectors
    num : int
        Number of random directions to sample
    dtype : jnp.dtype
        Data type for computation

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (num, dim) containing unit vectors
    """
    kernel = _get_rand_directions_kernel(dim, num, dtype)
    return kernel(key)


# Cache slice kernels to avoid recompilation
@functools.lru_cache(maxsize=16)
def _get_slice_kernel(k):
    """Get or create a cached slice kernel for the given slice size."""

    @jax.jit
    def slice_kernel(arr, i):
        return lax.dynamic_slice(arr, (i - k // 2,), (k,))

    return slice_kernel

def get_slice(arr, k, i):
    """
    Extract a slice of size k centered around index i using cached kernels.

    Parameters
    ----------
    arr : jax.numpy.ndarray
        Input array
    k : int
        Size of the slice
    i : int
        Center index position

    Returns
    -------
    jax.numpy.ndarray
        Slice of the input array
    """
    kernel = _get_slice_kernel(k)
    return kernel(arr, i)

# Cache sampling kernels to avoid recompilation in do_rand_sampling
@functools.lru_cache(maxsize=16)
def _get_sampling_kernel(n_components, n_samples, n_dirs, neg_ratio, dtype):
    """Get or create a cached sampling kernel for the given configuration."""

    @jax.jit
    def sampling_kernel(key, arr):
        arr_len = arr.shape[0]
        sampled_indices_list = []

        # Get random unit vectors for projections
        key, subkey = random.split(key)
        direction_vectors = _get_rand_directions_kernel(n_components, n_dirs, dtype)(subkey)

        # Get slice kernel for this n_samples
        slice_kernel = _get_slice_kernel(n_samples)

        # For each direction, sample points based on projections
        for i in range(n_dirs):
            vec = direction_vectors[i]

            # Project points onto the direction vector
            arr_proj = jnp.dot(vec, arr.T)

            # Sort indices by projection values
            indices_sort = jnp.argsort(arr_proj)

            # For each point, take n_samples points around it in sorted order
            # Use vmap on the cached slice kernel
            vmap_slice = vmap(slice_kernel, in_axes=(None, 0))
            indices = vmap_slice(indices_sort, jnp.arange(arr_len))

            # Reorder indices back to original ordering
            indices = indices[indices_sort]

            # Add to list of sampled indices
            sampled_indices_list.append(indices)

        # Generate random negative samples for repulsion
        n_neg_samples = int(neg_ratio * n_samples)
        key, subkey = random.split(key)
        neg_indices = random.randint(subkey, (arr_len, n_neg_samples), 0, arr_len)
        sampled_indices_list.append(neg_indices)

        # Combine all sampled indices
        sampled_indices = jnp.concatenate(sampled_indices_list, axis=-1)

        return sampled_indices

    return sampling_kernel


def _get_chunked_sampling_kernel(n_components, n_samples, n_dirs, neg_ratio, dtype, chunk_size):
    """Get or create a memory-efficient chunked sampling kernel for large datasets."""

    @jax.jit
    def chunked_sampling_kernel(key, arr):
        arr_len = arr.shape[0]

        # For large datasets, use a simpler approximation that reduces memory usage
        # Instead of processing in dynamic chunks, we use subsampling to reduce the problem size

        # Get random unit vectors for projections
        key, subkey = random.split(key)
        direction_vectors = _get_rand_directions_kernel(n_components, n_dirs, dtype)(subkey)

        # Subsample the array to reduce memory footprint
        subsample_size = min(chunk_size, arr_len)
        key, subkey = random.split(key)
        subsample_indices = random.choice(subkey, arr_len, (subsample_size,), replace=False)
        subsample_arr = arr[subsample_indices]

        sampled_indices_list = []

        # For each direction, sample points based on projections from subsampled array
        for i in range(n_dirs):
            vec = direction_vectors[i]

            # Project subsampled points onto the direction vector
            subsample_proj = jnp.dot(vec, subsample_arr.T)

            # Sort indices by projection values
            indices_sort = jnp.argsort(subsample_proj)

            # For each original point, find nearest points in the subsampled space
            arr_proj_full = jnp.dot(vec, arr.T)

            def find_neighbors_for_point(point_proj):
                # Find position in sorted subsampled projections
                pos = jnp.searchsorted(subsample_proj[indices_sort], point_proj)

                # Sample around this position in the subsampled space
                start = jnp.maximum(0, pos - n_samples // 2)
                end = jnp.minimum(subsample_size, start + n_samples)
                start = jnp.maximum(0, end - n_samples)

                # Map back to original indices using dynamic_slice
                subsampled_neighbors = jax.lax.dynamic_slice(indices_sort, (start,), (n_samples,))
                return subsample_indices[subsampled_neighbors]

            vmap_find_neighbors = vmap(find_neighbors_for_point)
            indices = vmap_find_neighbors(arr_proj_full)
            sampled_indices_list.append(indices)

        # Generate random negative samples
        n_neg_samples = int(neg_ratio * n_samples)
        key, subkey = random.split(key)
        neg_indices = random.randint(subkey, (arr_len, n_neg_samples), 0, arr_len)
        sampled_indices_list.append(neg_indices)

        # Combine all sampled indices
        sampled_indices = jnp.concatenate(sampled_indices_list, axis=-1)

        return sampled_indices

    return chunked_sampling_kernel

# Cache complete layout optimization kernels with max iterations
@functools.lru_cache(maxsize=8)
def _get_layout_optimization_kernel(max_iterations, sample_size, n_dirs, neg_ratio, cutoff, a, b, use_float32, large_dataset_mode=False, chunk_size=None):
    """Get or create a cached layout optimization kernel for the given configuration."""

    @jax.jit
    def layout_kernel(init_positions, neighbor_indices, initial_key, actual_iterations):
        """
        Complete JIT-compiled layout optimization loop with dynamic iteration count.
        This replaces the entire Python for-loop with a JAX-optimized implementation.
        """
        n_components = init_positions.shape[1]
        compute_dtype = jnp.float32 if use_float32 else jnp.float64

        # Ensure consistent dtypes
        positions = init_positions.astype(compute_dtype)
        cutoff_typed = jnp.array(cutoff, dtype=compute_dtype)
        a_typed = jnp.array(a, dtype=compute_dtype)
        b_typed = jnp.array(b, dtype=compute_dtype)

        # Get cached kernels - use chunked sampling for large datasets to reduce memory usage
        if large_dataset_mode and chunk_size is not None:
            sampling_kernel = _get_chunked_sampling_kernel(
                n_components, sample_size, n_dirs, neg_ratio, compute_dtype, chunk_size
            )
        else:
            sampling_kernel = _get_sampling_kernel(n_components, sample_size, n_dirs, neg_ratio, compute_dtype)

        force_kernel = _get_force_computation_kernel(a, b, use_float32)

        def optimization_step(carry, iter_id):
            current_positions, key = carry

            # Split key for this iteration
            key, subkey = random.split(key)

            # Skip iterations beyond actual_iterations
            skip_iteration = iter_id >= actual_iterations

            # Calculate cooling factor (use actual_iterations for proper scaling)
            alpha = jnp.where(skip_iteration, 0.0, 1.0 - iter_id / actual_iterations)

            # Sample points for force calculation
            sample_indices = sampling_kernel(subkey, current_positions)

            # Conditionally compute forces and update positions using lax.cond
            def do_update(_):
                n_points = current_positions.shape[0]
                chunk_indices = jnp.arange(n_points)
                net_force = force_kernel(
                    current_positions, chunk_indices, neighbor_indices, sample_indices, alpha
                )
                net_force = jnp.clip(net_force, -cutoff_typed, cutoff_typed)
                return current_positions + net_force

            def skip_update(_):
                return current_positions

            new_positions = lax.cond(skip_iteration, skip_update, do_update, None)

            return (new_positions, key), None

        # Run the optimization loop using scan for efficiency with max iterations
        (final_positions, _), _ = lax.scan(
            optimization_step,
            (positions, initial_key),
            jnp.arange(max_iterations)
        )

        # Final normalization
        mean_final = final_positions.mean(axis=0, keepdims=True)
        final_positions = final_positions - mean_final
        std_final = final_positions.std(axis=0, keepdims=True)
        eps = jnp.array(1e-7 if use_float32 else 1e-15, dtype=compute_dtype)
        final_positions = final_positions / jnp.maximum(std_final, eps)

        return final_positions

    return layout_kernel
