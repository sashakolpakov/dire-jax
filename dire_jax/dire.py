# dire.py

"""
Provides the main class for dimensionality reduction.

The DiRe (DImensionality REduction) class implements a modern approach to 
dimensionality reduction, leveraging JAX for efficient computation. It uses 
force-directed layout techniques combined with k-nearest neighbor graph 
construction to generate meaningful low-dimensional embeddings of 
high-dimensional data.
"""

#
# Imports
#

# JAX-related imports
from jax import jit, lax, vmap, random, device_put
import jax.numpy as jnp
import functools

# Scientific and numerical libraries
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA, KernelPCA

# Data structures and visualization
import pandas as pd
import plotly.express as px

# Utilities
from tqdm import tqdm
import gc
import sys
import os
from loguru import logger

# Nearest neighbor search
import faiss


#
# Main class for DImension REduction
#
class DiRe:
    """
    Dimension Reduction (DiRe) is a class designed to reduce the dimensionality of high-dimensional
    data using various embedding techniques and optimization algorithms. It supports embedding
    initialization methods such as random and spectral embeddings and utilizes k-nearest neighbors
    for manifold learning.

    Parameters
    ----------
    dimension: (int) Embedding dimension, default 2.
    n_neighbors: (int) Number of nearest neighbors to consider for each point, default 16.
    init_embedding_type: (str)
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

    Attributes
    ----------
    dimension: int
        Target dimensionality of the output space.
    n_neighbors: int
        Number of neighbors to consider in the k-nearest neighbors graph.
    init_embedding_type: str
        Chosen method for initial embedding.
    sim_kernel: callable
        Similarity kernel function to be used if 'init_embedding_type' is 'spectral', by default `None`.
    pca_kernel: callable
        Kernel function to be used if 'init_embedding_type' is 'pca', by default `None`.
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
    sample_size: int
        Number of samples per random direction.
    neg_ratio: int
        Ratio of negative to positive samples in the sampling process.
    logger: logger.Logger or `None`
        Logger used for logging informational and warning messages.

    Methods
    -------
    fit_transform(data)
        A convenience method that fits the model and then transforms the data.
        The separate `fit` and `transform` methods can only be used one after
        another because dimension reduction is applied to the dataset as a whole.
    visualize(labels=None, point_size=2)
        Visualizes the transformed data, optionally using labels to color the points.
    """

    def __init__(self,
                 dimension=2,
                 n_neighbors=16,
                 init_embedding_type='random',
                 sim_kernel=None,
                 pca_kernel=None,
                 max_iter_layout=128,
                 min_dist=1e-2,
                 spread=1.0,
                 cutoff=42.0,
                 n_sample_dirs=8,
                 sample_size=16,
                 neg_ratio=8,
                 my_logger=None,
                 verbose=True):
        """
        Class constructor
        """

        #
        self.dimension = dimension
        """ Embedding dimension """
        self.n_neighbors = n_neighbors
        """ Number of neighbors for kNN computations"""
        self.init_embedding_type = init_embedding_type
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
        self.init_embedding = None
        """ Initial embedding """
        self.layout = None
        """ Layout output """
        self.a = None
        """ Probability kernel parameter """
        self.b = None
        """ Probability kernel parameter """
        self.data = None
        """ Higher-dimensional data """
        self.n_samples = None
        """ Number of data points """
        self.data_dim = None
        """ Dimension of data """
        self.distances = None
        """ Distances in the kNN graph """
        self.indices = None
        """ Neighbor indices in the kNN graph """
        self.nearest_neighbor_distances = None
        """ Neighbor indices in the kNN graph, excluding the point itself """
        self.row_idx = None
        """ Row indices for nearest neighbors """
        self.col_idx = None
        """ Column indices for nearest neighbors """
        self.adjacency = None
        """ kNN adjacency matrix """
        #
        if my_logger is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, 'w')
            logger.add(sink, level="INFO")
            self.logger = logger
            """ System logger """
        else:
            self.logger = my_logger

    #
    # Fitting the distribution kernel with given min_dist and spread
    #

    def find_ab_params(self, min_dist=0.01, spread=1.0):
        """
        Rational function approximation to the probabilistic t-kernel
        """

        #
        self.logger.info('find_ab_params ...')

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))
        #
        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, _ = curve_fit(curve, xv, yv)
        #
        self.logger.info(f'a = {params[0]}, b = {params[1]}')
        self.logger.info('find_ab_params done ...')
        #
        return params[0], params[1]

    #
    # Fitting on data
    #

    def fit(self, data):
        """
        Fit the model to data: create the kNN graph and fit the probability kernel to force layout parameters.

        Parameters
        ----------
        data: (numpy.ndarray)
            High-dimensional data to fit the model. Shape (n_samples, n_features).

        Returns
        -------
        self: The DiRe instance fitted to data.
        """

        #
        self.logger.info('fit ...')
        #
        self.data = data

        self.n_samples = self.data.shape[0]
        self.data_dim = self.data.shape[1]

        self.make_knn_adjacency()

        self.a, self.b = self.find_ab_params(self.min_dist, self.spread)
        #
        self.logger.info('fit done ...')
        #
        return self

    #
    # Transform fitted data into lower-dimensional representation
    #

    def transform(self):
        """
        Transform the fitted data into a lower-dimensional layout.
        
        This method applies the selected embedding initialization technique
        to the data that has already been fitted (creating the kNN graph), 
        and then optimizes the layout using force-directed placement.
        
        The transformation process involves:
        1. Creating an initial embedding using the specified method
           (random projection, PCA, or spectral embedding)
        2. Optimizing the layout with attractive and repulsive forces
        
        Returns
        -------
        numpy.ndarray
            The lower-dimensional data embedding with shape (n_samples, dimension).
            Points are arranged to preserve the local structure of the original data.
            
        Raises
        ------
        ValueError
            If an unsupported embedding initialization method is specified.
        """
        self.logger.info('transform ...')
        
        # Create initial embedding based on specified initialization method
        if self.init_embedding_type == 'random':
            self.do_random_embedding()
        elif self.init_embedding_type == 'spectral':
            self.do_spectral_embedding()
        elif self.init_embedding_type == 'pca':
            self.do_pca_embedding()
        else:
            error_msg = f'Unsupported embedding method: "{self.init_embedding_type}"'
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Optimize the layout using force-directed placement
        self.do_layout()
        
        self.logger.info('transform done ...')
        
        return self.layout

    def fit_transform(self, data):
        """
        Fit the model to data and transform it into a low-dimensional layout.
        
        This is a convenience method that combines the fitting and transformation
        steps. It first builds the kNN graph and then creates the optimized
        layout in a single operation.
        
        Parameters
        ----------
        data : numpy.ndarray
            High-dimensional data to fit and transform.
            Shape (n_samples, n_features)
            
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
        # Store the data and perform fitting (build kNN graph)
        self.data = data
        self.fit(self.data)
        
        # Transform the data (create initial embedding and optimize layout)
        return self.transform()

    #
    # Computing the kNN adjacency matrix (sparse)
    #

    def make_knn_adjacency(self):
        """
        Internal routine building the adjacency matrix for the kNN graph.
        
        This method computes the k-nearest neighbors for each point in the dataset
        and constructs a sparse adjacency matrix representing the kNN graph.
        It attempts to use GPU acceleration if available, with a fallback to CPU.
        
        The method sets the following instance attributes:
        - distances: Distances to the k nearest neighbors (including self)
        - indices: Indices of the k nearest neighbors (including self)
        - nearest_neighbor_distances: Distances to the nearest neighbors (excluding self)
        - row_idx, col_idx: Indices for constructing the sparse adjacency matrix
        - adjacency: Sparse adjacency matrix of the kNN graph
        """
        self.logger.info('make_knn_adjacency ...')
        
        # Ensure data is in the right format for FAISS
        self.data = np.ascontiguousarray(self.data.astype(np.float32))
        n_neighbors = self.n_neighbors + 1  # Including the point itself
        data_dim = self.data.shape[1]

        # Try to use GPU for kNN search, fall back to CPU if necessary
        try:
            # Check if GPU resources are available
            gpu_available = hasattr(faiss, 'StandardGpuResources')
            
            if gpu_available:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, data_dim)
                self.logger.info('Using GPU for kNN search')
            else:
                self.logger.info('GPU resources not available, using CPU')
                index = faiss.IndexFlatL2(data_dim)
                
        except Exception as e:
            # Handle any exceptions during GPU initialization
            self.logger.warning(f'Error initializing GPU resources: {str(e)}. Falling back to CPU.')
            index = faiss.IndexFlatL2(data_dim)

        # Add data points to the index
        index.add(self.data)
        
        # Search for k nearest neighbors
        self.distances, self.indices = index.search(self.data, n_neighbors)
        
        # Extract nearest neighbor distances (excluding self)
        self.nearest_neighbor_distances = self.distances[:, 1]
        
        # Create indices for sparse matrix construction
        self.row_idx = np.repeat(np.arange(self.n_samples), n_neighbors)
        self.col_idx = self.indices.ravel()
        
        # Create sparse adjacency matrix
        data_values = self.distances.ravel()
        self.adjacency = csr_matrix(
            (data_values, (self.row_idx, self.col_idx)), 
            shape=(self.n_samples, self.n_samples)
        )

        # Clean up resources
        del index
        gc.collect()
        
        self.logger.info('make_knn_adjacency done ...')

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
        self.logger.info('do_pca_embedding ...')
        
        if self.pca_kernel is not None:
            # Use Kernel PCA for nonlinear dimensionality reduction
            self.logger.info('Using kernelized PCA embedding...')
            pca = KernelPCA(
                n_components=self.dimension, 
                kernel=self.pca_kernel
            )
            self.init_embedding = pca.fit_transform(self.data)
        else:
            # Use standard PCA for linear dimensionality reduction
            self.logger.info('Using standard PCA embedding...')
            pca = PCA(n_components=self.dimension)
            self.init_embedding = pca.fit_transform(self.data)
        
        self.logger.info('do_pca_embedding done ...')

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
        self.logger.info('do_spectral_embedding ...')
        
        # Apply similarity kernel if provided
        if self.sim_kernel is not None:
            self.logger.info('Applying similarity kernel to adjacency matrix...')
            # Transform distances using the similarity kernel
            data_values = self.sim_kernel(self.adjacency.data)
            # Create a new adjacency matrix with transformed values
            adj_mat = csr_matrix(
                (data_values, (self.row_idx, self.col_idx)), 
                shape=(self.n_samples, self.n_samples)
            )
        else:
            adj_mat = self.adjacency
        
        # Make the adjacency matrix symmetric by adding it to its transpose
        symmetric_adj = adj_mat + adj_mat.T
        
        # Compute the normalized Laplacian
        lap = laplacian(symmetric_adj, normed=True)
        
        # Find the k smallest eigenvectors (k = dimension + 1)
        k = self.dimension + 1
        _, eigenvectors = eigsh(lap, k, which='SM')
        
        # Skip the first eigenvector (corresponds to constant function)
        self.init_embedding = eigenvectors[:, 1:k]
        
        self.logger.info('do_spectral_embedding done ...')

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
        self.logger.info('do_random_embedding ...')
        
        # Create a random projection matrix
        key = random.PRNGKey(13)  # Fixed seed for reproducibility
        rand_basis = random.normal(key, (self.dimension, self.data_dim))
        
        # Move data and projection matrix to device memory
        data_matrix = device_put(self.data)
        rand_basis = device_put(rand_basis)
        
        # Project data onto random basis
        self.init_embedding = data_matrix @ rand_basis.T
        
        self.logger.info('do_random_embedding done ...')

    #
    # Efficient sampling for force-directed layout
    #

    def do_rand_sampling(self, key, arr, n_samples, n_dirs, neg_ratio):
        """
        Sample points for force calculation using random projections.
        
        This method implements an efficient sampling strategy to identify points
        for applying attractive and repulsive forces during layout optimization.
        It uses random projections to quickly identify nearby points in different
        directions, and also adds random negative samples for repulsion.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key
        arr : jax.numpy.ndarray
            Array of current point positions
        n_samples : int
            Number of samples to take in each direction
        n_dirs : int
            Number of random directions to sample
        neg_ratio : int
            Ratio of negative samples to positive samples
            
        Returns
        -------
        jax.numpy.ndarray
            Array of sampled indices for force calculations
            
        Notes
        -----
        The sampling strategy works as follows:
        1. Generate n_dirs random unit vectors
        2. Project the points onto each vector
        3. For each point, take the n_samples closest points in each direction
        4. Add random negative samples for repulsion
        5. Combine all sampled indices
        
        This approach is more efficient than a full nearest neighbor search
        while still capturing the important local relationships.
        """
        self.logger.info('do_rand_sampling ...')
        
        sampled_indices_list = []
        arr_len = len(arr)
        
        # Get random unit vectors for projections
        key, subkey = random.split(key)
        direction_vectors = rand_directions(subkey, self.dimension, n_dirs)
        
        # For each direction, sample points based on projections
        for vec in direction_vectors:
            # Project points onto the direction vector
            arr_proj = vec @ arr.T
            
            # Sort indices by projection values
            indices_sort = jnp.argsort(arr_proj)
            
            # For each point, take n_samples points around it in sorted order
            indices = vmap_get_slice(indices_sort, n_samples, jnp.arange(arr_len))
            
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
        
        self.logger.info('do_rand_sampling done ...')
        
        return sampled_indices

    #
    # Create layout using force-directed optimization
    #

    def do_layout(self):
        """
        Optimize the layout using force-directed placement.
        
        This method takes the initial embedding and iteratively refines it using
        attractive and repulsive forces to create a meaningful low-dimensional 
        representation of the high-dimensional data. The algorithm applies:
        
        1. Attraction forces between points that are neighbors in the high-dimensional space
        2. Repulsion forces between randomly sampled points in the low-dimensional space
        3. Gradual cooling (decreasing force impact) as iterations progress
        
        The final layout is normalized to have zero mean and unit standard deviation.
        """
        self.logger.info('do_layout ...')
        
        # Setup parameters
        cutoff = jnp.array([self.cutoff])
        num_iterations = self.max_iter_layout
        
        # Handle automatic batch size calculation if needed
        batch_size = self.sample_size
        if batch_size == 'auto':
            # Scale batch size based on dataset size and neighborhood size
            batch_size = int(self.n_neighbors * np.log(self.n_samples))
            # Ensure batch size is reasonable (not too small or large)
            batch_size = max(min(512, batch_size), 32)

        # Other parameters
        n_dirs = self.n_sample_dirs
        neg_ratio = self.neg_ratio

        # Initialize and normalize positions
        init_pos_jax = device_put(self.init_embedding)
        init_pos_jax -= init_pos_jax.mean(axis=0)  # Center positions
        init_pos_jax /= init_pos_jax.std(axis=0)   # Normalize variance

        # Transfer indices to device
        indices_jax = device_put(self.indices)

        # Set random seed for reproducibility
        key = random.PRNGKey(42)

        # Optimization loop
        for iter_id in tqdm(range(num_iterations)):
            logger.debug(f'Iteration {iter_id + 1}')
            
            # Sample random points for repulsion
            indices_emb_jax = self.do_rand_sampling(
                key, 
                init_pos_jax, 
                batch_size, 
                n_dirs, 
                neg_ratio
            )
            indices_emb_jax = device_put(indices_emb_jax)

            # ===== Attraction Forces =====
            # Points are attracted to their high-dimensional neighbors
            
            # Prepare positions and compute distances
            v_pos = init_pos_jax[:, None, :]  # Shape: [n_samples, 1, dimension]
            u_pos = init_pos_jax[indices_jax]  # Shape: [n_samples, n_neighbors, dimension]
            
            # Compute position differences and distances
            position_diff = u_pos - v_pos
            distance_geom = jnp.linalg.norm(position_diff, axis=2, keepdims=True)
            
            # Create mask to avoid division by zero
            mask = (distance_geom > 0)
            
            # Compute normalized direction vectors
            direction = jnp.where(mask, position_diff / distance_geom, 0.0)

            # Compute attraction forces with repulsion term
            # The repulsion term ensures that high-dimensional neighbors
            # aren't also pushed apart by the general repulsion phase
            grad_coeff_att_vals = jnp.where(
                mask,
                1.0 * vmap_coeff_att(distance_geom, self.a, self.b) - 
                1.0 * vmap_coeff_rep(distance_geom, self.a, self.b),
                0.0
            )
            
            # Sum forces from all neighbors for each point
            attraction_force = jnp.sum(grad_coeff_att_vals * direction, axis=1)

            # ===== Repulsion Forces =====
            # Points are repelled from randomly sampled points
            
            # Use sampled indices for repulsion
            u_pos = init_pos_jax[indices_emb_jax]
            position_diff = u_pos - v_pos  # Reuse v_pos
            distance_geom = jnp.linalg.norm(position_diff, axis=2, keepdims=True)
            
            # Create mask for non-zero distances
            mask = (distance_geom > 0)
            direction = jnp.where(mask, position_diff / distance_geom, 0.0)

            # Compute repulsion forces
            grad_coeff_rep_vals = jnp.where(
                mask,
                vmap_coeff_rep(distance_geom, self.a, self.b),
                0.0
            )
            repulsion_force = jnp.sum(grad_coeff_rep_vals * direction, axis=1)

            # ===== Combine Forces and Update Positions =====
            
            # Apply cooling factor (alpha) that decreases with iterations
            alpha = 1.0 - iter_id / num_iterations
            
            # Combine attraction and repulsion
            net_force = alpha * (attraction_force + repulsion_force)
            
            # Clip forces to prevent extreme movements
            net_force = jnp.clip(net_force, -cutoff, cutoff)
            
            # Update positions
            init_pos_jax += net_force

        # Normalize final layout
        init_pos_jax -= init_pos_jax.mean(axis=0)
        init_pos_jax /= init_pos_jax.std(axis=0)
        
        # Store final layout
        self.layout = np.asarray(init_pos_jax)
        
        self.logger.info('do_layout done ...')

    #
    # Visualize the layout
    #

    def visualize(self, labels=None, point_size=2, title=None, colormap=None, width=800, height=600, opacity=0.7):
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
        if self.layout is None:
            self.logger.warning('visualize ERROR: no layout available')
            return None
            
        # Set default title if not provided
        if title is None:
            title = f"{self.init_embedding_type.capitalize()} Initialized {self.dimension}D Embedding"
            
        # Common visualization parameters
        vis_params = {
            'color': 'label' if labels is not None else None,
            'color_continuous_scale': colormap,
            'opacity': opacity,
            'title': title,
            'hover_data': ['label'] if labels is not None else None,
        }
            
        # Create 2D visualization
        if self.dimension == 2:
            self.logger.info('visualize: 2D ...')
            
            # Create dataframe for plotting
            datadf = pd.DataFrame(self.layout, columns=['x', 'y'])
            
            # Add labels if provided
            if labels is not None:
                datadf['label'] = labels
                
            # Create scatter plot
            fig = px.scatter(
                datadf, 
                x='x', 
                y='y', 
                **vis_params
            )
            
            # Update layout
            fig.update_layout(
                width=width,
                height=height,
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
            )
            
        # Create 3D visualization
        elif self.dimension == 3:
            self.logger.info('visualize: 3D ...')
            
            # Create dataframe for plotting
            datadf = pd.DataFrame(self.layout, columns=['x', 'y', 'z'])
            
            # Add labels if provided
            if labels is not None:
                datadf['label'] = labels
                
            # Create 3D scatter plot
            fig = px.scatter_3d(
                datadf, 
                x='x', 
                y='y', 
                z='z', 
                **vis_params
            )
            
            # Update layout
            fig.update_layout(
                width=width,
                height=height,
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3',
                )
            )
            
        # Return None for higher dimensions
        else:
            self.logger.warning('visualize ERROR: dimension > 3')
            return None
            
        # Update marker properties
        fig.update_traces(marker=dict(size=point_size))
        
        return fig
##


#
# Auxiliary functions for force-directed layout
#

@functools.partial(jit, static_argnums=(1, 2))
def distribution_kernel(x, a, b):
    """
    Probability kernel that maps distances to similarity scores.
    
    This is a rational function approximation of a t-distribution.
    
    Parameters
    ----------
    x : float or jax.numpy.ndarray
        Distance value(s)
    a : float
        Scale parameter that controls the steepness of the distribution
    b : float
        Shape parameter that controls the tail behavior
        
    Returns
    -------
    float or jax.numpy.ndarray
        Similarity score(s) between 0 and 1
    """
    return 1.0 / (1.0 + a * x ** (2 * b))


@functools.partial(jit, static_argnums=(1, 2))
def grad_coeff_att(x, a, b):
    """
    Coefficient for attraction force based on distance.
    
    Parameters
    ----------
    x : float or jax.numpy.ndarray
        Distance value(s)
    a : float
        Scale parameter from the distribution kernel
    b : float
        Shape parameter from the distribution kernel
        
    Returns
    -------
    float or jax.numpy.ndarray
        Attraction coefficient(s)
    """
    y = distribution_kernel(1.0 / x, a, b)
    return 1.0 * y


@functools.partial(jit, static_argnums=(1, 2))
def grad_coeff_rep(x, a, b):
    """
    Coefficient for repulsion force based on distance.
    
    Parameters
    ----------
    x : float or jax.numpy.ndarray
        Distance value(s)
    a : float
        Scale parameter from the distribution kernel
    b : float
        Shape parameter from the distribution kernel
        
    Returns
    -------
    float or jax.numpy.ndarray
        Repulsion coefficient(s), negative to push points apart
    """
    y = distribution_kernel(x, a, b)
    return -1.0 * y


@functools.partial(jit, static_argnums=(1, 2))
def rand_directions(key, dim=2, num=100):
    """
    Sample unit vectors in random directions.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key
    dim : int
        Dimensionality of the vectors
    num : int
        Number of random directions to sample
        
    Returns
    -------
    jax.numpy.ndarray
        Array of shape (num, dim) containing unit vectors
    """
    points = random.normal(key, (num, dim))
    norms = jnp.sqrt(jnp.sum(points * points, axis=-1))
    return points / norms[:, None]


@functools.partial(jit, static_argnums=(1,))
def get_slice(arr, k, i):
    """
    Extract a slice of size k centered around index i.
    
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
    return lax.dynamic_slice(arr, (i - k // 2,), (k,))


# Vectorized functions for efficient parallel computation
vmap_coeff_att = vmap(grad_coeff_att, in_axes=(0, None, None))
vmap_coeff_rep = vmap(grad_coeff_rep, in_axes=(0, None, None))
vmap_get_slice = vmap(get_slice, in_axes=(None, None, 0))