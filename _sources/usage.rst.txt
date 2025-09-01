Usage Guide
===========

Basic Usage
-----------

DiRe-JAX provides a high-performance dimensionality reduction tool based on JAX. Here's a quick example of how to use it:

.. code-block:: python

    from dire_jax import DiRe
    import numpy as np
    
    # Create some sample data
    data = np.random.random((1000, 50))  # 1000 samples in 50 dimensions
    
    # Initialize DiRe with desired parameters
    reducer = DiRe(
        n_components=2,         # Target dimension
        n_neighbors=15,         # Number of neighbors to consider
        init='pca',             # Initialization method
        metric='lp',            # Distance metric ('lp', 'l1', 'linf', 'cosine', or callable)
        p=2,                    # For lp metric, p=2 gives squared L2 distance
        max_iter_layout=128,    # Maximum number of layout iterations
        verbose=True            # Show progress
    )
    
    # Fit and transform the data
    embedding = reducer.fit_transform(data)
    
    # Visualize the results
    reducer.visualize()

Distance Metrics
----------------

DiRe supports multiple distance metrics for k-nearest neighbor computation:

.. code-block:: python

    # L1 Manhattan distance
    reducer_l1 = DiRe(metric='l1')
    
    # L2 squared distance (default when p=2)  
    reducer_l2 = DiRe(metric='lp', p=2)
    
    # L-infinity Chebyshev distance
    reducer_linf = DiRe(metric='linf')
    
    # Cosine distance
    reducer_cosine = DiRe(metric='cosine')
    
    # Custom distance metric
    def weighted_euclidean(y_batch, x, weights):
        """Custom weighted Euclidean distance."""
        diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
        return jnp.sum(weights * diff**2, axis=2)
    
    import jax.numpy as jnp
    weights = jnp.array([2.0, 1.0, 0.5, ...])  # Feature importance weights
    reducer_custom = DiRe(metric=weighted_euclidean, weights=weights)

Available metrics:

* `'lp'`: p-th power of Lp distance (requires `p` parameter, must be ≥ 2)
* `'l1'`: Manhattan/L1 distance
* `'linf'`: Chebyshev/L-infinity distance
* `'cosine'`: Cosine distance
* Custom callable: User-defined function with signature `my_metric(y_batch, x, **kwargs)`

Custom Metrics
--------------

DiRe-JAX supports custom distance metrics through callable functions. This allows you to implement specialized 
distance measures tailored to your specific data and use case.

Custom metric functions should have the signature:

.. code-block:: python

    def my_metric(y_batch, x, **kwargs):
        """
        Compute distances between query points and database points.
        
        Args:
            y_batch: (batch_size, d) array of query points
            x: (n, d) array of database points
            **kwargs: Additional parameters passed to the metric
            
        Returns:
            (batch_size, n) array of distances
        """
        # Your distance computation here
        return distances

Examples of custom metrics:

.. code-block:: python

    import jax.numpy as jnp
    
    # Weighted Euclidean distance
    def weighted_euclidean(y_batch, x, weights):
        diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
        return jnp.sum(weights * diff**2, axis=2)
    
    # Mahalanobis distance
    def mahalanobis(y_batch, x, inv_cov):
        diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
        return jnp.sum(diff @ inv_cov * diff, axis=2)
    
    # Custom exponential distance
    def exponential_distance(y_batch, x, alpha=1.0):
        diff = jnp.abs(y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :])
        return jnp.sum(jnp.exp(alpha * diff), axis=2)
    
    # Usage
    weights = jnp.array([2.0, 1.0, 0.5, 1.5])  # Feature weights
    reducer = DiRe(metric=weighted_euclidean, weights=weights)

**Important Notes:**

* Custom metrics are automatically JIT-compiled by JAX for optimal performance
* All parameters beyond `y_batch` and `x` should be passed as keyword arguments
* The function should be JAX-compatible (use `jax.numpy` instead of `numpy`)
* Distance values should be non-negative for proper k-NN behavior

Advanced Configuration
----------------------

DiRe offers several parameters that can be tuned to optimize the dimensionality reduction process:

* `n_components`: Target dimension for the embedding (typically 2 or 3)
* `n_neighbors`: Number of neighbors to consider when constructing the graph
* `init`: Method to initialize the embedding ('pca', 'random', 'spectral')
* `metric`: Distance metric for k-nearest neighbor computation ('lp', 'l1', 'linf', 'cosine', or custom callable)
* `p`: Power parameter for 'lp' metric (default 2, must be >= 2)
* `max_iter_layout`: Maximum number of iterations for the layout algorithm
* `min_dist`: Minimum distance between points in the embedding
* `spread`: Controls how spread out the embedding is
* `cutoff`: Maximum distance for neighbor connections
* `n_sample_dirs`: Number of sample directions for the layout algorithm
* `sample_size`: Sample size for the layout algorithm
* `batch_size`: Number of samples to process at once for memory efficiency
* `neg_ratio`: Ratio of negative to positive samples
* `random_state`: Random seed for reproducible results

Benchmarking
------------

If you've installed DiRe-JAX with the `[utils]` extra, you can use the benchmarking utilities:

.. code-block:: python

    from dire_jax import DiRe
    from dire_jax.dire_utils import run_benchmark, viz_benchmark
    from sklearn.datasets import make_blobs
    from jax import random
    
    # Create data
    features, labels = make_blobs(n_samples=10000, n_features=100, centers=5, random_state=42)
    
    # Initialize reducer
    reducer = DiRe(n_components=2, n_neighbors=15, metric='lp', p=2)
    
    # Then either run the benchmark ...
    benchmark_results = run_benchmark(reducer,
				      features,
                          	      labels=labels,
                          	      dimension=1, # for persistence homology
                             	      subsample_threshold=0.1, # subsample for speed
                          	      rng_key=random.PRNGKey(42),
                          	      num_trials=1, # choose sample size
                          	      only_stats=True,)
    
    # and print the results ... 
    print(benchmark_results)

    # ... or visualize the benchmark ...
    viz_benchmark(reducer,
                  features,
                  labels=labels,
                  dimension=1, # for persistence homology
                  subsample_threshold=0.1, # subsample for speed
                  rng_key=random.PRNGKey(42),
                  point_size=2)
