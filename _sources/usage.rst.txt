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
        dimension=2,            # Target dimension
        n_neighbors=15,         # Number of neighbors to consider
        init_embedding_type='pca',  # Initialization method
        max_iter_layout=100,    # Maximum number of layout iterations
        verbose=True            # Show progress
    )
    
    # Fit and transform the data
    embedding = reducer.fit_transform(data)
    
    # Visualize the results
    reducer.visualize()

Advanced Configuration
----------------------

DiRe offers several parameters that can be tuned to optimize the dimensionality reduction process:

* `dimension`: Target dimension for the embedding (typically 2 or 3)
* `n_neighbors`: Number of neighbors to consider when constructing the graph
* `init_embedding_type`: Method to initialize the embedding ('pca', 'random')
* `max_iter_layout`: Maximum number of iterations for the layout algorithm
* `min_dist`: Minimum distance between points in the embedding
* `spread`: Controls how spread out the embedding is
* `cutoff`: Maximum distance for neighbor connections
* `n_sample_dirs`: Number of sample directions for the layout algorithm
* `sample_size`: Sample size for the layout algorithm
* `neg_ratio`: Ratio of negative to positive samples

Benchmarking
------------

If you've installed DiRe-JAX with the `[utils]` extra, you can use the benchmarking utilities:

.. code-block:: python

    from dire_jax import DiRe
    from dire_jax.dire_utils import benchmark_reducer, display_layout
    from sklearn.datasets import make_blobs
    
    # Create data
    features, labels = make_blobs(n_samples=10000, n_features=100, centers=5, random_state=42)
    
    # Initialize reducer
    reducer = DiRe(dimension=2, n_neighbors=15)
    
    # Run benchmark
    benchmark_results = benchmark_reducer(reducer, features, labels)
    
    # Print results
    print(benchmark_results)
