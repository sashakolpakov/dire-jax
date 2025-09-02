Usage Guide
===========

Basic Usage
-----------

DiRe-JAX offers fast dimensionality reduction with JAX-based computation.

Quick Start
~~~~~~~~~~~

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
        max_iter_layout=128,    # Maximum number of layout iterations
        verbose=True            # Show progress
    )
    
    # Fit and transform the data
    embedding = reducer.fit_transform(data)
    
    # Visualize the results
    reducer.visualize()

Performance Characteristics
---------------------------

DiRe-JAX is optimized for:

* Small to medium datasets (<50K points)
* Fully vectorized computation with JIT compilation
* Excellent CPU performance
* GPU acceleration when JAX is installed with CUDA support
* TPU support for cloud-based computation

Advanced Configuration
----------------------

DiRe offers several parameters that can be tuned to optimize the dimensionality reduction process:

* `n_components`: Target dimension for the embedding (typically 2 or 3)
* `n_neighbors`: Number of neighbors to consider when constructing the graph
* `init`: Method to initialize the embedding ('pca', 'random', 'spectral')
* `max_iter_layout`: Maximum number of iterations for the layout algorithm
* `min_dist`: Minimum distance between points in the embedding
* `spread`: Controls how spread out the embedding is
* `cutoff`: Maximum distance for neighbor connections
* `n_sample_dirs`: Number of sample directions for the layout algorithm
* `sample_size`: Sample size for the layout algorithm
* `neg_ratio`: Ratio of negative to positive samples

Example with Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from dire_jax import DiRe
    from sklearn.datasets import make_blobs
    
    # Create dataset with clusters
    features, labels = make_blobs(
        n_samples=10000, 
        n_features=100, 
        centers=5, 
        random_state=42
    )
    
    # Initialize with custom parameters
    reducer = DiRe(
        n_components=2,
        n_neighbors=30,          # More neighbors for global structure
        init='spectral',         # Spectral initialization
        max_iter_layout=256,     # More iterations for convergence
        min_dist=0.01,           # Tighter packing
        spread=2.0,              # More spread out embedding
        verbose=True
    )
    
    # Fit and transform
    embedding = reducer.fit_transform(features)
    
    # Visualize with labels
    reducer.visualize(labels=labels, point_size=3)

Benchmarking
------------

If you've installed DiRe-JAX with the `[utils]` extra, you can use the benchmarking utilities:

.. code-block:: python

    from dire_jax import DiRe
    from dire_jax.dire_utils import run_benchmark, viz_benchmark
    from sklearn.datasets import make_blobs
    from jax import random
    
    # Create data
    features, labels = make_blobs(
        n_samples=10000, 
        n_features=100, 
        centers=5, 
        random_state=42
    )
    
    # Initialize reducer
    reducer = DiRe(n_components=2, n_neighbors=15)
    
    # Run the benchmark
    benchmark_results = run_benchmark(
        reducer,
        features,
        labels=labels,
        dimension=1,  # for persistence homology
        subsample_threshold=0.1,  # subsample for speed
        rng_key=random.PRNGKey(42),
        num_trials=1,  # choose sample size
        only_stats=True,
    )
    
    # Print the results
    print(benchmark_results)
    
    # Or visualize the benchmark
    viz_benchmark(
        reducer,
        features,
        labels=labels,
        dimension=1,  # for persistence homology
        subsample_threshold=0.1,  # subsample for speed
        rng_key=random.PRNGKey(42),
        point_size=2
    )

Working with Different Data Types
----------------------------------

DiRe-JAX works with various data formats:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from dire_jax import DiRe
    
    # NumPy arrays
    data_numpy = np.random.random((1000, 50))
    
    # Pandas DataFrames
    data_df = pd.DataFrame(data_numpy)
    
    # Both work seamlessly
    reducer = DiRe(n_components=2)
    embedding_numpy = reducer.fit_transform(data_numpy)
    embedding_df = reducer.fit_transform(data_df.values)