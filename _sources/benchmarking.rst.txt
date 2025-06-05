Benchmarking
============

DiRe-JAX includes comprehensive benchmarking utilities to compare its performance against other dimensionality reduction methods.

Metrics
-------

The package includes various metrics for evaluating dimensionality reduction quality:

* **Local Metrics**: Measure how well local structures are preserved
* **Global Metrics**: Measure how well global structures are preserved
* **Context Measures**: Measure contextual preservation
* **Quality Measures**: Overall quality metrics

Benchmarking Functions
----------------------

The `dire_utils` module provides functions for benchmarking:

.. code-block:: python

    from dire_jax.dire_utils import run_benchmark, viz_benchmark

    # Run a single benchmark or a larger sample
    results = run_benchmark(reducer,
			    features,
                            labels=labels,
                            dimension=1, # for persistence homology
                            subsample_threshold=0.1, # subsample for speed
                            rng_key=random.PRNGKey(42),
                            num_trials=1, # choose sample size
                            only_stats=True,)
    
    # and print the results ... 
    print(results)

    # You can also visualize the benchmark, check how the persistence homology 
    # changes between the original data and the embedding, and compute several 
    # goodness-of-fit metrics for comparison
    viz_benchmark(reducer,
                  features,
                  labels=labels,
                  dimension=1, # for persistence homology
                  subsample_threshold=0.1, # subsample for speed
                  rng_key=random.PRNGKey(42),
                  point_size=2)

Jupyter Notebook
----------------

For detailed benchmarking examples, see the Jupyter notebook in the repository:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb
   :alt: Open in Colab
