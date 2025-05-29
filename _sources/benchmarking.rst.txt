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

    from dire_jax.dire_utils import benchmark_reducer, run_benchmarks, compare_benchmarks
    
    # Run a single benchmark
    results = benchmark_reducer(reducer, data, labels)
    
    # Run multiple benchmarks
    benchmark_results = run_benchmarks(
        data=data,
        labels=labels,
        reducers={'DiRe': reducer, 'PCA': pca_reducer},
        metrics=['local_preservation', 'global_preservation']
    )
    
    # Compare results
    compare_benchmarks(benchmark_results)

Jupyter Notebook
----------------

For detailed benchmarking examples, see the Jupyter notebook in the repository:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb
   :alt: Open in Colab
