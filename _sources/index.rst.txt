Welcome to DiRe-JAX's documentation!
====================================

.. image:: https://img.shields.io/badge/View-PDF-red?logo=adobe
   :target: https://github.com/sashakolpakov/dire-jax/blob/main/working_paper/dire_paper.pdf
   :alt: View PDF

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb
   :alt: Open in Colab

**DiRe-JAX** is a high-performance dimensionality reduction package built with JAX for efficient computation on CPUs, GPUs, and TPUs.

Quick Start
-----------

Installation
~~~~~~~~~~~~

Basic installation:

.. code-block:: bash

    pip install dire-jax

With utilities for benchmarking and metrics:

.. code-block:: bash

    pip install dire-jax[utils]

Complete installation (all utilities):

.. code-block:: bash

    pip install dire-jax[all]

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    from dire_jax import DiRe
    from sklearn.datasets import make_blobs
    
    # Create sample data
    features, labels = make_blobs(
        n_samples=10000, 
        n_features=100, 
        centers=5, 
        random_state=42
    )
    
    # Initialize reducer
    reducer = DiRe(
        n_components=2, 
        n_neighbors=16, 
        max_iter_layout=32
    )
    
    # Fit and transform
    embedding = reducer.fit_transform(features)
    
    # Visualize results
    reducer.visualize(labels=labels, point_size=4)

Key Features
~~~~~~~~~~~~

* **JAX-powered**: Leverages JAX for JIT compilation and automatic differentiation
* **Hardware acceleration**: Supports CPU, GPU (via CUDA), and TPU
* **Efficient**: Optimized for datasets up to 50K points
* **Research-friendly**: Clean, modular design for experimentation
* **Benchmarking tools**: Built-in utilities for performance evaluation

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api/modules
   benchmarking
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`