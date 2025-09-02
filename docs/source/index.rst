Welcome to DiRe-JAX's documentation!
====================================

.. image:: https://img.shields.io/badge/View-PDF-red?logo=adobe
   :target: https://github.com/sashakolpakov/dire-jax/blob/main/working_paper/dire_paper.pdf
   :alt: View PDF

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb
   :alt: Open in Colab

**DiRe** is a high-performance dimensionality reduction package available with both JAX and PyTorch/PyKeOps backends for different use cases.

Quick Start
-----------

Installation
~~~~~~~~~~~~

Basic installation (JAX backend only):

.. code-block:: bash

    pip install dire-jax

With PyTorch/PyKeOps backend (recommended for CUDA GPUs):

.. code-block:: bash

    pip install dire-jax[pytorch]

Complete installation (all backends and utilities):

.. code-block:: bash

    pip install dire-jax[all]

Example Usage
~~~~~~~~~~~~~

**JAX backend:**

.. code-block:: python

    from dire_jax import DiRe
    from sklearn.datasets import make_blobs
    
    features, labels = make_blobs(n_samples=10000, n_features=100, centers=5, random_state=42)
    
    reducer = DiRe(n_components=2, n_neighbors=16, max_iter_layout=32)
    embedding = reducer.fit_transform(features)
    reducer.visualize(labels=labels, point_size=4)

**PyTorch backend (faster on CUDA):**

.. code-block:: python

    from dire_jax import DiRePyTorch as DiRe  # Drop-in replacement
    from sklearn.datasets import make_blobs
    
    features, labels = make_blobs(n_samples=10000, n_features=100, centers=5, random_state=42)
    
    reducer = DiRe(n_components=2, n_neighbors=16, max_iter_layout=32)
    embedding = reducer.fit_transform(features)
    reducer.visualize(labels=labels, point_size=4)

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