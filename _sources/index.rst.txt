Welcome to dire-jax's documentation!
====================================

.. image:: https://img.shields.io/badge/View-PDF-red?logo=adobe
   :target: https://github.com/sashakolpakov/dire-jax/blob/main/working_paper/dire_paper.pdf
   :alt: View PDF

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb
   :alt: Open in Colab

**DiRe-JAX** is a new dimensionality reduction package written in JAX, offering high-performance dimensionality reduction with efficient computation.

Quick Start
-----------

Installation
~~~~~~~~~~~~

Install the main DiRe class only:

.. code-block:: bash

    pip install dire-jax

If you also need benchmarking utilities:

.. code-block:: bash

    pip install dire-jax[utils]

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    from dire_jax import DiRe
    from dire_jax.dire_utils import display_layout
    
    from sklearn.datasets import make_blobs
    
    n_samples  = 100_000
    n_features = 1_000
    n_centers  = 12
    features_blobs, labels_blobs = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=42)
    
    reducer_blobs = DiRe(dimension=2,
                         n_neighbors=16,
                         init_embedding_type='pca',
                         max_iter_layout=32,
                         min_dist=1e-4,
                         spread=1.0,
                         cutoff=4.0,
                         n_sample_dirs=8,
                         sample_size=16,
                         neg_ratio=32,
                         verbose=False,)
    
    _ = reducer_blobs.fit_transform(features_blobs)
    reducer_blobs.visualize(labels=labels_blobs, point_size=4)

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