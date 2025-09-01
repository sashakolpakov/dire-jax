Installation
============

Requirements
------------

DiRe-JAX has multiple dependency sets for different backends:

* **Core dependencies** (required): jax, numpy, scipy, tqdm, pandas, plotly, loguru, scikit-learn
* **Utilities dependencies** (optional): ripser, persim, fastdtw, fast-twed, pot
* **PyTorch backend dependencies** (optional): torch>=1.13.0, pykeops>=2.1.0

Backend Options
~~~~~~~~~~~~~~~

.. important::
   **JAX Backend (default)**
   
   - Best for TPUs and CPU processing
   - For GPU acceleration, JAX needs specific installation: `JAX GPU instructions <https://github.com/google/jax#installation>`
   - Moderate GPU performance

.. important::
   **PyTorch/PyKeOps Backend**
   
   - 100x+ faster on CUDA GPUs for datasets <2M points
   - Uses all-pairs force computation without k-NN graphs
   - Requires CUDA-compatible GPU

Installation Options
--------------------

Basic Installation (JAX backend only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dire-jax

With Utilities for Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dire-jax[utils]

With PyTorch/PyKeOps Backend (recommended for CUDA GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dire-jax[pytorch]

Complete Installation (all backends and utilities)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dire-jax[all]

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/sashakolpakov/dire-jax.git
    cd dire-jax
    pip install -e .[all]

After installation, you may need to install JAX with GPU/TPU support separately for the JAX backend.