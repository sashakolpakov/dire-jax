Installation
============

Requirements
------------

DiRe-JAX has three sets of dependencies:

* **Core dependencies** (required): jax, numpy, scipy, tqdm, pandas, plotly, loguru, scikit-learn
* **Utilities dependencies** (optional): ripser, persim, fastdtw, fast-twed, pot
* **PyTorch backend** (optional): torch, pykeops

.. important::
   **JAX GPU/TPU Support**
   
   For GPU or TPU acceleration, JAX needs to be specifically installed with hardware support. The default JAX installation through pip doesn't include GPU/TPU support.
   
   To enable GPU/TPU acceleration follow the `JAX installation instructions <https://github.com/google/jax#installation>`
   
   Installing JAX with hardware acceleration can significantly improve the performance of DiRe-JAX, especially for larger datasets.

Installation Options
--------------------

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

To install the main DiRe class only:

.. code-block:: bash

    pip install dire-jax

Installation with Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install DiRe-JAX with additional utilities for benchmarking and metrics:

.. code-block:: bash

    pip install dire-jax[utils]

PyTorch Backend (High Performance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install DiRe-JAX with PyTorch backend and PyKeOps for GPU acceleration:

.. code-block:: bash

    pip install dire-jax[torch]

All Features
~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

    pip install dire-jax[all]
    # or
    pip install dire-jax[utils,torch]

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

To install for development:

.. code-block:: bash

    git clone https://github.com/sashakolpakov/dire-jax.git
    cd dire-jax
    pip install -e .[utils]

After installation, you may need to install JAX with GPU/TPU support separately as described above.