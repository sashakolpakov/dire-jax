Installation
============

Requirements
------------

DiRe-JAX has the following dependencies:

* **Core dependencies** (required): jax, numpy, scipy, tqdm, pandas, plotly, loguru, scikit-learn
* **Utilities dependencies** (optional): ripser, persim, fastdtw, fast-twed, pot

JAX Implementation
~~~~~~~~~~~~~~~~~~

.. important::
   **DiRe-JAX Features**
   
   - Optimized for small-medium datasets (<50K points)
   - Excellent CPU performance with JIT compilation
   - GPU acceleration available when JAX is installed with CUDA support
   - TPU support for cloud-based computation
   - Ideal for research and development workflows

Installation Options
--------------------

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dire-jax

With Utilities for Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dire-jax[utils]

Complete Installation (with all utilities)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dire-jax[all]

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/sashakolpakov/dire-jax.git
    cd dire-jax
    pip install -e .[all]

Hardware Acceleration
---------------------

JAX GPU Support
~~~~~~~~~~~~~~~

For GPU acceleration, JAX needs to be installed with CUDA support:

.. code-block:: bash

    # For CUDA 12
    pip install --upgrade "jax[cuda12]"
    
    # For CUDA 11
    pip install --upgrade "jax[cuda11]"

See the `JAX GPU installation guide <https://github.com/google/jax#installation>`_ for detailed instructions.

JAX TPU Support
~~~~~~~~~~~~~~~

For TPU support on Google Cloud:

.. code-block:: bash

    pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

See the `JAX TPU documentation <https://github.com/google/jax#tpu-tpu-vm>`_ for more information.