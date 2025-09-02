
<!-- Logo + Project title -->
<p align="center">
  <img src="images/logo.png" alt="DiRe-JAX logo" width="280" style="margin-bottom:10px;">
</p>
<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
  <a href="https://www.python.org/downloads/">
    <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8+-blue.svg">
  </a>
  <a href="https://pypi.org/project/dire-jax/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/dire-jax.svg">
  </a>
<a style="border-width:0" href="https://doi.org/10.21105/joss.08264">
  <img src="https://joss.theoj.org/papers/10.21105/joss.08264/status.svg" alt="DOI badge" >
</a>
</p>
<p align="center">
  <a href="https://pepy.tech/projects/dire-jax">
    <img src="https://static.pepy.tech/badge/dire-jax" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/sashakolpakov/dire-jax/actions/workflows/pylint.yml">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/dire-jax/pylint.yml?branch=main&label=CI&logo=github">
  </a>
  <a href="https://github.com/sashakolpakov/dire-jax/actions/workflows/deploy_docs.yml">
    <img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/dire-jax/deploy_docs.yml?branch=main&label=Docs&logo=github">
  </a>
  <a href="https://sashakolpakov.github.io/dire-jax/">
    <img alt="Docs Live" src="https://img.shields.io/website-up-down-green-red/https/sashakolpakov.github.io/dire-jax?label=API%20Documentation">
  </a>
</p>


### A high-performance DImensionality REduction package with JAX and PyTorch backends

DiRe offers fast dimensionality reduction preserving the global dataset structure, with benchmarks showing competitive performance against UMAP and t-SNE. Available with both JAX and PyTorch/PyKeOps backends for different use cases.

### Quick start

**Basic installation (JAX backend only):**
```bash    
pip install dire-jax
```

**With utilities for benchmarking:**
```bash
pip install dire-jax[utils]
```

**With PyTorch/PyKeOps backend (recommended for CUDA GPUs):**
```bash
pip install dire-jax[pytorch]
```

**Complete installation:**
```bash
pip install dire-jax[all]
```

> **Note**: For GPU or TPU acceleration, JAX needs to be specifically installed with hardware support. See the [JAX documentation](https://github.com/google/jax#installation) for more details on enabling GPU/TPU support.


**JAX backend (default):**
```python
from dire_jax import DiRe
from sklearn.datasets import make_blobs
```

**PyTorch backend (faster on CUDA):**
```python
from dire_jax import DiRePyTorch as DiRe  # Drop-in replacement
from sklearn.datasets import make_blobs
```

**Example usage:** 

```python
n_samples  = 100_000
n_features = 1_000
n_centers  = 12
features_blobs, labels_blobs = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=42)

reducer_blobs = DiRe(n_components=2,
                     n_neighbors=16,
                     init='pca',
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

```

The output should look similar to

![12 blobs with 100k points in 1k dimensions embedded in dimension 2](images/blobs_layout.png)

### Documentation 

Please refer to the DiRe API [documentation](https://sashakolpakov.github.io/dire-jax/) for more instructions. 

### Working paper

Our working paper is available on the arXiv. [![Paper](https://img.shields.io/badge/arXiv-read%20PDF-b31b1b.svg)](https://arxiv.org/abs/2503.03156)

 Also, check out the Jupyter notebook with benchmarking results. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb
)


### Backend Performance Comparison

- **JAX backend**: Best for TPUs, good CPU performance, moderate GPU performance
- **PyTorch/PyKeOps backend**: 100x+ faster on CUDA GPUs for datasets <2M points

### Benchmarking and utilities

For benchmarking utilities and quality metrics:
```bash
pip install dire-jax[utils]
```

This provides access to dimensionality reduction quality metrics and benchmarking routines. Some utilities use external packages for persistent homology computations which may increase runtime. 

### Contributing

Please follow the [contibuting guide](https://sashakolpakov.github.io/dire-jax/contributing.html). Thanks!

### Acknowledgement 

This work is supported by the Google Cloud Research Award number GCP19980904.
