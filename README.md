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

## DImensionality REduction with JAX and PyTorch Backends

DiRe-JAX provides high-performance dimensionality reduction with support for both JAX and PyTorch backends. The PyTorch backend achieves strong performance improvements (100x+ speedup) through PyKeOps integration.

## Installation

### Standard Installation (JAX Backend)
```bash    
pip install dire-jax
```

### With Utilities
```bash
pip install dire-jax[utils]
```

### PyTorch Backend (High Performance)
```bash
pip install dire-jax[torch]
```

### All Features
```bash
pip install dire-jax[all]
# or
pip install dire-jax[utils,torch]
```

This installs PyTorch and PyKeOps for GPU-accelerated performance.

> **Note**: For GPU acceleration, JAX requires hardware-specific installation. See [JAX documentation](https://github.com/google/jax#installation) for GPU/TPU support.

## Quick Start

### JAX Backend
```python
from dire_jax import DiRe
from sklearn.datasets import make_blobs

# Generate test data
X, y = make_blobs(n_samples=10000, n_features=100, centers=12, random_state=42)

# Create reducer
reducer = DiRe(
    n_components=2,
    n_neighbors=16,
    init='pca',
    metric='lp',
    p=2,
    max_iter_layout=32,
    min_dist=1e-4,
    spread=1.0
)

# Fit and transform
embedding = reducer.fit_transform(X)
reducer.visualize(labels=y, point_size=4)
```

### PyTorch Backend (High Performance)
```python
from dire_jax import DiRePyTorch

# Drop-in replacement with same API
reducer = DiRePyTorch(
    n_components=2,
    n_neighbors=16,  # Ignored for datasets <2M points
    init='pca',
    max_iter_layout=32,
    force_knn_threshold=50000
)

embedding = reducer.fit_transform(X)
```

## Performance Comparison

### PyTorch vs JAX Backend (NVIDIA H100 80GB)

| Dataset Size | JAX Time | PyTorch Time | Speedup |
|-------------|----------|--------------|---------|
| 1K samples  | 165.5s   | 0.26s        | **626x** |
| 10K samples | 10.0s    | 0.27s        | **37x** |
| 100K samples| ~40s     | 3.2s         | **12x** |
| 1M samples  | N/A      | 85.8s        | - |
| 2M samples  | N/A      | 323.9s       | - |

The PyTorch backend eliminates k-NN computation overhead for datasets up to 2M points by using exact all-pairs force computation via PyKeOps.

## Distance Metrics

DiRe supports multiple distance metrics:

- `'lp'`: $p$-th power of $L_p$ distance (requires `p` parameter, $p \geq 2$)
- `'l1'`: Manhattan distance
- `'linf'`: Chebyshev distance  
- `'cosine'`: Cosine similarity distance
- Custom callable functions (see [documentation](https://sashakolpakov.github.io/dire-jax/))

```python
# Different metric examples
reducer_l1 = DiRe(metric='l1')
reducer_l2 = DiRe(metric='lp', p=2)  # L2 squared (default)
reducer_cosine = DiRe(metric='cosine')
```

## Backend Selection Guide

### JAX Backend
**Best for**: Research, custom metrics, TPU acceleration
- Cross-platform compatibility (CPU/GPU/TPU)
- Full support for custom distance metrics

### PyTorch Backend 
**Best for**: Performance-critical applications, large datasets (<2M points)
- Requires NVIDIA GPU with CUDA support
- Uses PyKeOps for optimized force computations
- Automatic GPU memory threshold detection


## Benchmarking

Run performance comparisons:

```bash
# Backend comparison
python tests/test_performance.py --backend-only

# Comprehensive benchmarks
python tests/test_performance.py --detailed

# Quick performance test
python tests/test_performance.py --quick
```

See the [benchmarking documentation](tests/README.md) for detailed testing options.

## Documentation

- [API Documentation](https://sashakolpakov.github.io/dire-jax/)
- [Working Paper](https://arxiv.org/abs/2503.03156) [![Paper](https://img.shields.io/badge/arXiv-read%20PDF-b31b1b.svg)](https://arxiv.org/abs/2503.03156)
- [Interactive Benchmark](https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashakolpakov/dire-jax/blob/main/tests/dire_benchmarks.ipynb)

## Contributing

Please follow the [contributing guide](https://sashakolpakov.github.io/dire-jax/contributing.html).

## Limitations

### PyTorch Backend
- Requires NVIDIA GPU with CUDA support
- Datasets >2M points need k-NN fallback (in development)
- Limited custom metric support for force computations

### JAX Backend  
- Performance limitations for large datasets (GPU memory tiling and batching)

## Acknowledgements

This work is supported by the Google Cloud Research Award number GCP19980904.