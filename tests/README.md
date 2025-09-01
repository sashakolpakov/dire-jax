# DiRe-JAX Testing

This directory contains tests for both JAX and PyTorch backends of DiRe-JAX.

## Unit Tests

Unit tests are located in the `unit/` subdirectory and can be run using the `run_tests.py` script.

```bash
# From the root directory of the project
python tests/run_tests.py
```

## Benchmarks

The `dire_benchmarks.ipynb` notebook contains benchmarking code that compares both DiRe backends to other dimensionality reduction methods like UMAP and t-SNE on various datasets.

## Running Tests

To run the unit tests, make sure you have the required dependencies installed:

```bash
pip install pytest pytest-cov
```

You can run the tests with coverage reporting:

```bash
pytest tests/unit/ --cov=dire_jax
```

## Testing Both Backends

**JAX backend:**
```python
from dire_jax import DiRe

reducer = DiRe(dimension=2, n_neighbors=16)
layout = reducer.fit_transform(data)
```

**PyTorch backend (faster on CUDA):**
```python
from dire_jax import DiRePyTorch

reducer = DiRePyTorch(dimension=2, n_neighbors=16)
layout = reducer.fit_transform(data)
```

## Testing Large Datasets

For JAX backend with large datasets, use memory-efficient options:

```python
from dire_jax import DiRe

reducer = DiRe(
    dimension=2,
    n_neighbors=16,
    init_embedding_type='pca',
    max_iter_layout=32
)
layout = reducer.fit_transform(data)
```

The PyTorch backend handles large datasets (up to 2M points) automatically with all-pairs force computation.