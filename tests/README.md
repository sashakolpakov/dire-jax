# DiRe-JAX Testing

This directory contains tests for the DiRe-JAX package.

## Unit Tests

Unit tests are located in the `unit/` subdirectory and can be run using the `run_tests.py` script.

```bash
# From the root directory of the project
python tests/run_tests.py
```

## Benchmarks

The `dire_benchmarks.ipynb` notebook contains benchmarking code that compares DiRe-JAX to other dimensionality reduction methods like UMAP and t-SNE on various datasets.

## Running Tests

To run the unit tests, make sure you have the required dependencies installed:

```bash
pip install pytest pytest-cov
```

You can run the tests with coverage reporting:

```bash
pytest tests/unit/ --cov=dire_jax
```

## Testing Large Datasets

For testing with large datasets, you can use the memory-efficient options in the DiRe class:

```python
from dire_jax import DiRe

# Initialize with memory-efficient options
reducer = DiRe(
    n_components=2,
    n_neighbors=16,
    init='pca',
    metric='lp',     # Distance metric: 'lp', 'l1', 'linf', 'cosine', or custom callable
    p=2,             # For lp metric, p=2 gives squared L2 distance
    max_iter_layout=32,
    batch_size=5000
)

# Use fit_transform with automatic memory-efficient mode for large datasets
layout = reducer.fit_transform(data)
```

Or configure batch processing explicitly:

```python
# Create a reducer for large datasets
reducer = DiRe(
    n_components=2,
    n_neighbors=16,
    init='pca',
    metric='lp',     # Distance metric: 'lp', 'l1', 'linf', 'cosine', or custom callable  
    p=2,             # For lp metric, p=2 gives squared L2 distance
    max_iter_layout=32,
    batch_size=5000
)

# Apply fit_transform with memory-efficient options
layout = reducer.fit_transform(data)

# Custom distance metrics can also be used:
import jax.numpy as jnp

def weighted_euclidean(y_batch, x, weights):
    """Custom weighted Euclidean distance."""
    diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
    return jnp.sum(weights * diff**2, axis=2)

# Use custom metric
feature_weights = jnp.ones(data.shape[1])  # Example weights
reducer_custom = DiRe(
    n_components=2,
    metric=weighted_euclidean,
    weights=feature_weights
)

# Compute metrics with memory-efficient option
from dire_jax.hpmetrics import compute_local_metrics
metrics = compute_local_metrics(data, layout, n_neighbors=16, memory_efficient=True)
```