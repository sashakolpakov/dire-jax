# DiRe-JAX Testing

This directory contains testing and benchmarking tools for the DiRe-JAX package.

## Unit Tests

Run unit tests from the project root:
```bash
python tests/run_tests.py
```

Or with pytest:
```bash
pip install pytest pytest-cov
pytest tests/unit/ --cov=dire_jax
```

## Performance Benchmarking

### test_performance.py
Comprehensive benchmarking suite supporting JAX and PyTorch backends.

```bash
# Default benchmark (includes backend comparison if PyTorch available)
python tests/test_performance.py

# Quick benchmark for CI
python tests/test_performance.py --quick --no-save

# Backend comparison only
python tests/test_performance.py --backend-only

# MPA performance comparison
python tests/test_performance.py --mpa-only

# Scaling performance analysis
python tests/test_performance.py --scaling

# Comprehensive benchmark
python tests/test_performance.py --detailed
```

**Options:**
- `--quick`: Fast benchmark with smaller datasets
- `--detailed`: Comprehensive benchmark including all tests
- `--backend-only`: JAX vs PyTorch backend comparison
- `--mpa-only`: Mixed Precision Arithmetic performance test
- `--scaling`: Dataset size and feature scaling tests
- `--large-scale`: Tests with 10k-100k samples
- `--no-save`: Don't save results to JSON
- `--verbose`: Enable JAX compilation logging

## Specialized Benchmarks

### benchmark_pytorch.py
Direct JAX vs PyTorch backend comparison across multiple dataset sizes.
*Requires: `pip install dire-jax[torch]`*

### benchmark_scaling.py
PyKeOps scaling benchmark to find computational limits.
*Requires: `pip install dire-jax[torch]`*

### benchmark_memory.py
Memory efficiency testing for large datasets.

### check_memory_methods.py
Validation of memory-efficient method implementations.

## Testing Large Datasets

Example usage for memory-efficient processing:
```python
from dire_jax import DiRe

# Memory-efficient configuration
reducer = DiRe(
    n_components=2,
    n_neighbors=16,
    batch_size=5000,
    max_iter_layout=32
)

# Process large dataset
layout = reducer.fit_transform(large_data)
```

## Results

Benchmark results are saved to `performance_results/` with timestamps and include:
- Device information and software versions
- Detailed timing metrics
- Performance comparisons and analysis
- Historical tracking capability