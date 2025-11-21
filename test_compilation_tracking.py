#!/usr/bin/env python3
"""
Simple compilation tracking test with JAX's built-in logging.
This test enables jax_log_compiles to see exactly when recompilation happens.
"""

import os
os.environ['JAX_LOG_COMPILES'] = '1'  # Enable before importing JAX

import jax
import numpy as np
from sklearn.datasets import make_blobs

# Now configure JAX logging
jax.config.update('jax_log_compiles', True)
jax.config.update('jax_enable_x64', True)

from dire_jax import DiRe

print("=" * 70)
print("JAX Compilation Tracking Test")
print("=" * 70)
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX platform: {jax.devices()[0].platform}")
print(f"jax_log_compiles: {jax.config.jax_log_compiles}")
print("=" * 70)

# Test 1: Basic run with small dataset
print("\n" + "=" * 70)
print("TEST 1: Basic run with small dataset (n=200, d=20)")
print("=" * 70)
print("Creating dataset...")
X1, y1 = make_blobs(n_samples=200, n_features=20, centers=3, random_state=42)

print("\nRunning DiRe with max_iter_layout=16...")
dire1 = DiRe(
    n_components=2,
    n_neighbors=8,
    init="random",
    max_iter_layout=16,
    verbose=True,
    random_state=42,
    mpa=True  # Use float32 for faster compilation
)
layout1 = dire1.fit_transform(X1)
print(f"✓ Completed. Layout shape: {layout1.shape}")

# Test 2: Same dataset size, different max_iter_layout
print("\n" + "=" * 70)
print("TEST 2: Same dataset, different iteration count (max_iter_layout=32)")
print("=" * 70)
print("This should trigger recompilation if actual_iterations is not static!")

X2, y2 = make_blobs(n_samples=200, n_features=20, centers=3, random_state=43)
dire2 = DiRe(
    n_components=2,
    n_neighbors=8,
    init="random",
    max_iter_layout=32,  # Different iteration count
    verbose=True,
    random_state=42,
    mpa=True
)
layout2 = dire2.fit_transform(X2)
print(f"✓ Completed. Layout shape: {layout2.shape}")

# Test 3: Same parameters again (should use cached kernels)
print("\n" + "=" * 70)
print("TEST 3: Same parameters as Test 1 (should NOT recompile)")
print("=" * 70)

X3, y3 = make_blobs(n_samples=200, n_features=20, centers=3, random_state=44)
dire3 = DiRe(
    n_components=2,
    n_neighbors=8,
    init="random",
    max_iter_layout=16,  # Same as Test 1
    verbose=True,
    random_state=42,
    mpa=True
)
layout3 = dire3.fit_transform(X3)
print(f"✓ Completed. Layout shape: {layout3.shape}")

# Test 4: Different n_neighbors (will cause recompilation)
print("\n" + "=" * 70)
print("TEST 4: Different n_neighbors (expected recompilation)")
print("=" * 70)

X4, y4 = make_blobs(n_samples=200, n_features=20, centers=3, random_state=45)
dire4 = DiRe(
    n_components=2,
    n_neighbors=16,  # Different n_neighbors
    init="random",
    max_iter_layout=16,
    verbose=True,
    random_state=42,
    mpa=True
)
layout4 = dire4.fit_transform(X4)
print(f"✓ Completed. Layout shape: {layout4.shape}")

# Test 5: Different MPA setting (will cause recompilation)
print("\n" + "=" * 70)
print("TEST 5: Different precision (mpa=False, expected recompilation)")
print("=" * 70)

X5, y5 = make_blobs(n_samples=200, n_features=20, centers=3, random_state=46)
dire5 = DiRe(
    n_components=2,
    n_neighbors=8,
    init="random",
    max_iter_layout=16,
    verbose=True,
    random_state=42,
    mpa=False  # Different precision
)
layout5 = dire5.fit_transform(X5)
print(f"✓ Completed. Layout shape: {layout5.shape}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Expected behavior:")
print("  - Test 1: Initial compilation (EXPECTED)")
print("  - Test 2: Recompilation due to different max_iter_layout (BUG if happens)")
print("  - Test 3: No recompilation - uses cached kernels (EXPECTED)")
print("  - Test 4: Recompilation due to different n_neighbors (EXPECTED)")
print("  - Test 5: Recompilation due to different dtype (EXPECTED)")
print("\nLook for 'Compiling...' or 'Finished XLA compilation' messages above.")
print("If Test 2 shows compilation, we have the actual_iterations recompilation bug!")
print("=" * 70)
