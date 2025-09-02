#!/usr/bin/env python
"""
Compare performance and results between MPA enabled and disabled.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.datasets import make_blobs
from dire_jax import DiRe

# Print device info
print(f"JAX devices: {jax.devices()}")
print(f"Device platform: {jax.devices()[0].platform}")
print(f"Device type: {jax.devices()[0].device_kind}")
print()

# Generate test data
print("Generating test data...")
n_samples = 5000
n_features = 100
n_centers = 12

X, y = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_centers,
    random_state=42
)

print(f"Data shape: {X.shape}")
print(f"Number of clusters: {n_centers}")
print()

# Test with MPA disabled (float64)
print("=" * 60)
print("Testing with MPA=False (float64)")
print("=" * 60)

reducer_no_mpa = DiRe(
    n_components=2,
    n_neighbors=15,
    max_iter_layout=50,  # Reduced iterations for faster testing
    verbose=True,
    mpa=False,  # Disable MPA
    random_state=42
)

start_time = time.time()
layout_no_mpa = reducer_no_mpa.fit_transform(X)
time_no_mpa = time.time() - start_time

print(f"\nTime with MPA=False: {time_no_mpa:.2f} seconds")
print(f"Output dtype: {layout_no_mpa.dtype}")
print(f"Output shape: {layout_no_mpa.shape}")
print(f"Output range: [{layout_no_mpa.min():.4f}, {layout_no_mpa.max():.4f}]")
print()

# Test with MPA enabled (float32)
print("=" * 60)
print("Testing with MPA=True (float32)")
print("=" * 60)

reducer_mpa = DiRe(
    n_components=2,
    n_neighbors=15,
    max_iter_layout=50,  # Reduced iterations for faster testing
    verbose=True,
    mpa=True,  # Enable MPA
    random_state=42
)

start_time = time.time()
layout_mpa = reducer_mpa.fit_transform(X)
time_mpa = time.time() - start_time

print(f"\nTime with MPA=True: {time_mpa:.2f} seconds")
print(f"Output dtype: {layout_mpa.dtype}")
print(f"Output shape: {layout_mpa.shape}")
print(f"Output range: [{layout_mpa.min():.4f}, {layout_mpa.max():.4f}]")
print()

# Compare results
print("=" * 60)
print("Comparison Summary")
print("=" * 60)
print(f"Speedup from MPA: {time_no_mpa / time_mpa:.2f}x")
print(f"Time saved: {time_no_mpa - time_mpa:.2f} seconds")
print(f"Memory reduction: ~50% (float32 vs float64)")

# Check similarity of results
correlation = np.corrcoef(layout_no_mpa.flatten(), layout_mpa.flatten())[0, 1]
print(f"\nCorrelation between layouts: {correlation:.4f}")

# Calculate mean absolute difference
mad = np.mean(np.abs(layout_no_mpa - layout_mpa))
print(f"Mean absolute difference: {mad:.6f}")

# Check if layouts preserve cluster structure similarly
from sklearn.metrics import silhouette_score

if n_centers > 1:
    silhouette_no_mpa = silhouette_score(layout_no_mpa, y)
    silhouette_mpa = silhouette_score(layout_mpa, y)
    print(f"\nSilhouette score (MPA=False): {silhouette_no_mpa:.4f}")
    print(f"Silhouette score (MPA=True): {silhouette_mpa:.4f}")
    print(f"Silhouette difference: {abs(silhouette_no_mpa - silhouette_mpa):.4f}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)