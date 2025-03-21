#!/usr/bin/env python
"""
Simple test of DiRe with timing.
"""

import time
from sklearn.datasets import make_blobs
from dire_jax import DiRe

# Generate a very small dataset
n_samples = 50
n_features = 20
n_centers = 3
print(f"Generating dataset with {n_samples} samples, {n_features} features, {n_centers} centers...")
X, y = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_centers,
    random_state=42
)

# Create reducer with minimal parameters
print("Creating DiRe reducer...")
reducer = DiRe(
    dimension=2,
    n_neighbors=5,
    init_embedding_type='random',  # Faster than PCA
    max_iter_layout=3,  # Very few iterations for testing
    verbose=True
)

# Time the operations
print("Starting fit_transform...")
start_time = time.time()
layout = reducer.fit_transform(X)
end_time = time.time()

print(f"Completed in {end_time - start_time:.2f} seconds")
print(f"Layout shape: {layout.shape}")

# Test memory-efficient version
print("\nTesting memory-efficient version...")
reducer = DiRe(
    dimension=2,
    n_neighbors=5,
    init_embedding_type='random',
    max_iter_layout=3,
    verbose=True
)

start_time = time.time()
reducer.fit(X)
reducer.make_knn_adjacency(batch_size=10)  # Small batch size for testing
layout = reducer.transform()
end_time = time.time()

print(f"Memory-efficient completed in {end_time - start_time:.2f} seconds")
print(f"Layout shape: {layout.shape}")
