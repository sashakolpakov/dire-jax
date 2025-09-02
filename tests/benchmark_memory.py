#!/usr/bin/env python
"""
Simple benchmark to test memory efficiency improvements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import time

from sklearn.datasets import make_blobs

from dire_jax import DiRe


def run_benchmark(n_samples, n_features, n_centers, use_memory_efficient=False):
    """Run benchmark with specified parameters."""
    # Generate synthetic dataset
    print(
        f"Generating dataset with {n_samples} samples, {n_features} features, {n_centers} centers..."
    )

    X, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=42
    )

    # Create reducer
    print(f"Creating DiRe reducer (memory_efficient={use_memory_efficient})...")
    reducer = DiRe(
        n_components=2,
        n_neighbors=8,  # Reduced from 16
        init="pca",
        max_iter_layout=3,  # Reduced from 10 for faster testing
        verbose=True,
    )

    # Time fit_transform process
    gc.collect()  # Clean up memory before benchmark
    print("Starting fit_transform...")
    start_time = time.time()

    if use_memory_efficient:
        # Use memory-efficient approach
        print("Using memory-efficient approach...")
        reducer.fit(X)

        # Use batched kNN computation
        batch_size = min(5000, n_samples)
        reducer.make_knn_adjacency(batch_size=batch_size)

        # Use memory-efficient layout optimization
        layout = reducer.transform()
    else:
        # Use standard approach
        print("Using standard approach...")
        layout = reducer.fit_transform(X)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Layout shape: {layout.shape}")

    return elapsed


def main():
    """Run benchmarks with different dataset sizes."""
    print("==== Small Dataset Benchmark ====")
    standard_time = run_benchmark(200, 20, 3, use_memory_efficient=False)
    print("\n")
    efficient_time = run_benchmark(200, 20, 3, use_memory_efficient=True)
    print(
        f"\nSmall dataset - Standard: {standard_time:.2f}s, Efficient: {efficient_time:.2f}s"
    )

    print("\n\n==== Medium Dataset Benchmark ====")
    standard_time = run_benchmark(500, 50, 5, use_memory_efficient=False)
    print("\n")
    efficient_time = run_benchmark(500, 50, 5, use_memory_efficient=True)
    print(
        f"\nMedium dataset - Standard: {standard_time:.2f}s, Efficient: {efficient_time:.2f}s"
    )

    # Larger dataset only with memory efficient mode
    print("\n\n==== Larger Dataset Benchmark (Efficient only) ====")
    efficient_time = run_benchmark(1000, 100, 8, use_memory_efficient=True)
    print(f"\nLarger dataset - Efficient: {efficient_time:.2f}s")


if __name__ == "__main__":
    main()
