#!/usr/bin/env python3

"""
Benchmark script comparing JAX and PyTorch/PyKeOps implementations.
"""

import time
import numpy as np
from sklearn.datasets import make_blobs
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dire_jax import DiRe
from dire_jax.dire_pytorch import DiRePyTorch


def benchmark_implementation(implementation_class, X, name, **kwargs):
    """Benchmark a single implementation."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Initialize
    start = time.time()
    reducer = implementation_class(
        n_components=2,
        n_neighbors=16,
        init='pca',
        max_iter_layout=32,
        min_dist=1e-4,
        spread=1.0,
        verbose=False,
        **kwargs
    )
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.3f}s")
    
    # Fit and transform
    start = time.time()
    try:
        embedding = reducer.fit_transform(X)
        transform_time = time.time() - start
        print(f"Fit-transform time: {transform_time:.3f}s")
        print(f"Embedding shape: {embedding.shape}")
        
        # Compute some basic statistics
        print(f"Embedding mean: {np.mean(embedding):.6f}")
        print(f"Embedding std: {np.std(embedding):.6f}")
        
        return {
            'name': name,
            'init_time': init_time,
            'transform_time': transform_time,
            'total_time': init_time + transform_time,
            'embedding': embedding
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Run benchmarks on different dataset sizes."""
    
    # Check GPU availability
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Test different dataset sizes
    test_configs = [
        (1000, 100, 5),    # 1K samples, 100 features, 5 clusters
        (5000, 200, 10),   # 5K samples, 200 features, 10 clusters
        (10000, 500, 12),  # 10K samples, 500 features, 12 clusters
        (25000, 500, 15),  # 25K samples, 500 features, 15 clusters
    ]
    
    results = []
    
    for n_samples, n_features, n_centers in test_configs:
        print(f"\n{'#'*70}")
        print(f"Dataset: {n_samples} samples, {n_features} features, {n_centers} clusters")
        print(f"{'#'*70}")
        
        # Generate data
        print("Generating dataset...")
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_centers,
            random_state=42
        )
        
        # Benchmark JAX implementation
        jax_result = benchmark_implementation(
            DiRe, X, "JAX Implementation"
        )
        
        # Benchmark PyTorch implementation
        pytorch_result = benchmark_implementation(
            DiRePyTorch, X, "PyTorch/PyKeOps Implementation",
            force_knn_threshold=50000
        )
        
        # Compare results
        if jax_result and pytorch_result:
            speedup = jax_result['transform_time'] / pytorch_result['transform_time']
            print(f"\n{'*'*60}")
            print(f"SPEEDUP: {speedup:.2f}x")
            
            # Check embedding similarity (they won't be identical due to randomness)
            # but should have similar structure
            jax_embed = jax_result['embedding']
            pytorch_embed = pytorch_result['embedding']
            
            # Normalize for comparison
            jax_norm = (jax_embed - jax_embed.mean(axis=0)) / jax_embed.std(axis=0)
            pytorch_norm = (pytorch_embed - pytorch_embed.mean(axis=0)) / pytorch_embed.std(axis=0)
            
            # Check correlation (absolute value since orientation might differ)
            corr_x = abs(np.corrcoef(jax_norm[:, 0], pytorch_norm[:, 0])[0, 1])
            corr_y = abs(np.corrcoef(jax_norm[:, 1], pytorch_norm[:, 1])[0, 1])
            
            print(f"Embedding correlation (X axis): {corr_x:.4f}")
            print(f"Embedding correlation (Y axis): {corr_y:.4f}")
            
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'jax_time': jax_result['transform_time'],
                'pytorch_time': pytorch_result['transform_time'],
                'speedup': speedup,
                'correlation': (corr_x + corr_y) / 2
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Samples':<10} {'Features':<10} {'JAX (s)':<10} {'PyTorch (s)':<12} {'Speedup':<10} {'Correlation':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['n_samples']:<10} {r['n_features']:<10} "
              f"{r['jax_time']:<10.3f} {r['pytorch_time']:<12.3f} "
              f"{r['speedup']:<10.2f}x {r['correlation']:<10.4f}")
    
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()