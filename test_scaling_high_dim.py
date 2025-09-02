#!/usr/bin/env python3

"""
Test PyTorch backend scaling with high-dimensional data (1000D).
Tests from 5K to 3M points.
"""

import numpy as np
import torch
from sklearn.datasets import make_blobs
import time
import gc
import traceback

from dire_jax.dire_pytorch import DiRePyTorch

def format_memory(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}PB"

def test_scaling(n_samples, n_features=1000, n_clusters=50):
    """
    Test with high-dimensional blobs.
    """
    print(f"\n{'='*70}")
    print(f"Testing {n_samples:,} samples × {n_features} dimensions")
    print(f"{'='*70}")
    
    # Memory estimate
    data_memory = n_samples * n_features * 4  # float32
    print(f"Data size: {format_memory(data_memory)}")
    
    # GPU memory check
    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        gpu_free = torch.cuda.mem_get_info()[0]
        print(f"GPU memory: {format_memory(gpu_free)} free / {format_memory(gpu_total)} total")
    
    try:
        # Generate high-dimensional blobs
        print(f"\nGenerating {n_clusters} clusters in {n_features}D...")
        t0 = time.time()
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=5.0,  # Larger spread in high dimensions
            random_state=42
        )
        gen_time = time.time() - t0
        print(f"Data generation: {gen_time:.1f}s")
        print(f"Data shape: {X.shape}")
        print(f"Data range: [{X.min():.1f}, {X.max():.1f}]")
        
        # Parameters for DIRE
        params = {
            'n_components': 2,
            'n_neighbors': 30,  # Fixed k
            'init': 'random',  # PCA would be slow in 1000D
            'max_iter_layout': 20,  # Fewer iterations for speed
            'verbose': True,
            'random_state': 42,
            'neg_ratio': 5,  # Less negative sampling for speed
        }
        
        # Run dimensionality reduction
        print(f"\nRunning PyTorch DiRe...")
        print(f"Parameters: k={params['n_neighbors']}, neg_ratio={params['neg_ratio']}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        t0 = time.time()
        reducer = DiRePyTorch(**params)
        
        # Time k-NN computation separately
        t_knn_start = time.time()
        reducer._find_ab_params()
        reducer._compute_knn(X)
        t_knn = time.time() - t_knn_start
        print(f"\nk-NN computation: {t_knn:.1f}s")
        
        # Continue with embedding
        initial_embedding = reducer._initialize_embedding(X)
        final_embedding = reducer._optimize_layout(initial_embedding)
        embedding = final_embedding.cpu().numpy()
        
        total_time = time.time() - t0
        
        # Results
        print(f"\n{'='*50}")
        print(f"SUCCESS!")
        print(f"Total time: {total_time:.1f}s")
        print(f"  - k-NN: {t_knn:.1f}s ({t_knn/total_time*100:.1f}%)")
        print(f"  - Forces + layout: {total_time-t_knn:.1f}s ({(total_time-t_knn)/total_time*100:.1f}%)")
        print(f"Throughput: {n_samples/total_time:.0f} points/sec")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding range: [{embedding.min():.2f}, {embedding.max():.2f}]")
        
        # Memory stats
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            print(f"\nPeak GPU memory: {format_memory(peak_memory)}")
            print(f"Memory per point: {peak_memory/n_samples:.0f} bytes")
        
        # Cluster quality check
        cluster_stds = []
        for i in range(min(10, n_clusters)):  # Check first 10 clusters
            cluster_points = embedding[y == i]
            if len(cluster_points) > 1:
                cluster_stds.append(np.std(cluster_points))
        avg_cluster_std = np.mean(cluster_stds)
        print(f"\nAvg cluster std (first 10): {avg_cluster_std:.3f}")
        
        # Clean up
        del X, y, embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True, total_time
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        
        # Clean up on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return False, None


if __name__ == "__main__":
    print("="*70)
    print("HIGH-DIMENSIONAL SCALING TEST (1000D)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, this will be SLOW!")
    else:
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Test points - exponential scaling
    test_sizes = [
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        3_000_000,
    ]
    
    results = []
    for n in test_sizes:
        success, time_taken = test_scaling(n, n_features=1000)
        results.append((n, success, time_taken))
        
        if not success:
            print(f"\nStopping - failed at {n:,} points")
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'N Points':<15} {'Status':<10} {'Time (s)':<10} {'Points/sec':<15}")
    print("-"*50)
    for n, success, t in results:
        status = "✓" if success else "✗"
        if t:
            throughput = f"{n/t:,.0f}"
            time_str = f"{t:.1f}"
        else:
            throughput = "N/A"
            time_str = "N/A"
        print(f"{n:<15,} {status:<10} {time_str:<10} {throughput:<15}")
    
    # Find max successful
    successful = [n for n, s, _ in results if s]
    if successful:
        max_n = max(successful)
        print(f"\nMax successful: {max_n:,} points in 1000D!")
        print(f"That's {max_n * 1000:,} total dimensions!")