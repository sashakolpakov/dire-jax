#!/usr/bin/env python3

"""
Quick test comparing broken vs fixed PyTorch implementations.
"""

import numpy as np
from sklearn.datasets import make_blobs
import time

# Import both PyTorch versions
from dire_jax.dire_pytorch import DiRePyTorch  # Current (broken)
from dire_jax.dire_pytorch_fixed import DiRePyTorchFixed  # Fixed version

def test_pytorch_versions(n_samples=1000):
    """
    Compare the two PyTorch implementations.
    """
    print(f"\nTesting with {n_samples} samples")
    print("="*60)
    
    # Generate test data with clear clusters
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=50,
        centers=5,
        cluster_std=0.5,
        random_state=42
    )
    
    # Common parameters
    params = {
        'n_components': 2,
        'n_neighbors': 15,
        'init': 'pca',
        'max_iter_layout': 16,  # Fewer iterations for speed
        'min_dist': 0.01,
        'spread': 1.0,
        'verbose': True,
        'random_state': 42
    }
    
    # Test current PyTorch (broken - all pairs)
    print("\n1. Current PyTorch (All-pairs attraction/repulsion):")
    print("-"*50)
    try:
        t0 = time.time()
        reducer_current = DiRePyTorch(**params)
        embedding_current = reducer_current.fit_transform(X)
        time_current = time.time() - t0
        
        print(f"\nResults:")
        print(f"  Time: {time_current:.2f}s")
        print(f"  Shape: {embedding_current.shape}")
        print(f"  Range X: [{embedding_current[:, 0].min():.2f}, {embedding_current[:, 0].max():.2f}]")
        print(f"  Range Y: [{embedding_current[:, 1].min():.2f}, {embedding_current[:, 1].max():.2f}]")
        print(f"  Std: {embedding_current.std():.4f}")
        
        # Check cluster separation
        cluster_centers_current = []
        for i in range(5):
            cluster_points = embedding_current[y == i]
            center = cluster_points.mean(axis=0)
            cluster_centers_current.append(center)
        
        # Average distance between cluster centers
        total_dist = 0
        count = 0
        for i in range(5):
            for j in range(i+1, 5):
                dist = np.linalg.norm(cluster_centers_current[i] - cluster_centers_current[j])
                total_dist += dist
                count += 1
        avg_separation_current = total_dist / count
        print(f"  Avg cluster separation: {avg_separation_current:.3f}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        embedding_current = None
        avg_separation_current = 0
    
    # Test fixed PyTorch (k-NN attraction only)
    print("\n2. Fixed PyTorch (k-NN attraction only):")
    print("-"*50)
    try:
        t0 = time.time()
        reducer_fixed = DiRePyTorchFixed(**params)
        embedding_fixed = reducer_fixed.fit_transform(X)
        time_fixed = time.time() - t0
        
        print(f"\nResults:")
        print(f"  Time: {time_fixed:.2f}s")
        print(f"  Shape: {embedding_fixed.shape}")
        print(f"  Range X: [{embedding_fixed[:, 0].min():.2f}, {embedding_fixed[:, 0].max():.2f}]")
        print(f"  Range Y: [{embedding_fixed[:, 1].min():.2f}, {embedding_fixed[:, 1].max():.2f}]")
        print(f"  Std: {embedding_fixed.std():.4f}")
        
        # Check cluster separation
        cluster_centers_fixed = []
        for i in range(5):
            cluster_points = embedding_fixed[y == i]
            center = cluster_points.mean(axis=0)
            cluster_centers_fixed.append(center)
        
        # Average distance between cluster centers
        total_dist = 0
        count = 0
        for i in range(5):
            for j in range(i+1, 5):
                dist = np.linalg.norm(cluster_centers_fixed[i] - cluster_centers_fixed[j])
                total_dist += dist
                count += 1
        avg_separation_fixed = total_dist / count
        print(f"  Avg cluster separation: {avg_separation_fixed:.3f}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        embedding_fixed = None
        avg_separation_fixed = 0
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON:")
    print(f"  Cluster separation improvement: {avg_separation_fixed/avg_separation_current:.2f}x")
    print("\nExpected: Fixed version should have BETTER cluster separation")
    print("because attraction is limited to k-NN neighbors only!")
    
    return embedding_current, embedding_fixed


if __name__ == "__main__":
    print("="*70)
    print("TESTING PYTORCH FORCE COMPUTATION FIX")
    print("="*70)
    
    # Test with small dataset for speed
    test_pytorch_versions(n_samples=500)
    
    print("\n" + "="*70)
    print("The fixed version should show:")
    print("1. Better cluster separation (clusters more spread out)")
    print("2. More stable convergence")
    print("3. Results closer to what UMAP/original DIRE produces")
    print("="*70)