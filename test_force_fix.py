#!/usr/bin/env python3

"""
Test script to compare the broken vs fixed PyTorch implementations.
The original applies attraction/repulsion to ALL pairs.
The fixed version applies attraction ONLY to k-NN neighbors.
"""

import numpy as np
from sklearn.datasets import make_blobs
import time

# Import both versions
from dire_jax import DiRe  # JAX reference
from dire_jax.dire_pytorch import DiRePyTorch  # Current (potentially broken)
from dire_jax.dire_pytorch_fixed import DiRePyTorchFixed  # Fixed version

def test_implementations(n_samples=1000, n_features=50):
    """
    Compare the three implementations on the same data.
    """
    print(f"\nTesting with {n_samples} samples, {n_features} features")
    print("="*60)
    
    # Generate test data
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=5,
        random_state=42
    )
    
    # Common parameters
    params = {
        'n_components': 2,
        'n_neighbors': 15,
        'init': 'pca',
        'max_iter_layout': 32,
        'min_dist': 0.01,
        'spread': 1.0,
        'verbose': False,
        'random_state': 42
    }
    
    # Test JAX implementation (reference)
    print("\n1. JAX Implementation (Reference):")
    try:
        t0 = time.time()
        reducer_jax = DiRe(**params)
        embedding_jax = reducer_jax.fit_transform(X)
        time_jax = time.time() - t0
        print(f"   Time: {time_jax:.2f}s")
        print(f"   Embedding shape: {embedding_jax.shape}")
        print(f"   Mean: {embedding_jax.mean():.4f}, Std: {embedding_jax.std():.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        embedding_jax = None
    
    # Test current PyTorch implementation
    print("\n2. PyTorch Implementation (Current - All-pairs forces):")
    try:
        t0 = time.time()
        reducer_pytorch = DiRePyTorch(**params)
        embedding_pytorch = reducer_pytorch.fit_transform(X)
        time_pytorch = time.time() - t0
        print(f"   Time: {time_pytorch:.2f}s")
        print(f"   Embedding shape: {embedding_pytorch.shape}")
        print(f"   Mean: {embedding_pytorch.mean():.4f}, Std: {embedding_pytorch.std():.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        embedding_pytorch = None
    
    # Test fixed PyTorch implementation
    print("\n3. PyTorch Implementation (Fixed - k-NN attraction only):")
    try:
        t0 = time.time()
        reducer_fixed = DiRePyTorchFixed(**params)
        embedding_fixed = reducer_fixed.fit_transform(X)
        time_fixed = time.time() - t0
        print(f"   Time: {time_fixed:.2f}s")
        print(f"   Embedding shape: {embedding_fixed.shape}")
        print(f"   Mean: {embedding_fixed.mean():.4f}, Std: {embedding_fixed.std():.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        embedding_fixed = None
    
    # Skip visualization for now
    
    # Compute correlations if possible
    print("\n" + "="*60)
    print("QUALITY COMPARISON:")
    
    if embedding_jax is not None and embedding_pytorch is not None:
        # Normalize for comparison
        jax_norm = (embedding_jax - embedding_jax.mean(axis=0)) / embedding_jax.std(axis=0)
        pytorch_norm = (embedding_pytorch - embedding_pytorch.mean(axis=0)) / embedding_pytorch.std(axis=0)
        
        # Correlation (absolute since orientation might differ)
        corr_x = abs(np.corrcoef(jax_norm[:, 0], pytorch_norm[:, 0])[0, 1])
        corr_y = abs(np.corrcoef(jax_norm[:, 1], pytorch_norm[:, 1])[0, 1])
        
        print(f"JAX vs PyTorch (current): Correlation X={corr_x:.3f}, Y={corr_y:.3f}")
    
    if embedding_jax is not None and embedding_fixed is not None:
        # Normalize for comparison
        jax_norm = (embedding_jax - embedding_jax.mean(axis=0)) / embedding_jax.std(axis=0)
        fixed_norm = (embedding_fixed - embedding_fixed.mean(axis=0)) / embedding_fixed.std(axis=0)
        
        # Correlation
        corr_x = abs(np.corrcoef(jax_norm[:, 0], fixed_norm[:, 0])[0, 1])
        corr_y = abs(np.corrcoef(jax_norm[:, 1], fixed_norm[:, 1])[0, 1])
        
        print(f"JAX vs PyTorch (fixed):   Correlation X={corr_x:.3f}, Y={corr_y:.3f}")
    
    return embedding_jax, embedding_pytorch, embedding_fixed


def main():
    print("="*70)
    print("TESTING DIRE FORCE COMPUTATION: All-pairs vs k-NN Attraction")
    print("="*70)
    print("\nThe issue: Current PyTorch implementation applies attraction")
    print("and repulsion to ALL pairs, but should only apply attraction")
    print("to k-NN neighbors!")
    
    # Test with different dataset sizes
    test_implementations(n_samples=500, n_features=20)
    test_implementations(n_samples=2000, n_features=50)
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("The fixed version should produce embeddings more similar to JAX,")
    print("with better cluster separation since attraction is limited to k-NN.")
    print("="*70)


if __name__ == "__main__":
    main()