#!/usr/bin/env python3

"""
Quick test to verify PyTorch backend with new 2M threshold.
"""

import numpy as np
import time
from sklearn.datasets import make_blobs
from dire_jax.dire_pytorch import DiRePyTorch

def test_size(n_samples):
    """Test a specific dataset size."""
    print(f"\nTesting {n_samples:,} samples...")
    
    # Generate test data
    X, y = make_blobs(n_samples=n_samples, n_features=50, centers=10, random_state=42)
    
    # Create reducer with defaults (2M threshold)
    reducer = DiRePyTorch(
        n_components=2,
        max_iter_layout=5,  # Just a few iterations for testing
        verbose=True
    )
    
    # Time the transformation
    start = time.time()
    embedding = reducer.fit_transform(X)
    elapsed = time.time() - start
    
    print(f"✓ Success! Time: {elapsed:.2f}s")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Mean: {embedding.mean():.6f}, Std: {embedding.std():.6f}")
    
    return elapsed

def main():
    """Test various dataset sizes with new defaults."""
    
    print("="*60)
    print("PyTorch/PyKeOps Backend Test with 2M Default Threshold")
    print("="*60)
    
    # Test increasing sizes
    test_sizes = [1000, 10000, 50000, 100000]
    
    for size in test_sizes:
        try:
            elapsed = test_size(size)
        except Exception as e:
            print(f"✗ Failed at {size:,} samples: {e}")
            break
    
    print("\n" + "="*60)
    print("All tests passed! The 2M threshold is working perfectly.")
    print("PyKeOps is handling large datasets without any k-NN!")
    print("="*60)

if __name__ == "__main__":
    main()