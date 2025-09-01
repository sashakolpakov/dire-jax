#!/usr/bin/env python3
"""
Quick test to verify the HPIndex fix works with small datasets.
"""
import numpy as np
from dire_jax import DiRe

print("Testing DiRe with small dataset...")

# Create small test dataset
np.random.seed(42)
X = np.random.randn(100, 10).astype(np.float32)

print(f"Dataset shape: {X.shape}")

try:
    # Test with default parameters
    dire = DiRe(n_neighbors=5, max_iter_layout=10, verbose=True)
    embedding = dire.fit_transform(X)
    
    print(f"Success! Embedding shape: {embedding.shape}")
    print("Test passed - no hangs or errors!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()