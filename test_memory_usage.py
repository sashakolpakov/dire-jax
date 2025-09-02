#!/usr/bin/env python3

"""
Test memory usage of different PyTorch implementations.
"""

import numpy as np
import torch
from sklearn.datasets import make_blobs
import time
import traceback

# Import implementations
from dire_jax.dire_pytorch import DiRePyTorch  # Current (with memory issues)
from dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient  # Fixed

def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def test_memory_usage(n_samples=5000):
    """
    Compare memory usage between implementations.
    """
    print(f"\nTesting with {n_samples} samples")
    print("="*70)
    
    # Generate test data
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=100,
        centers=10,
        cluster_std=0.5,
        random_state=42
    )
    
    # Common parameters
    params = {
        'n_components': 2,
        'n_neighbors': 30,  # Higher to stress memory
        'init': 'pca',
        'max_iter_layout': 10,  # Just a few iterations to test
        'verbose': True,
        'random_state': 42
    }
    
    # Test current implementation
    print("\n1. Current PyTorch Implementation (vectorized):")
    print("-"*50)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        mem_before = get_gpu_memory()
        print(f"GPU memory before: {mem_before:.1f} MB")
        
        t0 = time.time()
        reducer_current = DiRePyTorch(**params)
        embedding_current = reducer_current.fit_transform(X)
        time_current = time.time() - t0
        
        mem_after = get_gpu_memory()
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory = 0
        
        print(f"\nResults:")
        print(f"  Time: {time_current:.2f}s")
        print(f"  GPU memory after: {mem_after:.1f} MB")
        print(f"  Peak GPU memory: {peak_memory:.1f} MB")
        print(f"  Memory used: {peak_memory - mem_before:.1f} MB")
        
        current_memory = peak_memory - mem_before
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  This is likely due to out-of-memory!")
        traceback.print_exc()
        current_memory = float('inf')
        time_current = float('inf')
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test memory-efficient implementation
    print("\n2. Memory-Efficient Implementation:")
    print("-"*50)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        mem_before = get_gpu_memory()
        print(f"GPU memory before: {mem_before:.1f} MB")
        
        t0 = time.time()
        reducer_efficient = DiRePyTorchMemoryEfficient(**params)
        embedding_efficient = reducer_efficient.fit_transform(X)
        time_efficient = time.time() - t0
        
        mem_after = get_gpu_memory()
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory = 0
        
        print(f"\nResults:")
        print(f"  Time: {time_efficient:.2f}s")
        print(f"  GPU memory after: {mem_after:.1f} MB")
        print(f"  Peak GPU memory: {peak_memory:.1f} MB")
        print(f"  Memory used: {peak_memory - mem_before:.1f} MB")
        
        efficient_memory = peak_memory - mem_before
        
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        efficient_memory = float('inf')
        time_efficient = float('inf')
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON:")
    if current_memory != float('inf') and efficient_memory != float('inf'):
        print(f"  Memory reduction: {current_memory/efficient_memory:.2f}x")
        print(f"  Speed difference: {time_current/time_efficient:.2f}x")
    else:
        print("  Current implementation failed (likely OOM)")
        print("  Memory-efficient version succeeded!")
    
    print("\nKey differences:")
    print("  - Current: Creates (chunk_size, k, D) tensors")
    print("  - Efficient: Processes point-by-point for k-NN")
    print("  - Efficient: Uses PyKeOps LazyTensors for repulsion")


if __name__ == "__main__":
    print("="*70)
    print("MEMORY USAGE COMPARISON")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, memory tracking won't work")
        print("But we can still test for crashes")
    
    # Test with increasing sizes
    for n_samples in [1000, 5000, 10000, 20000]:
        test_memory_usage(n_samples)
        print("\n" + "="*70)