#!/usr/bin/env python3

"""
Aggressive scaling benchmark to really find the limits of PyKeOps.
"""

import time
import numpy as np
from sklearn.datasets import make_blobs
import torch
import sys
import os

# Try to import GPUtil, install if needed
try:
    import GPUtil
except ImportError:
    print("Installing GPUtil for GPU monitoring...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gputil"])
    import GPUtil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dire_jax.dire_pytorch import DiRePyTorch


def get_gpu_memory():
    """Get GPU memory usage in GB."""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed / 1024, gpus[0].memoryUtil * 100
    except:
        pass
    return 0, 0


def benchmark_size(n_samples, max_iter=2):
    """Quick benchmark at a specific size."""
    
    print(f"\n{'='*60}")
    print(f"Testing {n_samples:,} samples with {max_iter} iterations")
    print(f"{'='*60}")
    
    # Check initial GPU memory
    gpu_before, gpu_percent_before = get_gpu_memory()
    print(f"GPU memory before: {gpu_before:.2f}GB ({gpu_percent_before:.1f}%)")
    
    try:
        # Generate minimal data quickly
        print("Generating data...", end=" ")
        t0 = time.time()
        X = np.random.randn(n_samples, 50).astype(np.float32)  # Smaller feature size for speed
        print(f"{time.time()-t0:.1f}s")
        
        # Create reducer with very high threshold
        reducer = DiRePyTorch(
            n_components=2,
            n_neighbors=16,
            init='random',
            max_iter_layout=max_iter,
            verbose=False,  # Less verbose for quick tests
            force_knn_threshold=10_000_000  # 10M threshold!
        )
        
        # Time the critical part
        print(f"Running PyKeOps...", end=" ")
        t0 = time.time()
        embedding = reducer.fit_transform(X)
        elapsed = time.time() - t0
        print(f"{elapsed:.2f}s")
        
        # Check GPU memory
        gpu_after, gpu_percent_after = get_gpu_memory()
        
        print(f"✓ SUCCESS!")
        print(f"  Time per iteration: {elapsed/max_iter:.2f}s")
        print(f"  GPU memory peak: {gpu_after:.2f}GB ({gpu_percent_after:.1f}%)")
        print(f"  GPU memory increase: +{gpu_after-gpu_before:.2f}GB")
        
        # Estimate operations per second
        ops = n_samples * n_samples * max_iter
        ops_per_sec = ops / elapsed
        print(f"  Performance: {ops_per_sec/1e9:.2f} billion ops/sec")
        
        # Cleanup
        del reducer, embedding, X
        torch.cuda.empty_cache()
        
        return True, elapsed/max_iter
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        
        # Check memory at failure
        gpu_fail, gpu_percent_fail = get_gpu_memory()
        print(f"  GPU memory at failure: {gpu_fail:.2f}GB ({gpu_percent_fail:.1f}%)")
        
        # Cleanup
        if 'X' in locals():
            del X
        torch.cuda.empty_cache()
        
        return False, None


def main():
    """Aggressively test scaling limits."""
    
    print("AGGRESSIVE SCALING TEST")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Start with known good, then jump aggressively
    test_sizes = [
        50_000,    # Warmup
        100_000,   # Should work
        200_000,   # Probably works
        300_000,   # Getting interesting
        500_000,   # Half million!
        750_000,   # Pushing hard
        1_000_000, # The dream - 1M samples!
        1_500_000, # Really pushing
        2_000_000, # 2M if we're lucky
    ]
    
    results = []
    last_good = 0
    last_time = 0
    
    for n_samples in test_sizes:
        # Skip if previous size was already too slow
        if last_time > 30:  # 30s per iteration is our patience limit
            print(f"\nSkipping {n_samples:,} - previous size already too slow")
            continue
            
        success, time_per_iter = benchmark_size(n_samples, max_iter=2)
        
        if success:
            last_good = n_samples
            last_time = time_per_iter
            results.append((n_samples, time_per_iter))
        else:
            print(f"\nFailed at {n_samples:,} samples")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if results:
        print(f"{'Samples':<15} {'Time/iter (s)':<15} {'Est. total (32 iter)'}")
        print("-" * 60)
        for n, t in results:
            total_est = t * 32  # Estimate for full 32 iterations
            print(f"{n:<15,} {t:<15.2f} {total_est:.1f}s ({total_est/60:.1f} min)")
        
        print(f"\n✓ Maximum successful size: {last_good:,} samples")
        
        # Quadratic fit to predict limits
        if len(results) >= 2:
            ns = np.array([r[0] for r in results])
            ts = np.array([r[1] for r in results])
            
            # Fit t = a * n^2
            a = np.mean(ts / (ns ** 2))
            
            # Predict various thresholds
            print(f"\nPREDICTIONS (based on O(n²) scaling):")
            for time_limit in [1, 10, 60, 300]:
                n_max = int(np.sqrt(time_limit / a))
                print(f"  Max samples for {time_limit}s/iter: {n_max:,}")


if __name__ == "__main__":
    main()