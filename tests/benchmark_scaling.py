#!/usr/bin/env python3

"""
Scaling benchmark to find the breaking point of PyKeOps all-pairs computation.
"""

import time
import numpy as np
from sklearn.datasets import make_blobs
import psutil
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

# Import with error handling
try:
    import torch
    from dire_jax import DiRe
    from dire_jax.dire_pytorch import DiRePyTorch
except ImportError as e:
    print(f"Error: PyTorch backend dependencies not available.")
    print("Install with: pip install dire-jax[torch]")
    print(f"Specific error: {e}")
    sys.exit(1)


def get_memory_usage():
    """Get current memory usage."""
    # CPU memory
    process = psutil.Process()
    cpu_mem_gb = process.memory_info().rss / 1024**3
    
    # GPU memory
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_mem_gb = gpus[0].memoryUsed / 1024  # Convert MB to GB
            gpu_mem_percent = gpus[0].memoryUtil * 100
        else:
            gpu_mem_gb = 0
            gpu_mem_percent = 0
    except:
        gpu_mem_gb = 0
        gpu_mem_percent = 0
    
    return cpu_mem_gb, gpu_mem_gb, gpu_mem_percent


def benchmark_pytorch_scaling(n_samples, n_features=100, max_iter=16):
    """Benchmark PyTorch/PyKeOps at different scales."""
    
    print(f"\n{'='*70}")
    print(f"Testing PyTorch/PyKeOps with {n_samples:,} samples")
    print(f"{'='*70}")
    
    # Check initial memory
    cpu_before, gpu_before, gpu_percent_before = get_memory_usage()
    print(f"Memory before: CPU {cpu_before:.2f}GB, GPU {gpu_before:.2f}GB ({gpu_percent_before:.1f}%)")
    
    # Generate data
    print("Generating dataset...")
    t0 = time.time()
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=min(20, n_samples//100),  # Scale clusters with data size
        random_state=42
    )
    data_gen_time = time.time() - t0
    print(f"Data generation: {data_gen_time:.2f}s")
    
    # Memory after data generation
    cpu_after_data, gpu_after_data, gpu_percent_data = get_memory_usage()
    print(f"Memory after data: CPU {cpu_after_data:.2f}GB, GPU {gpu_after_data:.2f}GB ({gpu_percent_data:.1f}%)")
    
    try:
        # Initialize reducer
        reducer = DiRePyTorch(
            n_components=2,
            n_neighbors=16,
            init='random',  # Use random init for speed
            max_iter_layout=max_iter,
            verbose=True,
            force_knn_threshold=500000  # Force PyKeOps up to 500K samples
        )
        
        # Fit and transform
        print(f"\nRunning PyKeOps optimization ({max_iter} iterations)...")
        t0 = time.time()
        embedding = reducer.fit_transform(X)
        transform_time = time.time() - t0
        
        # Final memory usage
        cpu_final, gpu_final, gpu_percent_final = get_memory_usage()
        
        print(f"\n{'*'*70}")
        print(f"SUCCESS for {n_samples:,} samples!")
        print(f"Transform time: {transform_time:.2f}s")
        print(f"Time per iteration: {transform_time/max_iter:.2f}s")
        print(f"Memory peak: CPU {cpu_final:.2f}GB, GPU {gpu_final:.2f}GB ({gpu_percent_final:.1f}%)")
        print(f"Memory increase: CPU +{cpu_final-cpu_before:.2f}GB, GPU +{gpu_final-gpu_before:.2f}GB")
        
        # Estimate quadratic scaling
        ops_per_iter = n_samples * n_samples  # All-pairs
        time_per_op = transform_time / (max_iter * ops_per_iter)
        print(f"Time per pairwise operation: {time_per_op*1e9:.2f} nanoseconds")
        
        return {
            'n_samples': n_samples,
            'success': True,
            'time': transform_time,
            'time_per_iter': transform_time/max_iter,
            'gpu_mem_gb': gpu_final,
            'gpu_mem_increase': gpu_final - gpu_before
        }
        
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"FAILED for {n_samples:,} samples!")
        print(f"Error: {e}")
        
        # Get memory at failure
        cpu_fail, gpu_fail, gpu_percent_fail = get_memory_usage()
        print(f"Memory at failure: CPU {cpu_fail:.2f}GB, GPU {gpu_fail:.2f}GB ({gpu_percent_fail:.1f}%)")
        
        return {
            'n_samples': n_samples,
            'success': False,
            'error': str(e),
            'gpu_mem_gb': gpu_fail
        }
    
    finally:
        # Clean up
        if 'reducer' in locals():
            del reducer
        if 'embedding' in locals():
            del embedding
        del X, y
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Show cleaned memory
        cpu_clean, gpu_clean, gpu_percent_clean = get_memory_usage()
        print(f"Memory after cleanup: CPU {cpu_clean:.2f}GB, GPU {gpu_clean:.2f}GB ({gpu_percent_clean:.1f}%)")


def main():
    """Test scaling limits of PyKeOps."""
    
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Test increasingly large datasets
    # Using fewer iterations for larger datasets to save time
    test_configs = [
        (5_000, 16),     # 5K - baseline
        (10_000, 16),    # 10K
        (25_000, 8),     # 25K
        (50_000, 8),     # 50K - likely still OK
        (75_000, 4),     # 75K - pushing it
        (100_000, 4),    # 100K - probably the limit
        (150_000, 2),    # 150K - likely to fail
        (200_000, 2),    # 200K - definitely pushing limits
    ]
    
    results = []
    
    for n_samples, max_iter in test_configs:
        result = benchmark_pytorch_scaling(n_samples, n_features=100, max_iter=max_iter)
        results.append(result)
        
        # Stop if we hit a failure
        if not result['success']:
            print(f"\nStopping benchmark - hit scaling limit at {n_samples:,} samples")
            break
        
        # Also stop if getting too slow (>60s per iteration)
        if result['time_per_iter'] > 60:
            print(f"\nStopping benchmark - too slow at {n_samples:,} samples")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Samples':<12} {'Status':<10} {'Time(s)':<10} {'Time/iter':<12} {'GPU(GB)':<10}")
    print("-" * 70)
    
    for r in results:
        if r['success']:
            print(f"{r['n_samples']:<12,} {'✓ OK':<10} {r['time']:<10.2f} "
                  f"{r['time_per_iter']:<12.2f} {r['gpu_mem_gb']:<10.2f}")
        else:
            print(f"{r['n_samples']:<12,} {'✗ FAIL':<10} {'N/A':<10} "
                  f"{'N/A':<12} {r.get('gpu_mem_gb', 0):<10.2f}")
    
    # Find the breaking point
    successful = [r for r in results if r['success']]
    if successful:
        max_successful = max(successful, key=lambda x: x['n_samples'])
        print(f"\nMaximum successful dataset size: {max_successful['n_samples']:,} samples")
        print(f"Time for {max_successful['n_samples']:,} samples: {max_successful['time']:.2f}s")
        
        # Estimate where O(n²) breaks down for 1 minute runtime
        if len(successful) >= 2:
            # Use quadratic fit to estimate
            ns = np.array([r['n_samples'] for r in successful])
            ts = np.array([r['time_per_iter'] for r in successful])
            
            # Fit quadratic model: time = a * n²
            a = np.mean(ts / (ns ** 2))
            
            # Solve for n where time = 60 seconds
            n_max_60s = int(np.sqrt(60 / a))
            print(f"\nEstimated max samples for 60s/iteration: {n_max_60s:,}")
            
            # Memory estimate (assuming linear scaling of GPU memory)
            if successful[-1]['n_samples'] > 0:
                gb_per_sample = successful[-1]['gpu_mem_gb'] / successful[-1]['n_samples']
                max_samples_80gb = int(75 / gb_per_sample)  # Leave 5GB headroom on 80GB GPU
                print(f"Estimated max samples for 80GB GPU: {max_samples_80gb:,}")


if __name__ == "__main__":
    main()