#!/usr/bin/env python3

"""
Test cuVS backend with 1000-dimensional data at scale.
Compare with PyTorch backend to see if cuVS helps with high-D.
"""

import numpy as np
import torch
import time
import gc
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

# Import backends
from dire_jax import DiRePyTorch, DiReCuVS

def format_memory(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}PB"

def test_backend(backend, X, name, **kwargs):
    """Test a specific backend with timing and memory tracking."""
    print(f"\n{name}:")
    print("-" * 50)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    try:
        # Just test k-NN computation (the bottleneck)
        t0 = time.time()
        
        reducer = backend(
            n_components=2,
            n_neighbors=30,
            max_iter_layout=3,  # Very few iterations
            verbose=True,
            **kwargs
        )
        
        # Time k-NN separately
        print("Computing k-NN graph...")
        t_knn_start = time.time()
        reducer._find_ab_params()
        reducer._compute_knn(X, use_fp16=True)  # Use FP16 for PyTorch
        t_knn = time.time() - t_knn_start
        
        # Quick embedding
        print("Computing layout...")
        initial = reducer._initialize_embedding(X)
        embedding = reducer._optimize_layout(initial).cpu().numpy()
        
        total_time = time.time() - t0
        
        # Results
        print(f"\n✅ SUCCESS!")
        print(f"  k-NN time: {t_knn:.1f}s ({len(X)/t_knn:.0f} points/sec)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Embedding shape: {embedding.shape}")
        
        # Memory stats
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            print(f"  Peak GPU memory: {format_memory(peak_bytes)}")
        
        return t_knn, total_time
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    print("=" * 70)
    print("TESTING cuVS WITH 1000-DIMENSIONAL DATA")
    print("=" * 70)
    
    # Check environment
    print("\nEnvironment:")
    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
    
    try:
        import cuvs
        print("  cuVS: ✅ Available")
    except ImportError:
        print("  cuVS: ❌ Not available")
        return
    
    # Test sizes - start at 250K where PyTorch fails
    test_configs = [
        (250_000, 1000),
        (500_000, 1000),
        (750_000, 1000),
        (1_000_000, 1000),
        (1_500_000, 1000),
        (2_000_000, 1000),
        (3_000_000, 1000),
    ]
    
    results = []
    
    for n_samples, n_dims in test_configs:
        print("\n" + "=" * 70)
        print(f"Testing {n_samples:,} points in {n_dims}D")
        print("=" * 70)
        
        # Generate data
        print(f"\nGenerating {n_samples:,} × {n_dims} data...")
        t0 = time.time()
        
        # Use random data for speed (make_blobs is slow for 1000D)
        X = np.random.randn(n_samples, n_dims).astype(np.float32)
        
        gen_time = time.time() - t0
        data_size = X.nbytes
        print(f"  Generation time: {gen_time:.1f}s")
        print(f"  Data size: {format_memory(data_size)}")
        
        # Test PyTorch backend with FP16
        t_pytorch, total_pytorch = test_backend(
            DiRePyTorch, X, 
            "PyTorch Backend (FP16)"
        )
        
        # Test cuVS backend
        t_cuvs, total_cuvs = test_backend(
            DiReCuVS, X,
            "cuVS Backend",
            use_cuvs=True,
            cuvs_index_type='auto'
        )
        
        # Compare
        if t_pytorch and t_cuvs:
            speedup_knn = t_pytorch / t_cuvs
            speedup_total = total_pytorch / total_cuvs
            print(f"\n🚀 SPEEDUP:")
            print(f"  k-NN: {speedup_knn:.1f}x")
            print(f"  Total: {speedup_total:.1f}x")
            
            results.append({
                'n_samples': n_samples,
                'n_dims': n_dims,
                'pytorch_knn': t_pytorch,
                'cuvs_knn': t_cuvs,
                'speedup': speedup_knn
            })
        
        # Clean up
        del X
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Stop if we're taking too long
        if t_pytorch and t_pytorch > 120:
            print("\n⚠️ Stopping - k-NN taking too long (>2 minutes)")
            break
        
        if not (t_pytorch and t_cuvs):
            print("\n⚠️ Stopping - backend failed")
            break
    
    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY: cuVS vs PyTorch for 1000D k-NN")
        print("=" * 70)
        print(f"{'Samples':<15} {'PyTorch (s)':<15} {'cuVS (s)':<15} {'Speedup':<10}")
        print("-" * 55)
        
        for r in results:
            print(f"{r['n_samples']:<15,} {r['pytorch_knn']:<15.1f} {r['cuvs_knn']:<15.1f} {r['speedup']:<10.1f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.1f}x")
        
        # Check if cuVS is better
        if avg_speedup > 1.5:
            print("\n✅ cuVS is significantly faster for 1000D data!")
        elif avg_speedup > 1.0:
            print("\n⚠️ cuVS is marginally faster for 1000D data")
        else:
            print("\n❌ PyTorch is faster for 1000D data (cuVS overhead)")


if __name__ == "__main__":
    # Run in rapids environment
    import sys
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd() if 'os' in dir() else 'unknown'}")
    
    import os
    os.chdir('/home/ubuntu/devel/dire-jax')
    
    main()