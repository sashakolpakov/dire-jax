#!/usr/bin/env python3

"""
Test cuVS scaling with 1000D data.
Focus on cuVS performance only since PyTorch OOMs.
"""

import numpy as np
import torch
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Import backends
from dire_jax import DiReCuVS

def format_memory(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}PB"

def test_cuvs_scaling(n_samples, n_dims, max_iter=3):
    """Test cuVS at specific scale."""
    print(f"\nTesting {n_samples:,} points in {n_dims}D")
    print("-" * 50)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Generate data
    print(f"Generating data...")
    t0 = time.time()
    X = np.random.randn(n_samples, n_dims).astype(np.float32)
    gen_time = time.time() - t0
    data_size = X.nbytes
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Data size: {format_memory(data_size)}")
    
    try:
        # Test cuVS
        print(f"\nComputing k-NN with cuVS...")
        t0 = time.time()
        
        reducer = DiReCuVS(
            n_components=2,
            n_neighbors=30,
            max_iter_layout=max_iter,
            verbose=True,
            use_cuvs=True,
            cuvs_index_type='auto'
        )
        
        # Time k-NN separately
        t_knn_start = time.time()
        reducer._find_ab_params()
        reducer._compute_knn(X)
        t_knn = time.time() - t_knn_start
        
        # Quick embedding
        print("Computing layout...")
        initial = reducer._initialize_embedding(X)
        embedding = reducer._optimize_layout(initial).cpu().numpy()
        
        total_time = time.time() - t0
        
        # Memory stats
        peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        # Results
        print(f"\n✅ SUCCESS!")
        print(f"  k-NN time: {t_knn:.1f}s ({n_samples/t_knn:.0f} points/sec)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Peak GPU memory: {format_memory(peak_bytes)}")
        print(f"  Throughput: {n_samples*n_dims*4 / t_knn / 1e9:.1f} GB/s")
        
        return {
            'n_samples': n_samples,
            'n_dims': n_dims,
            'knn_time': t_knn,
            'total_time': total_time,
            'peak_memory': peak_bytes,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'n_samples': n_samples,
            'n_dims': n_dims,
            'success': False,
            'error': str(e)
        }
    finally:
        # Clean up
        del X
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("TESTING cuVS SCALING WITH 1000-DIMENSIONAL DATA")
    print("=" * 70)
    
    # Check environment
    print("\nEnvironment:")
    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
    
    try:
        import cuvs
        import cupy
        print("  cuVS: ✅ Available")
        print(f"  CuPy: ✅ v{cupy.__version__}")
    except ImportError as e:
        print(f"  cuVS/CuPy: ❌ Not available - {e}")
        return
    
    # Test sizes - go big!
    test_configs = [
        (250_000, 1000),
        (500_000, 1000),
        (750_000, 1000),
        (1_000_000, 1000),
        (1_500_000, 1000),
        (2_000_000, 1000),
        (2_500_000, 1000),
        (3_000_000, 1000),
    ]
    
    results = []
    
    for n_samples, n_dims in test_configs:
        result = test_cuvs_scaling(n_samples, n_dims, max_iter=3)
        results.append(result)
        
        # Stop if failed or taking too long
        if not result['success']:
            print(f"\n⚠️ Stopping - cuVS failed at {n_samples:,} points")
            break
        
        if result['knn_time'] > 300:
            print(f"\n⚠️ Stopping - k-NN taking too long (>5 minutes)")
            break
    
    # Summary
    successful = [r for r in results if r['success']]
    if successful:
        print("\n" + "=" * 70)
        print("SUMMARY: cuVS SCALING FOR 1000D k-NN")
        print("=" * 70)
        print(f"{'Samples':<15} {'k-NN Time (s)':<15} {'Memory':<15} {'Throughput':<15}")
        print("-" * 60)
        
        for r in successful:
            throughput = r['n_samples'] / r['knn_time'] if 'knn_time' in r else 0
            mem = format_memory(r.get('peak_memory', 0))
            print(f"{r['n_samples']:<15,} {r['knn_time']:<15.1f} {mem:<15} {throughput:<15.0f} pts/s")
        
        max_points = max(r['n_samples'] for r in successful)
        print(f"\n✅ Successfully scaled to {max_points:,} points in 1000D!")
        
        # Estimate limits
        if successful[-1]['knn_time'] < 300:
            rate = successful[-1]['n_samples'] / successful[-1]['knn_time']
            est_5min = int(rate * 300)
            print(f"📈 Estimated capacity at 5min limit: {est_5min:,} points")


if __name__ == "__main__":
    import os
    os.chdir('/home/ubuntu/devel/dire-jax')
    main()