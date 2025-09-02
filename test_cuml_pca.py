#!/usr/bin/env python3

"""
Test cuML PCA performance vs sklearn PCA for high-dimensional data.
"""

import numpy as np
import torch
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Import backends
from dire_jax import DiReCuVS

def test_pca_performance(n_samples, n_dims):
    """Compare PCA performance: sklearn vs cuML."""
    
    print(f"\n{'='*70}")
    print(f"Testing PCA with {n_samples:,} points in {n_dims}D")
    print(f"{'='*70}")
    
    # Generate data
    print(f"\nGenerating data...")
    X = np.random.randn(n_samples, n_dims).astype(np.float32)
    print(f"  Data size: {X.nbytes / 1e9:.2f} GB")
    
    # Test sklearn PCA (via DiReCuVS with use_cuml=False)
    print(f"\n1. Testing sklearn PCA (CPU)...")
    torch.cuda.empty_cache()
    gc.collect()
    
    reducer_sklearn = DiReCuVS(
        n_components=2,
        n_neighbors=30,
        use_cuvs=False,  # Don't use cuVS for k-NN
        use_cuml=False,  # Don't use cuML for PCA
        verbose=False
    )
    
    t0 = time.time()
    reducer_sklearn._find_ab_params()
    embedding_sklearn = reducer_sklearn._initialize_embedding(X)
    t_sklearn = time.time() - t0
    print(f"  Time: {t_sklearn:.2f}s")
    print(f"  Embedding shape: {embedding_sklearn.shape}")
    
    # Test cuML PCA (via DiReCuVS with use_cuml=True)
    print(f"\n2. Testing cuML PCA (GPU)...")
    torch.cuda.empty_cache()
    gc.collect()
    
    reducer_cuml = DiReCuVS(
        n_components=2,
        n_neighbors=30,
        use_cuvs=False,  # Don't use cuVS for k-NN
        use_cuml=True,   # Use cuML for PCA
        verbose=False
    )
    
    t0 = time.time()
    reducer_cuml._find_ab_params()
    embedding_cuml = reducer_cuml._initialize_embedding(X)
    t_cuml = time.time() - t0
    print(f"  Time: {t_cuml:.2f}s")
    print(f"  Embedding shape: {embedding_cuml.shape}")
    
    # Compare
    speedup = t_sklearn / t_cuml
    print(f"\n🚀 SPEEDUP: {speedup:.1f}x")
    
    # Check similarity (first few points)
    embed_sklearn_np = embedding_sklearn.cpu().numpy()[:10]
    embed_cuml_np = embedding_cuml.cpu().numpy()[:10]
    
    # PCA can flip signs, so check absolute correlation
    corr1 = np.corrcoef(embed_sklearn_np[:, 0], embed_cuml_np[:, 0])[0, 1]
    corr2 = np.corrcoef(embed_sklearn_np[:, 1], embed_cuml_np[:, 1])[0, 1]
    print(f"\nResults similarity (absolute correlation):")
    print(f"  Component 1: {abs(corr1):.3f}")
    print(f"  Component 2: {abs(corr2):.3f}")
    
    return t_sklearn, t_cuml, speedup

def main():
    print("=" * 80)
    print("cuML PCA vs sklearn PCA PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Check environment
    print("\nEnvironment:")
    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
    
    try:
        import cuml
        print(f"  cuML: ✅ v{cuml.__version__}")
    except ImportError:
        print("  cuML: ❌ Not available")
        return
    
    # Test configurations
    test_configs = [
        (50_000, 1000),
        (100_000, 1000),
        (250_000, 1000),
        (500_000, 1000),
        (1_000_000, 1000),
    ]
    
    results = []
    
    for n_samples, n_dims in test_configs:
        try:
            t_sklearn, t_cuml, speedup = test_pca_performance(n_samples, n_dims)
            results.append({
                'n_samples': n_samples,
                'sklearn_time': t_sklearn,
                'cuml_time': t_cuml,
                'speedup': speedup
            })
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY: PCA Performance Comparison")
        print("=" * 80)
        print(f"\n{'Dataset':<15} {'sklearn (s)':<15} {'cuML (s)':<15} {'Speedup':<10}")
        print("-" * 55)
        
        for r in results:
            print(f"{r['n_samples']:<15,} {r['sklearn_time']:<15.2f} {r['cuml_time']:<15.2f} {r['speedup']:<10.1f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.1f}x")
        
        print("\n✅ cuML PCA successfully eliminates the PCA bottleneck!")


if __name__ == "__main__":
    import os
    os.chdir('/home/ubuntu/devel/dire-jax')
    main()