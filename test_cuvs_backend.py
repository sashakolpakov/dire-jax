#!/usr/bin/env python3

"""
Test cuVS backend for DIRE and compare with PyTorch backend.
"""

import numpy as np
import time
from sklearn.datasets import make_blobs
import torch

# Try to import both backends
try:
    from dire_jax.dire_cuvs import DiReCuVS, CUVS_AVAILABLE
except ImportError:
    print("Could not import DiReCuVS")
    CUVS_AVAILABLE = False

from dire_jax.dire_pytorch import DiRePyTorch


def test_backend(backend_class, X, backend_name="Backend", **kwargs):
    """Test a specific backend."""
    print(f"\n{backend_name}:")
    print("-" * 50)
    
    try:
        t0 = time.time()
        
        # Initialize
        reducer = backend_class(
            n_components=2,
            n_neighbors=30,
            init='random',
            max_iter_layout=10,  # Few iterations for testing
            verbose=True,
            **kwargs
        )
        
        # Fit and transform
        embedding = reducer.fit_transform(X)
        
        dt = time.time() - t0
        
        print(f"✅ Success!")
        print(f"  Time: {dt:.2f}s")
        print(f"  Throughput: {len(X)/dt:.0f} points/sec")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding range: [{embedding.min():.2f}, {embedding.max():.2f}]")
        
        # Memory usage
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"  Peak GPU memory: {peak_mb:.0f} MB")
            torch.cuda.empty_cache()
        
        return embedding, dt
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_knn_accuracy(X, k=30):
    """Compare k-NN accuracy between backends."""
    print("\nComparing k-NN accuracy:")
    print("-" * 50)
    
    # PyTorch k-NN (exact)
    print("Computing exact k-NN with PyTorch...")
    reducer_torch = DiRePyTorch(n_neighbors=k, verbose=False)
    reducer_torch._find_ab_params()
    reducer_torch._compute_knn(X)
    knn_exact = reducer_torch._knn_indices
    
    if CUVS_AVAILABLE:
        # cuVS k-NN (approximate)
        print("Computing approximate k-NN with cuVS...")
        reducer_cuvs = DiReCuVS(n_neighbors=k, verbose=False, use_cuvs=True)
        reducer_cuvs._find_ab_params()
        reducer_cuvs._compute_knn(X)
        knn_approx = reducer_cuvs._knn_indices
        
        # Compute recall
        recall_scores = []
        for i in range(len(X)):
            exact_set = set(knn_exact[i])
            approx_set = set(knn_approx[i])
            recall = len(exact_set & approx_set) / len(exact_set)
            recall_scores.append(recall)
        
        avg_recall = np.mean(recall_scores)
        print(f"\nk-NN Recall (cuVS vs exact): {avg_recall*100:.1f}%")
        
        if avg_recall < 0.9:
            print("⚠️  Warning: Low recall! Consider tuning cuVS parameters.")
    else:
        print("cuVS not available for comparison")


def test_scaling():
    """Test scaling with different dataset sizes."""
    print("\n" + "=" * 70)
    print("SCALING TEST: cuVS vs PyTorch")
    print("=" * 70)
    
    sizes = [10000, 50000, 100000, 250000]
    dims = [100, 500]
    
    results = {}
    
    for dim in dims:
        print(f"\n\nDimension: {dim}D")
        print("=" * 50)
        
        for n in sizes:
            print(f"\nTesting {n:,} points in {dim}D:")
            
            # Generate data
            X, _ = make_blobs(
                n_samples=n,
                n_features=dim,
                centers=50,
                random_state=42
            )
            
            # Test PyTorch
            _, time_torch = test_backend(
                DiRePyTorch, X, 
                backend_name="PyTorch Backend"
            )
            
            # Test cuVS if available
            if CUVS_AVAILABLE:
                _, time_cuvs = test_backend(
                    DiReCuVS, X,
                    backend_name="cuVS Backend",
                    use_cuvs=True
                )
                
                if time_torch and time_cuvs:
                    speedup = time_torch / time_cuvs
                    print(f"\n🚀 Speedup: {speedup:.1f}x")
                    results[(n, dim)] = speedup
            
            # Clean up
            torch.cuda.empty_cache()
    
    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SPEEDUP SUMMARY (cuVS vs PyTorch):")
        print("=" * 70)
        print(f"{'Points':<10} {'Dims':<10} {'Speedup':<10}")
        print("-" * 30)
        for (n, d), speedup in results.items():
            print(f"{n:<10} {d:<10} {speedup:.1f}x")


def test_index_types():
    """Test different cuVS index types."""
    if not CUVS_AVAILABLE:
        print("cuVS not available, skipping index type test")
        return
    
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT cuVS INDEX TYPES")
    print("=" * 70)
    
    # Generate test data
    X, _ = make_blobs(n_samples=100000, n_features=128, centers=100, random_state=42)
    
    index_types = ['ivf_flat', 'ivf_pq', 'cagra']
    
    for idx_type in index_types:
        print(f"\nTesting {idx_type}:")
        _, dt = test_backend(
            DiReCuVS, X,
            backend_name=f"cuVS ({idx_type})",
            use_cuvs=True,
            cuvs_index_type=idx_type
        )
        
        if dt:
            print(f"  Index type: {idx_type}")
            print(f"  Time: {dt:.2f}s")


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING cuVS BACKEND FOR DIRE")
    print("=" * 70)
    
    # Check environment
    print("\nEnvironment:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  cuVS available: {CUVS_AVAILABLE}")
    
    if not CUVS_AVAILABLE:
        print("\n⚠️  cuVS not available!")
        print("To enable cuVS backend, install RAPIDS:")
        print("  conda install -c rapidsai -c conda-forge rapids=25.08")
        print("\nFalling back to PyTorch-only tests...")
    
    # Basic functionality test
    print("\n" + "=" * 70)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 70)
    
    # Generate test data
    X_test, y_test = make_blobs(
        n_samples=50000,
        n_features=100,
        centers=10,
        random_state=42
    )
    
    # Test PyTorch backend
    embed_torch, _ = test_backend(
        DiRePyTorch, X_test,
        backend_name="PyTorch Backend"
    )
    
    # Test cuVS backend if available
    if CUVS_AVAILABLE:
        embed_cuvs, _ = test_backend(
            DiReCuVS, X_test,
            backend_name="cuVS Backend",
            use_cuvs=True
        )
        
        # Compare k-NN accuracy
        compare_knn_accuracy(X_test[:10000])  # Subset for speed
    
    # Scaling tests
    if CUVS_AVAILABLE:
        test_scaling()
        test_index_types()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)