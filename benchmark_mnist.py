#!/usr/bin/env python3

"""
Benchmark DIRE on full MNIST dataset from OpenML.
Tests both speed and clustering quality.
"""

import numpy as np
import torch
import time
import gc
from sklearn.datasets import fetch_openml
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import backends
from dire_jax import DiReCuVS, DiRePyTorch, create_dire

def load_mnist_full():
    """Load full MNIST dataset from OpenML."""
    print("Loading full MNIST dataset from OpenML...")
    print("(This may take a minute on first run to download)")
    
    # Fetch MNIST from OpenML (dataset ID 554)
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    
    X = mnist['data'].astype(np.float32)
    y = mnist['target'].astype(int)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    print(f"  Dataset shape: {X.shape}")
    print(f"  Number of classes: {len(np.unique(y))}")
    print(f"  Data type: {X.dtype}")
    print(f"  Memory size: {X.nbytes / 1e9:.2f} GB")
    
    return X, y

def evaluate_clustering(X_2d, y_true, n_clusters=10):
    """Evaluate clustering quality on 2D embedding."""
    print("\nEvaluating clustering quality...")
    
    # Perform k-means clustering on 2D embedding
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_2d)
    
    # Calculate metrics
    silhouette = silhouette_score(X_2d, y_true, sample_size=10000)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # Calculate cluster purity
    from scipy.stats import mode
    purity = 0
    for cluster_id in range(n_clusters):
        mask = (y_pred == cluster_id)
        if mask.sum() > 0:
            cluster_labels = y_true[mask]
            most_common = mode(cluster_labels, keepdims=False)[0]
            purity += (cluster_labels == most_common).sum()
    purity = purity / len(y_true)
    
    print(f"  Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
    print(f"  Adjusted Rand Index: {ari:.3f} (higher is better, range [-1, 1])")
    print(f"  Cluster Purity: {purity:.3f} (higher is better, range [0, 1])")
    
    return {
        'silhouette': silhouette,
        'ari': ari,
        'purity': purity
    }

def visualize_embedding(X_2d, y, title="DIRE MNIST Embedding", save_path=None):
    """Visualize 2D embedding with true labels."""
    print("\nCreating visualization...")
    
    # Subsample for visualization if needed
    if len(X_2d) > 10000:
        idx = np.random.choice(len(X_2d), 10000, replace=False)
        X_plot = X_2d[idx]
        y_plot = y[idx]
    else:
        X_plot = X_2d
        y_plot = y
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], 
                         c=y_plot, cmap='tab10', 
                         s=1, alpha=0.5)
    plt.colorbar(scatter, label='Digit Class')
    plt.title(title)
    plt.xlabel('DIRE Component 1')
    plt.ylabel('DIRE Component 2')
    
    # Add legend for digit classes
    for digit in range(10):
        mask = y_plot == digit
        if mask.sum() > 0:
            center = X_plot[mask].mean(axis=0)
            plt.text(center[0], center[1], str(digit), 
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', alpha=0.7))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def benchmark_dire(X, y, backend='auto', max_iter=30):
    """Benchmark DIRE on dataset."""
    print(f"\n{'='*70}")
    print(f"Benchmarking DIRE with {backend.upper()} backend")
    print(f"{'='*70}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Create reducer
    if backend == 'cuvs':
        reducer = DiReCuVS(
            n_components=2,
            n_neighbors=30,
            max_iter_layout=max_iter,
            verbose=True,
            use_cuvs=True,
            use_cuml=True,
            cuvs_index_type='auto'
        )
    elif backend == 'pytorch':
        reducer = DiRePyTorch(
            n_components=2,
            n_neighbors=30,
            max_iter_layout=max_iter,
            verbose=True
        )
    else:  # auto
        reducer = create_dire(backend='auto', 
                             n_components=2,
                             n_neighbors=30,
                             max_iter_layout=max_iter,
                             verbose=True)
    
    # Time the embedding
    print(f"\nComputing 2D embedding...")
    t_start = time.time()
    
    # Detailed timing
    t0 = time.time()
    X_2d = reducer.fit_transform(X)
    total_time = time.time() - t0
    
    # Convert to numpy if needed
    if torch.is_tensor(X_2d):
        X_2d = X_2d.cpu().numpy()
    
    print(f"\n✅ Embedding complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Throughput: {len(X)/total_time:.0f} points/sec")
    
    # Memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"  Peak GPU memory: {peak_memory/1e9:.2f} GB")
    
    return X_2d, total_time

def main():
    print("=" * 80)
    print("MNIST BENCHMARK - SPEED AND QUALITY TEST")
    print("=" * 80)
    
    # Environment info
    print("\nEnvironment:")
    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
    
    try:
        import cuvs
        import cuml
        print(f"  cuVS: ✅ Available")
        print(f"  cuML: ✅ v{cuml.__version__}")
        has_rapids = True
    except ImportError:
        print("  RAPIDS: ❌ Not available")
        has_rapids = False
    
    # Load MNIST
    X, y = load_mnist_full()
    
    # Benchmark different backends
    results = {}
    
    # 1. PyTorch backend (if dataset not too large)
    if len(X) <= 100000:
        print("\n" + "="*80)
        print("TEST 1: PyTorch Backend")
        print("="*80)
        X_2d_pytorch, time_pytorch = benchmark_dire(X, y, backend='pytorch', max_iter=30)
        quality_pytorch = evaluate_clustering(X_2d_pytorch, y)
        visualize_embedding(X_2d_pytorch, y, 
                          title="MNIST - PyTorch Backend",
                          save_path="mnist_pytorch.png")
        results['pytorch'] = {
            'time': time_pytorch,
            'quality': quality_pytorch
        }
    
    # 2. cuVS/cuML backend
    if has_rapids:
        print("\n" + "="*80)
        print("TEST 2: Full GPU Pipeline (cuVS + cuML)")
        print("="*80)
        X_2d_cuvs, time_cuvs = benchmark_dire(X, y, backend='cuvs', max_iter=30)
        quality_cuvs = evaluate_clustering(X_2d_cuvs, y)
        visualize_embedding(X_2d_cuvs, y,
                          title="MNIST - cuVS/cuML Backend", 
                          save_path="mnist_cuvs.png")
        results['cuvs'] = {
            'time': time_cuvs,
            'quality': quality_cuvs
        }
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\n{'Backend':<15} {'Time (s)':<12} {'Silhouette':<12} {'ARI':<12} {'Purity':<12}")
    print("-" * 63)
    
    for backend, res in results.items():
        print(f"{backend.upper():<15} {res['time']:<12.1f} "
              f"{res['quality']['silhouette']:<12.3f} "
              f"{res['quality']['ari']:<12.3f} "
              f"{res['quality']['purity']:<12.3f}")
    
    # Compare backends if both available
    if 'pytorch' in results and 'cuvs' in results:
        speedup = results['pytorch']['time'] / results['cuvs']['time']
        quality_diff = (results['cuvs']['quality']['silhouette'] - 
                       results['pytorch']['quality']['silhouette'])
        
        print(f"\n🚀 Performance Comparison:")
        print(f"  Speed: cuVS is {speedup:.1f}x faster")
        print(f"  Quality: {'Similar' if abs(quality_diff) < 0.05 else 'Different'} "
              f"(Δ silhouette = {quality_diff:+.3f})")
    
    print("\n✅ Benchmark complete! Check mnist_*.png for visualizations.")


if __name__ == "__main__":
    import os
    os.chdir('/home/ubuntu/devel/dire-jax')
    main()