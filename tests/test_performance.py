#!/usr/bin/env python3
"""
Performance benchmarking and scaling tests for DiRe layout algorithm.

This test suite can be used to:
1. Benchmark performance across different configurations
2. Test scaling behavior with dataset size
3. Validate optimization effectiveness
4. Track performance regression

Usage:
    python test_performance.py                    # Run default benchmarks
    python test_performance.py --quick           # Quick test mode
    python test_performance.py --detailed        # Detailed benchmarks
    python test_performance.py --mpa-only        # Test only MPA configurations
    python test_performance.py --scaling         # Focus on scaling tests
"""

import argparse
import json
import time
import os
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
from jax import device_put, random
import functools

# Import DiRe
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dire_jax.dire import DiRe

# Disable compilation logging for cleaner output by default
jax.config.update("jax_log_compiles", False)


class PerformanceBenchmark:
    """Performance benchmarking suite for DiRe."""
    
    def __init__(self, save_results=True, results_dir="performance_results"):
        self.save_results = save_results
        self.results_dir = results_dir
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'jax_version': jax.__version__,
            'devices': [str(device) for device in jax.devices()],
            'benchmarks': {}
        }
        
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
    
    def create_test_dataset(self, n_samples=2000, n_features=50, n_clusters=5, random_state=42):
        """Create a realistic test dataset with clusters."""
        np.random.seed(random_state)
        
        cluster_size = n_samples // n_clusters
        data = []
        
        for i in range(n_clusters):
            # Create cluster center
            center = np.random.randn(n_features) * 2
            # Create cluster points around center
            cluster_data = np.random.randn(cluster_size, n_features) * 0.5 + center
            data.append(cluster_data)
        
        # Add remaining points if any
        remaining = n_samples - n_clusters * cluster_size
        if remaining > 0:
            remaining_data = np.random.randn(remaining, n_features) * 3
            data.append(remaining_data)
        
        return np.vstack(data)
    
    def benchmark_configuration(self, data, config_name, **dire_params):
        """Benchmark a specific DiRe configuration."""
        print(f"\n--- Benchmarking: {config_name} ---")
        print(f"Parameters: {dire_params}")
        
        # Default parameters
        default_params = {
            'n_components': 2,
            'n_neighbors': 16,
            'init': 'random',
            'max_iter_layout': 20,
            'verbose': False,
            'random_state': 42
        }
        default_params.update(dire_params)
        
        # Create DiRe instance
        dire = DiRe(**default_params)
        
        # Warm-up run (smaller dataset to compile functions)
        if data.shape[0] > 500:
            warmup_data = data[:500]
            dire_warmup = DiRe(**{**default_params, 'max_iter_layout': 2})
            dire_warmup.fit_transform(warmup_data)
        
        # Benchmark the full run
        start_time = time.time()
        embedding = dire.fit_transform(data)
        total_time = time.time() - start_time
        
        # Calculate metrics
        n_samples, n_features = data.shape
        max_iter = default_params['max_iter_layout']
        
        result = {
            'total_time': total_time,
            'time_per_iteration': total_time / max_iter,
            'time_per_sample': total_time / n_samples,
            'samples_per_second': n_samples / total_time,
            'data_shape': data.shape,
            'embedding_shape': embedding.shape,
            'parameters': default_params
        }
        
        print(f"Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Time per iteration: {result['time_per_iteration']:.4f}s")
        print(f"  Time per sample: {result['time_per_sample']:.6f}s")
        print(f"  Samples per second: {result['samples_per_second']:.1f}")
        
        return result
    
    def test_mpa_performance(self, test_sizes=None):
        """Compare performance with and without MPA."""
        if test_sizes is None:
            test_sizes = [5000, 10000]  # More realistic sizes
        
        print("\n" + "="*60)
        print("MPA PERFORMANCE COMPARISON")
        print("="*60)
        
        results = {}
        
        for size in test_sizes:
            print(f"\n=== Dataset size: {size} samples ===")
            data = self.create_test_dataset(n_samples=size, n_features=30)
            
            # Test without MPA
            result_no_mpa = self.benchmark_configuration(
                data, f"No MPA (size={size})", 
                mpa=False, max_iter_layout=15
            )
            
            # Test with MPA
            result_mpa = self.benchmark_configuration(
                data, f"With MPA (size={size})", 
                mpa=True, max_iter_layout=15
            )
            
            # Calculate speedup
            speedup = result_no_mpa['total_time'] / result_mpa['total_time']
            
            results[size] = {
                'no_mpa': result_no_mpa,
                'mpa': result_mpa,
                'speedup': speedup
            }
            
            print(f"\n  MPA Speedup: {speedup:.2f}x")
        
        return results
    
    def test_scaling_performance(self, test_sizes=None, test_features=None):
        """Test performance scaling with dataset size and dimensionality."""
        if test_sizes is None:
            test_sizes = [1000, 5000, 10000, 50000]  # Realistic large dataset sizes
        if test_features is None:
            test_features = [20, 50, 100, 200]  # Include higher dimensional data
        
        print("\n" + "="*60)
        print("SCALING PERFORMANCE TEST")
        print("="*60)
        
        results = {'size_scaling': {}, 'feature_scaling': {}}
        
        # Test scaling with dataset size (fixed features)
        print(f"\n--- Scaling with dataset size (features=50) ---")
        for size in test_sizes:
            print(f"\nTesting {size} samples...")
            data = self.create_test_dataset(n_samples=size, n_features=50)
            
            # Use fewer iterations for very large datasets to keep test time reasonable
            iterations = 5 if size >= 50000 else (8 if size >= 10000 else 10)
            
            result = self.benchmark_configuration(
                data, f"Size scaling (n={size})", 
                mpa=True, max_iter_layout=iterations
            )
            
            results['size_scaling'][size] = result
        
        # Test scaling with feature dimensionality (fixed size)
        print(f"\n--- Scaling with feature dimensionality (samples=5000) ---")
        for features in test_features:
            print(f"\nTesting {features} features...")
            data = self.create_test_dataset(n_samples=5000, n_features=features)
            
            # Use fewer iterations for high-dimensional data
            iterations = 6 if features >= 200 else 8
            
            result = self.benchmark_configuration(
                data, f"Feature scaling (d={features})", 
                mpa=True, max_iter_layout=iterations
            )
            
            results['feature_scaling'][features] = result
        
        return results
    
    def test_initialization_methods(self, data_size=1000):
        """Compare different initialization methods."""
        print("\n" + "="*60)
        print("INITIALIZATION METHODS COMPARISON")
        print("="*60)
        
        data = self.create_test_dataset(n_samples=data_size, n_features=40)
        results = {}
        
        init_methods = ['random', 'pca', 'spectral']
        
        for init_method in init_methods:
            try:
                result = self.benchmark_configuration(
                    data, f"Init: {init_method}", 
                    init=init_method, mpa=True, max_iter_layout=15
                )
                results[init_method] = result
            except Exception as e:
                print(f"Failed to test {init_method}: {e}")
                results[init_method] = {'error': str(e)}
        
        return results
    
    def analyze_results(self, all_results):
        """Analyze and summarize benchmark results."""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        # MPA Analysis
        if 'mpa_comparison' in all_results:
            print(f"\n📊 MPA Performance Impact:")
            mpa_results = all_results['mpa_comparison']
            speedups = [result['speedup'] for result in mpa_results.values()]
            avg_speedup = np.mean(speedups)
            print(f"  Average MPA speedup: {avg_speedup:.2f}x")
            print(f"  MPA speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
        
        # Scaling Analysis
        if 'scaling' in all_results:
            scaling = all_results['scaling']
            
            if 'size_scaling' in scaling:
                print(f"\n📈 Dataset Size Scaling:")
                size_results = scaling['size_scaling']
                sizes = sorted(size_results.keys())
                
                for i, size in enumerate(sizes[1:], 1):
                    prev_size = sizes[i-1]
                    size_ratio = size / prev_size
                    time_ratio = size_results[size]['total_time'] / size_results[prev_size]['total_time']
                    efficiency = size_ratio / time_ratio
                    
                    print(f"  {prev_size} → {size} samples: {efficiency:.2f} efficiency (1.0 = linear)")
            
            if 'feature_scaling' in scaling:
                print(f"\n🔢 Feature Dimensionality Scaling:")
                feature_results = scaling['feature_scaling']
                features = sorted(feature_results.keys())
                
                for i, feat in enumerate(features[1:], 1):
                    prev_feat = features[i-1]
                    feat_ratio = feat / prev_feat
                    time_ratio = feature_results[feat]['total_time'] / feature_results[prev_feat]['total_time']
                    efficiency = feat_ratio / time_ratio
                    
                    print(f"  {prev_feat} → {feat} features: {efficiency:.2f} efficiency")
        
        # Initialization Analysis
        if 'initialization' in all_results:
            print(f"\n🎯 Initialization Methods:")
            init_results = all_results['initialization']
            
            valid_results = {k: v for k, v in init_results.items() if 'error' not in v}
            if valid_results:
                sorted_methods = sorted(valid_results.items(), key=lambda x: x[1]['total_time'])
                fastest_method, fastest_time = sorted_methods[0][0], sorted_methods[0][1]['total_time']
                
                print(f"  Fastest: {fastest_method} ({fastest_time:.3f}s)")
                for method, result in sorted_methods[1:]:
                    slowdown = result['total_time'] / fastest_time
                    print(f"  {method}: {result['total_time']:.3f}s ({slowdown:.2f}x slower)")
    
    def save_benchmark_results(self, results):
        """Save benchmark results to JSON file."""
        if not self.save_results:
            return
        
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy({**self.results, 'benchmarks': results})
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
    
    def run_benchmarks(self, mode='default'):
        """Run the full benchmark suite."""
        print("=" * 60)
        print(f"DIRE PERFORMANCE BENCHMARK SUITE - {mode.upper()} MODE")
        print("=" * 60)
        print(f"JAX version: {jax.__version__}")
        print(f"Available devices: {jax.devices()}")
        
        all_results = {}
        
        if mode in ['default', 'detailed', 'mpa-only']:
            # MPA comparison
            if mode == 'quick':
                test_sizes = [1000]
            elif mode == 'large-scale':
                test_sizes = [10000, 50000]
            else:
                test_sizes = [5000, 10000]
            all_results['mpa_comparison'] = self.test_mpa_performance(test_sizes)
        
        if mode in ['default', 'detailed', 'scaling', 'large-scale']:
            # Scaling tests
            if mode == 'quick':
                size_range = [1000, 5000]
                feat_range = [20, 50]
            elif mode == 'large-scale':
                size_range = [5000, 10000, 50000, 100000]
                feat_range = [50, 100, 200]
            else:
                size_range = [1000, 5000, 10000]
                feat_range = [20, 50, 100]
            all_results['scaling'] = self.test_scaling_performance(size_range, feat_range)
        
        if mode in ['detailed']:
            # Initialization comparison
            all_results['initialization'] = self.test_initialization_methods(1000)
        
        # Analyze and summarize results
        self.analyze_results(all_results)
        
        # Save results
        self.save_benchmark_results(all_results)
        
        print(f"\n🚀 Benchmark suite completed successfully!")
        print("="*60)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='DiRe Performance Benchmark Suite')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick benchmark with smaller datasets')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed benchmark including all initialization methods')
    parser.add_argument('--mpa-only', action='store_true',
                       help='Test only MPA performance comparison')
    parser.add_argument('--scaling', action='store_true',
                       help='Focus on scaling performance tests')
    parser.add_argument('--large-scale', action='store_true',
                       help='Run large-scale tests with 10k-100k samples (takes longer)')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save results to file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable JAX compilation logging')
    
    args = parser.parse_args()
    
    # Set mode based on arguments
    if args.detailed:
        mode = 'detailed'
    elif args.mpa_only:
        mode = 'mpa-only'
    elif args.scaling:
        mode = 'scaling'
    elif args.large_scale:
        mode = 'large-scale'
    elif args.quick:
        mode = 'quick'
    else:
        mode = 'default'
    
    # Enable verbose logging if requested
    if args.verbose:
        jax.config.update("jax_log_compiles", True)
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(save_results=not args.no_save)
    
    # Run benchmarks
    results = benchmark.run_benchmarks(mode)
    
    return results


if __name__ == "__main__":
    main()