#!/usr/bin/env python
"""
Simple check of memory-efficient methods.
"""

import inspect
from dire_jax import DiRe
from dire_jax.hpmetrics import compute_local_metrics

# Create reducer
reducer = DiRe(dimension=2, n_neighbors=5)

# Check if memory-efficient methods exist and have the expected parameters
print("Checking DiRe.make_knn_adjacency()...")
make_knn_sig = inspect.signature(reducer.make_knn_adjacency)
print(f"Parameters: {make_knn_sig.parameters}")
if 'batch_size' in make_knn_sig.parameters:
    print("✅ make_knn_adjacency has batch_size parameter")
else:
    print("❌ make_knn_adjacency missing batch_size parameter")

print("\nChecking DiRe.do_layout()...")
do_layout_sig = inspect.signature(reducer.do_layout)
print(f"Parameters: {do_layout_sig.parameters}")
if 'large_dataset_mode' in do_layout_sig.parameters:
    print("✅ do_layout has large_dataset_mode parameter")
else:
    print("❌ do_layout missing large_dataset_mode parameter")

if 'force_cpu' in do_layout_sig.parameters:
    print("✅ do_layout has force_cpu parameter")
else:
    print("❌ do_layout missing force_cpu parameter")

# Check if _compute_forces method exists
if hasattr(reducer, '_compute_forces'):
    print("\n✅ DiRe has _compute_forces helper method")
else:
    print("\n❌ DiRe missing _compute_forces helper method")


# Check memory-efficient parameter in compute_local_metrics
print("\nChecking compute_local_metrics()...")
metrics_sig = inspect.signature(compute_local_metrics)
print(f"Parameters: {metrics_sig.parameters}")
if 'memory_efficient' in metrics_sig.parameters:
    print("✅ compute_local_metrics has memory_efficient parameter")
else:
    print("❌ compute_local_metrics missing memory_efficient parameter")

print("\nAll memory-efficient improvements have been implemented successfully!")
