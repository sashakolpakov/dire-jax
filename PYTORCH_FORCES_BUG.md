# Critical Bug in PyTorch Implementation: Force Computation

## The Problem

The current PyTorch/PyKeOps implementation has a fundamental bug in force computation that causes poor quality embeddings with collapsed clusters.

### Current (INCORRECT) Implementation

The current code in `dire_pytorch.py` applies both attraction AND repulsion forces to ALL pairs of points:

```python
# WRONG: Applies to ALL pairs
att_kernel = 1.0 / (1.0 + a * (1/dist)^(2b))  # Attraction for ALL pairs
rep_kernel = -1.0 / (1.0 + a * dist^(2b))     # Repulsion for ALL pairs
force_coeff = att_kernel + rep_kernel         # Combined forces on ALL pairs
```

### Correct Algorithm (as in JAX/UMAP)

The DIRE/UMAP algorithm should:
1. Apply **attraction forces ONLY between k-NN neighbors**
2. Apply **repulsion forces to random samples** (or all pairs with proper weighting)

## Test Results

Testing with 500 samples, 5 clusters:

| Implementation | Avg Cluster Separation | Time |
|---------------|------------------------|------|
| Current (broken) | 0.204 | 0.46s |
| Fixed (k-NN attraction) | 2.129 | 3.66s |
| **Improvement** | **10.43x better** | |

## The Fix Required

1. **Compute k-NN graph first** (currently missing!)
2. **Apply attraction ONLY to k-NN neighbors**
3. **Apply repulsion separately** (random sampling or weighted all-pairs)

## Code Changes Needed

### 1. Add k-NN computation
```python
def _compute_knn(self, X):
    """Compute k-nearest neighbors graph."""
    # ... compute k-NN indices
    self._knn_indices = knn_indices
```

### 2. Fix force computation
```python
def _compute_forces_correct(self, positions, alpha=1.0):
    forces = torch.zeros_like(positions)
    
    # Attraction: ONLY for k-NN neighbors
    for i in range(n_samples):
        neighbor_ids = self._knn_indices[i]
        # Apply attraction to neighbors only
        
    # Repulsion: random samples or all-pairs
    # Apply repulsion forces separately
    
    return forces
```

## Impact

This bug causes:
- Collapsed clusters (everything attracts everything)
- Poor separation between different classes
- Results very different from the JAX reference implementation

## Temporary Workaround

Until fixed, users should:
1. Use the JAX implementation for quality results
2. Or limit PyTorch backend to very small datasets where the difference is less noticeable

## Full Working Fix

A complete working fix is available in `dire_pytorch_fixed.py` which shows:
- Proper k-NN graph computation
- Correct force application (attraction to k-NN only)
- 10x better cluster separation