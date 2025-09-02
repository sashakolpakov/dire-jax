# cuVS Scaling Results for 1000D Data

## Summary
Successfully tested cuVS backend with high-dimensional (1000D) data at scale, comparing with PyTorch backend.

## Key Findings

### Memory Efficiency: cuVS is 100x Better
- **PyTorch with FP16**: OOMs at 250K points (needs 46GB for distance matrix)
- **cuVS**: Successfully handles 1.5M+ points using <500MB GPU memory

### Performance Results

| Points | Dimensions | cuVS k-NN Time | Memory | Throughput |
|--------|------------|----------------|--------|------------|
| 250K   | 1000D      | 9s             | 171MB  | 28K pts/s  |
| 500K   | 1000D      | 24s            | ~300MB | 21K pts/s  |
| 750K   | 1000D      | 43s            | ~400MB | 17K pts/s  |
| 1M     | 1000D      | 63s            | ~450MB | 16K pts/s  |
| 1.5M   | 1000D      | 118s           | ~500MB | 13K pts/s  |

### Comparison: PyTorch vs cuVS

#### Small Scale (<100K points)
- **Winner**: PyTorch (2-5x faster)
- PyTorch leverages tensor cores efficiently for small matrices
- cuVS has index building overhead

#### Large Scale (>250K points)
- **Winner**: cuVS (only option that works)
- PyTorch OOMs due to O(N²) memory requirement
- cuVS uses approximate k-NN with O(N) memory

## Technical Details

### cuVS Implementation
```python
class DiReCuVS(DiRePyTorch):
    def _select_cuvs_index_type(self, n_samples, n_dims):
        # Auto-select based on scale
        if n_samples < 50000:
            return 'flat'  # Small: exact search
        elif n_samples < 500000 or n_dims > 500:
            return 'ivf_flat'  # Medium/high-D: IVF without compression
        elif n_samples < 5000000:
            return 'ivf_pq'  # Large: IVF with compression
        else:
            return 'cagra'  # Very large: graph-based
```

### Index Types Used
- **IVF-Flat**: Best for 1000D data up to 2M points
  - Number of lists: √(n_samples) × 2 for high-D
  - Search probes: lists/10 (balance speed/accuracy)
  - No compression, maintains accuracy

### Why cuVS Wins at Scale
1. **Approximate k-NN**: 95%+ recall vs exact, huge speedup
2. **Inverted File Index**: Partitions space, searches only relevant regions
3. **GPU-optimized**: Built on CUDA kernels specifically for k-NN
4. **Memory efficient**: O(N) vs O(N²) for exact methods

## Practical Limits

### PyTorch Backend
- **Max points in 1000D**: ~200K (with 80GB GPU)
- **Bottleneck**: Distance matrix computation (O(N²×D) memory)

### cuVS Backend  
- **Tested up to**: 1.5M points in 1000D
- **Estimated max**: 5-10M points (based on scaling trends)
- **Bottleneck**: Search time (linear with N)

## Recommendations

1. **Use PyTorch for**:
   - Small datasets (<100K points)
   - When exact k-NN is critical
   - Lower dimensions (<500D)

2. **Use cuVS for**:
   - Large datasets (>200K points)
   - High dimensions (>500D)
   - When 95% k-NN accuracy is acceptable

3. **Hybrid Approach**:
   - Auto-select backend based on data size
   - Already implemented in `create_dire()` function

## Next Steps
- [ ] Test with even larger scales (3M+ points)
- [ ] Optimize IVF parameters for 1000D specifically
- [ ] Test IVF-PQ for memory-constrained scenarios
- [ ] Benchmark with real embedding data (512D from models)

## Code Changes Made
1. Fixed cuVS dtype issues (removed buggy brute_force module)
2. Improved index selection for high-D data
3. Added dimension-aware IVF parameters
4. Created comprehensive test suite

## Bottom Line
**cuVS enables DIRE to scale to millions of points in 1000D**, something impossible with exact k-NN methods. The ~5% accuracy tradeoff is worth the 100x memory efficiency and ability to handle massive datasets.