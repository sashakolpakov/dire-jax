# DiRe Performance Benchmarking

This directory contains comprehensive performance benchmarking tools for the DiRe layout algorithm.

## Usage

### Quick Performance Check
```bash
python test_performance.py --quick --no-save
```

### MPA Performance Comparison  
```bash
python test_performance.py --mpa-only
```

### Scaling Performance Analysis
```bash
python test_performance.py --scaling
```

### Comprehensive Benchmark
```bash
python test_performance.py --detailed
```

## Command Line Options

- `--quick`: Run fast benchmark with smaller datasets (good for CI/regression testing)
- `--detailed`: Run comprehensive benchmark including all initialization methods
- `--mpa-only`: Test only Mixed Precision Arithmetic performance impact
- `--scaling`: Focus on dataset size and feature scaling performance
- `--no-save`: Don't save results to JSON files
- `--verbose`: Enable JAX compilation logging for debugging

## Output

The benchmark produces:
1. **Console output**: Real-time performance metrics and analysis
2. **JSON results**: Detailed results saved to `performance_results/` directory (unless `--no-save`)
3. **Performance analysis**: Summary of speedups, scaling efficiency, and optimization effectiveness

## Example Results

```
📊 MPA Performance Impact:
  Average MPA speedup: 1.17x
  MPA speedup range: 1.16x - 1.18x

📈 Dataset Size Scaling:
  500 → 1000 samples: 2.30 efficiency (1.0 = linear)
  1000 → 2000 samples: 1.85 efficiency

🎯 Initialization Methods:
  Fastest: random (2.141s)
  pca: 2.489s (1.16x slower)
  spectral: 3.687s (1.72x slower)
```

## Integration with CI/CD

For continuous performance monitoring:
```bash
# Quick regression test (< 30 seconds)
python test_performance.py --quick --no-save

# Weekly comprehensive benchmark
python test_performance.py --detailed
```

## Performance Tracking

Results are automatically timestamped and include:
- JAX version and device information
- Complete parameter configurations
- Detailed timing metrics
- Scaling analysis
- Historical comparison capability

Use the saved JSON files to track performance changes over time and detect regressions.