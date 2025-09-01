# DiRe-JAX Documentation

This directory contains the Sphinx documentation for the DiRe-JAX package.

## Backend Options

DiRe-JAX supports multiple backends:

### JAX Backend (default)
- Best for TPUs and CPU processing
- For GPU acceleration, JAX needs specific installation: [JAX GPU instructions](https://github.com/google/jax#installation)
- For TPU support: [JAX TPU instructions](https://github.com/google/jax#tpu-tpu-vm)

### PyTorch/PyKeOps Backend
- 100x+ faster on CUDA GPUs for datasets <2M points
- Install with: `pip install dire-jax[pytorch]`
- Uses all-pairs force computation without k-NN graphs

## Building the Documentation

### Prerequisites

To build the documentation, you need to have Sphinx and the required extensions installed:

```bash
pip install sphinx sphinx_rtd_theme
```

### Building

From this directory, run:

```bash
# On Unix/Linux/macOS
make html

# On Windows
make.bat html
```

The built documentation will be available in the `build/html` directory.

## Documentation Structure

- `source/` - Source files for the documentation
  - `api/` - API reference documentation
  - `conf.py` - Sphinx configuration
  - `*.rst` - Documentation pages

## Updating the Documentation

When updating the package, please also update the documentation to reflect any changes.

### Note on JAX Hardware Acceleration

The documentation includes important information about JAX GPU/TPU support. Please ensure this information is preserved when rebuilding the documentation.

In `installation.rst`, there is a section about JAX GPU/TPU support that explains how to install JAX with hardware acceleration. This information is crucial for users who want to leverage GPU or TPU resources for better performance.