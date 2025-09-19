# DiRe-JAX Documentation

This directory contains the Sphinx documentation for the DiRe-JAX package.

## JAX Implementation

DiRe-JAX is a pure JAX implementation optimized for:
- Excellent CPU performance with JIT compilation
- GPU acceleration when JAX is installed with CUDA support
- TPU support for cloud-based computation
- Memory-efficient chunking for large datasets (>65K points)
- Mixed precision arithmetic (MPA) for enhanced performance
- Optimized kernel caching to minimize recompilation overhead

### Hardware Acceleration
- For GPU acceleration, JAX needs specific installation: [JAX GPU instructions](https://github.com/google/jax#installation)
- For TPU support: [JAX TPU instructions](https://github.com/google/jax#tpu-tpu-vm)

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