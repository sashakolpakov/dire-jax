#!/bin/bash
set -e  # Exit on any error

# Script to build the documentation for DiRe-JAX

# Ensure pip packages are installed
echo "Installing required packages..."
pip install -r docs/requirements.txt

# Install the project and its dependencies
echo "Installing project dependencies..."
pip install -e .

# Build the documentation
echo "Building documentation..."
cd docs
make clean
make html

echo "Documentation built successfully. Open docs/build/html/index.html to view."