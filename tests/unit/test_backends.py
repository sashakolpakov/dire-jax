# test_backends.py

"""
Tests for both JAX and PyTorch backends.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import numpy as np
from sklearn.datasets import make_blobs

from dire_jax import DiRe


class TestBackends(unittest.TestCase):
    """Tests for both JAX and PyTorch backends."""

    def setUp(self):
        """Set up test data for each test case."""
        self.n_samples = 50
        self.n_features = 20
        self.n_centers = 3
        self.random_state = 42

        # Generate test dataset
        self.X, self.y = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.n_centers,
            random_state=self.random_state,
        )

        # Test parameters
        self.n_components = 2
        self.n_neighbors = 5
        self.sample_size = 3
        self.max_iter_layout = 5

    def test_jax_backend(self):
        """Test JAX backend functionality."""
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            sample_size=self.sample_size,
            max_iter_layout=self.max_iter_layout,
        )

        layout = reducer.fit_transform(self.X)
        
        # Basic validation
        self.assertEqual(layout.shape[0], self.n_samples)
        self.assertEqual(layout.shape[1], self.n_components)
        self.assertTrue(np.isfinite(layout).all())

    def test_pytorch_backend_import(self):
        """Test PyTorch backend import and initialization."""
        try:
            from dire_jax import DiRePyTorch
            
            reducer = DiRePyTorch(
                dimension=self.n_components,
                n_neighbors=self.n_neighbors,
                max_iter_layout=self.max_iter_layout,
            )
            
            # Basic initialization test
            self.assertEqual(reducer.dimension, self.n_components)
            self.assertEqual(reducer.n_neighbors, self.n_neighbors)
            
        except ImportError:
            self.skipTest("PyTorch backend not available (missing torch/pykeops)")

    def test_pytorch_backend_functionality(self):
        """Test PyTorch backend fit_transform."""
        try:
            from dire_jax import DiRePyTorch
            
            reducer = DiRePyTorch(
                dimension=self.n_components,
                n_neighbors=self.n_neighbors,
                max_iter_layout=self.max_iter_layout,
            )

            layout = reducer.fit_transform(self.X)
            
            # Basic validation
            self.assertEqual(layout.shape[0], self.n_samples)
            self.assertEqual(layout.shape[1], self.n_components)
            self.assertTrue(np.isfinite(layout).all())
            
        except ImportError:
            self.skipTest("PyTorch backend not available (missing torch/pykeops)")


if __name__ == "__main__":
    unittest.main()