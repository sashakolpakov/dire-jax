# test_utils.py

"""
Tests for the DiRe utilities
"""


import unittest
import numpy as np
from sklearn.datasets import make_blobs
from jax import random
from dire_jax.dire_utils import display_layout, compute_local_metrics


class TestDireUtils(unittest.TestCase):
    """Tests for the utilities in dire_utils.py."""

    def setUp(self):
        """Set up test data for each test case."""
        # Create a small test dataset
        self.n_samples = 50
        self.n_features = 20
        self.n_centers = 3
        self.random_state = 42

        # Generate blob dataset with known centers
        self.data, self.labels = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.n_centers,
            random_state=self.random_state
        )

        # Create a simple 2D embedding
        np.random.seed(self.random_state)
        self.layout_2d = np.random.randn(self.n_samples, 2)

        # Create a simple 3D embedding
        self.layout_3d = np.random.randn(self.n_samples, 3)

        # Create a 4D embedding (cannot be visualized)
        self.layout_4d = np.random.randn(self.n_samples, 4)

        # Create JAX random key
        self.rng_key = random.PRNGKey(self.random_state)

    def test_display_layout_2d(self):
        """Test 2D layout visualization."""
        # Test with labels
        fig = display_layout(self.layout_2d, self.labels)
        self.assertIsNotNone(fig, "2D visualization with labels should return a figure")

        # Test without labels
        fig = display_layout(self.layout_2d, None)
        self.assertIsNotNone(fig, "2D visualization without labels should return a figure")

    def test_display_layout_3d(self):
        """Test 3D layout visualization."""
        # Test with labels
        fig = display_layout(self.layout_3d, self.labels)
        self.assertIsNotNone(fig, "3D visualization with labels should return a figure")

        # Test without labels
        fig = display_layout(self.layout_3d, None)
        self.assertIsNotNone(fig, "3D visualization without labels should return a figure")

    def test_display_layout_4d(self):
        """Test handling of higher-dimensional layouts."""
        # 4D layout should return None since it can't be visualized
        fig = display_layout(self.layout_4d, self.labels)
        self.assertIsNone(fig, "4D visualization should return None")

    def test_compute_local_metrics(self):
        """Test local metrics computation."""
        # Basic functionality test
        n_neighbors = 5
        metrics = compute_local_metrics(self.data, self.layout_2d, n_neighbors)

        # Check if metrics are returned
        self.assertIn('stress', metrics, "Stress metric should be computed")
        self.assertIn('neighbor', metrics, "Neighborhood preservation metric should be computed")

        # Check types
        self.assertIsInstance(metrics['stress'], float, "Stress metric should be a float")
        self.assertIsInstance(metrics['neighbor'], list, "Neighborhood preservation metric should be a list")
        self.assertEqual(len(metrics['neighbor']), 2, "Neighborhood preservation metric should have two values")

        # Stress should be non-negative
        self.assertGreaterEqual(metrics['stress'], 0, "Stress should be non-negative")

        # Neighborhood preservation should be between 0 and 1
        self.assertGreaterEqual(metrics['neighbor'][0], 0, "Mean neighborhood preservation should be non-negative")
        self.assertLessEqual(metrics['neighbor'][0], 1, "Mean neighborhood preservation should be at most 1")


if __name__ == '__main__':
    unittest.main()
