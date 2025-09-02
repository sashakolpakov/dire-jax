# test_dire.py

"""
Tests for the DiRe main class.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest

import numpy as np
from sklearn.datasets import make_blobs

from dire_jax import DiRe


class TestDiRe(unittest.TestCase):
    """Tests for the DiRe class functionality."""

    def setUp(self):
        """Set up test data for each test case."""
        # Create a very small test dataset for quick testing
        self.n_samples = 50  # Reduced from 100
        self.n_features = 20  # Reduced from 50
        self.n_centers = 3
        self.random_state = 42

        # Generate blob dataset with known centers
        self.X, self.y = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.n_centers,
            random_state=self.random_state,
        )

        # Standard reducer parameters
        self.n_components = 2
        self.n_neighbors = 5  # Reduced from 8
        self.sample_size = 3  # Reduced from 4
        self.max_iter_layout = 5  # Reduced from 10 for faster tests

    def test_init(self):
        """Test initialization of DiRe."""
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            sample_size=self.sample_size,
            max_iter_layout=self.max_iter_layout,
        )

        self.assertEqual(reducer.n_components, self.n_components)
        self.assertEqual(reducer.n_neighbors, self.n_neighbors)
        self.assertEqual(reducer.sample_size, self.sample_size)
        self.assertEqual(reducer.max_iter_layout, self.max_iter_layout)

    def test_fit_transform(self):
        """Test fit_transform method."""
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            sample_size=self.sample_size,
            max_iter_layout=self.max_iter_layout,
        )

        # Apply fit_transform
        layout = reducer.fit_transform(self.X)

        # Check output shape
        self.assertEqual(layout.shape[0], self.n_samples)
        self.assertEqual(layout.shape[1], self.n_components)

        # Check output is finite
        self.assertTrue(np.isfinite(layout).all())

        # Check cluster preservation (basic test)
        # Points from the same original cluster should be closer in the embedding
        for cluster_id in range(self.n_centers):
            cluster_mask = self.y == cluster_id

            # Skip if there's only 1 point in the cluster
            if np.sum(cluster_mask) <= 1:
                continue

            # Get layout for points in this cluster
            cluster_points = layout[cluster_mask]

            # Compute mean distance within cluster
            within_dists = []
            for i, point_i in enumerate(cluster_points):
                for j in range(i + 1, len(cluster_points)):
                    within_dists.append(np.linalg.norm(point_i - cluster_points[j]))
            mean_within_dist = np.mean(within_dists)

            # Compute mean distance to points in other clusters
            other_mask = ~cluster_mask
            other_points = layout[other_mask]

            between_dists = []
            for _, point_i in enumerate(cluster_points):
                for point_j in other_points:
                    between_dists.append(np.linalg.norm(point_i - point_j))
            mean_between_dist = np.mean(between_dists)

            # Within-cluster distances should be smaller than between-cluster distances
            # Note: This is a probabilistic test and might occasionally fail
            self.assertLess(mean_within_dist, mean_between_dist)

    def test_different_embedding_types(self):
        """Test different embedding initialization methods."""
        embedding_types = ["random", "pca", "spectral"]

        for embed_type in embedding_types:
            reducer = DiRe(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                init=embed_type,
                sample_size=self.sample_size,
                max_iter_layout=self.max_iter_layout,
            )

            # Apply fit_transform
            layout = reducer.fit_transform(self.X)

            # Check output shape
            self.assertEqual(layout.shape[0], self.n_samples)
            self.assertEqual(layout.shape[1], self.n_components)

            # Check output is finite
            self.assertTrue(np.isfinite(layout).all())


if __name__ == "__main__":
    unittest.main()
