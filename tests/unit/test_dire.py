# test_dire.py

"""
Tests for the DiRe main class.
"""

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

    def test_l1_metric(self):
        """Test DiRe with L1 (Manhattan) distance metric."""
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric="l1",
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

        # Verify metric was set correctly
        self.assertEqual(reducer.metric, "l1")

    def test_lp_metric_p2(self):
        """Test DiRe with Lp distance metric where p=2 (squared L2)."""
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric="lp",
            p=2,
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

        # Verify metric was set correctly
        self.assertEqual(reducer.metric, "lp")
        self.assertEqual(reducer.metric_kwargs["p"], 2)

    def test_cosine_metric(self):
        """Test DiRe with cosine distance metric."""
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric="cosine",
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

        # Verify metric was set correctly
        self.assertEqual(reducer.metric, "cosine")

    def test_linf_metric(self):
        """Test DiRe with L-infinity (Chebyshev/max) distance metric."""
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric="linf",
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

        # Verify metric was set correctly
        self.assertEqual(reducer.metric, "linf")

    def test_custom_metric(self):
        """Test DiRe with a custom distance metric."""
        import jax.numpy as jnp
        
        def weighted_euclidean(y_batch, x, weights):
            """Custom weighted Euclidean distance metric."""
            diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
            distances = jnp.sum(weights * diff**2, axis=2)
            # Ensure we return the same dtype as the input arrays
            return distances.astype(x.dtype)
        
        # Create feature weights - emphasize some features more than others
        feature_weights = jnp.ones(self.n_features) * 0.1  # Base weight
        feature_weights = feature_weights.at[:5].set(2.0)  # Higher weight for first 5 features
        
        reducer = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric=weighted_euclidean,
            weights=feature_weights,
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

        # Verify metric was set correctly (should be the function itself)
        self.assertEqual(reducer.metric, weighted_euclidean)
        
        # Check that weights are in metric_kwargs
        self.assertIn('weights', reducer.metric_kwargs)
        np.testing.assert_array_equal(reducer.metric_kwargs['weights'], feature_weights)

    def test_custom_metric_vs_builtin(self):
        """Test that custom metric gives different results from builtin metrics."""
        import jax.numpy as jnp
        
        def weighted_euclidean_vs_builtin(y_batch, x, weights):
            """Custom weighted Euclidean distance - different from uniform weighting."""
            diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
            distances = jnp.sum(weights * diff**2, axis=2)
            return distances.astype(x.dtype)
        
        # Test builtin lp metric (p=2, which gives L2 squared with uniform weighting)
        reducer_builtin = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric="lp",
            p=2,
            sample_size=self.sample_size,
            max_iter_layout=self.max_iter_layout,
        )
        layout_builtin = reducer_builtin.fit_transform(self.X)

        # Test custom metric with non-uniform weights
        feature_weights = jnp.ones(self.n_features)
        feature_weights = feature_weights.at[:5].set(5.0)  # Much higher weight for first 5 features
        
        reducer_custom = DiRe(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric=weighted_euclidean_vs_builtin,
            weights=feature_weights,
            sample_size=self.sample_size,
            max_iter_layout=self.max_iter_layout,
        )
        layout_custom = reducer_custom.fit_transform(self.X)

        # Both should have correct shape and be finite
        self.assertEqual(layout_builtin.shape, (self.n_samples, self.n_components))
        self.assertEqual(layout_custom.shape, (self.n_samples, self.n_components))
        self.assertTrue(np.isfinite(layout_builtin).all())
        self.assertTrue(np.isfinite(layout_custom).all())

        # Results should be different (non-uniform weighting changes the relative distances)
        # Normalize layouts to account for different scales/orientations
        layout_builtin_norm = layout_builtin - layout_builtin.mean(axis=0)
        layout_builtin_norm = layout_builtin_norm / (layout_builtin_norm.std(axis=0) + 1e-8)
        layout_custom_norm = layout_custom - layout_custom.mean(axis=0)
        layout_custom_norm = layout_custom_norm / (layout_custom_norm.std(axis=0) + 1e-8)

        # Check that they're not identical
        diff = np.linalg.norm(layout_builtin_norm - layout_custom_norm)
        self.assertGreater(diff, 0.01, "Custom weighted metric should produce different results from builtin uniform weighting")

    def test_all_metrics_consistency(self):
        """Test that all metrics produce valid embeddings and maintain relative cluster structure."""
        import jax.numpy as jnp
        
        def hinge_distance(y_batch, x, threshold=0.5, power=2.0):
            """Custom hinge distance metric - applies penalty only beyond threshold."""
            diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
            l2_dist = jnp.sqrt(jnp.sum(diff**2, axis=2))
            # Hinge loss: max(0, distance - threshold)^power
            hinged = jnp.maximum(0, l2_dist - threshold)
            distances = hinged**power
            return distances.astype(x.dtype)
        
        metrics_configs = [
            {"metric": "l1"},
            {"metric": "lp", "p": 2},
            {"metric": "linf"},
            {"metric": "cosine"},
            {"metric": hinge_distance, "threshold": 0.8, "power": 1.5},
        ]
        
        layouts = {}
        
        # Test each metric
        for config in metrics_configs:
            metric = config["metric"]
            if callable(metric):
                metric_name = metric.__name__
                # Add any additional params to name
                extra_params = {k: v for k, v in config.items() if k != "metric"}
                if extra_params:
                    param_str = "_".join(f"{k}{v}" for k, v in extra_params.items())
                    metric_name += f"_{param_str}"
            else:
                metric_name = metric
                if "p" in config:
                    metric_name += f"_p{config['p']}"
                
            reducer = DiRe(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                sample_size=self.sample_size,
                max_iter_layout=self.max_iter_layout,
                **config,
            )

            # Apply fit_transform
            layout = reducer.fit_transform(self.X)
            layouts[metric_name] = layout

            # Basic shape and validity checks
            self.assertEqual(layout.shape[0], self.n_samples)
            self.assertEqual(layout.shape[1], self.n_components)
            self.assertTrue(np.isfinite(layout).all())

            # Check that different clusters maintain some separation
            # This is a basic test that the embedding preserves some structure
            for cluster_id in range(self.n_centers):
                cluster_mask = self.y == cluster_id
                
                # Skip if there's only 1 point in the cluster
                if np.sum(cluster_mask) <= 1:
                    continue

                # Get layout for points in this cluster
                cluster_points = layout[cluster_mask]
                
                # The standard deviation within cluster should be reasonable
                # (not zero, but not too large either)
                cluster_std = np.std(cluster_points, axis=0)
                self.assertTrue(np.all(cluster_std > 1e-6))  # Not collapsed
                self.assertTrue(np.all(cluster_std < 10))    # Not too spread

        # Verify all metrics produced different results (they should, given different distance measures)
        metric_names = list(layouts.keys())
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                layout1 = layouts[metric_names[i]]
                layout2 = layouts[metric_names[j]]
                
                # Layouts should be different (not identical)
                # We normalize them first to account for different scales/orientations
                layout1_norm = layout1 - layout1.mean(axis=0)
                layout1_norm = layout1_norm / (layout1_norm.std(axis=0) + 1e-8)
                layout2_norm = layout2 - layout2.mean(axis=0) 
                layout2_norm = layout2_norm / (layout2_norm.std(axis=0) + 1e-8)
                
                # Check that they're not too similar (allowing for some numerical precision issues)
                diff = np.linalg.norm(layout1_norm - layout2_norm)
                self.assertGreater(diff, 0.1, 
                    f"Layouts for {metric_names[i]} and {metric_names[j]} are too similar")


if __name__ == "__main__":
    unittest.main()
