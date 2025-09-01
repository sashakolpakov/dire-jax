"""
A JAX-based implementation of efficient k-nearest neighbors.
"""

from functools import partial
import jax
import jax.numpy as jnp


# Separate JIT-compiled distance functions for different metrics
@jax.jit
def _compute_batch_distances_lp(y_batch, x):
    """Compute L2 (Euclidean) squared distances."""
    x_norm = jnp.sum(x**2, axis=1)
    y_norm = jnp.sum(y_batch**2, axis=1)
    xy = jnp.dot(y_batch, x.T)
    dists = y_norm[:, jnp.newaxis] + x_norm[jnp.newaxis, :] - 2 * xy
    return jnp.clip(dists, 0, jnp.finfo(jnp.float32).max)

@jax.jit
def _compute_batch_distances_l1(y_batch, x):
    """Compute L1 (Manhattan) distances."""
    diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
    return jnp.sum(jnp.abs(diff), axis=2)

@jax.jit
def _compute_batch_distances_linf(y_batch, x):
    """Compute L∞ (Chebyshev) distances."""
    diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
    return jnp.max(jnp.abs(diff), axis=2)

@jax.jit
def _compute_batch_distances_cosine(y_batch, x):
    """Compute cosine distances (1 - cosine similarity)."""
    # Normalize vectors
    y_norm = jnp.linalg.norm(y_batch, axis=1, keepdims=True)
    x_norm = jnp.linalg.norm(x, axis=1, keepdims=True)
    
    y_normalized = y_batch / (y_norm + 1e-10)
    x_normalized = x / (x_norm.T + 1e-10)
    
    # Compute cosine similarity
    cos_sim = jnp.dot(y_normalized, x_normalized.T)
    
    # Convert to distance (1 - similarity), clipped to [0, 2]
    return jnp.clip(1.0 - cos_sim, 0, 2)


class HPIndex:

    """
    A kernelized kNN index that uses batching / tiling to efficiently handle
    large datasets with limited memory usage.
    """

    def __init__(self):
        pass

    @staticmethod
    def knn_tiled(x, y, k=5, x_tile_size=8192, y_batch_size=1024, metric="lp", **metric_kwargs):
        """
        Advanced implementation that tiles both database and query points.
        This wrapper handles the dynamic aspects before calling the JIT-compiled
        function.

        Args:
            x: (n, d) array of database points
            y: (m, d) array of query points
            k: number of nearest neighbors
            x_tile_size: size of database tiles
            y_batch_size: size of query batches
            metric: distance metric ("lp" for L2/Euclidean, "l1" for Manhattan, "linf" for Chebyshev, "cosine" for cosine distance)
            **metric_kwargs: additional parameters (e.g., p for Lp norm, currently unused as p=2 is hardcoded)

        Returns:
            (m, k) array of indices of nearest neighbors
        """
        n_x, _ = x.shape
        n_y, _ = y.shape

        # Ensure batch sizes aren't larger than the data dimensions
        x_tile_size = min(x_tile_size, n_x)
        y_batch_size = min(y_batch_size, n_y)

        # Calculate batching parameters
        num_y_batches = n_y // y_batch_size
        y_remainder = n_y % y_batch_size
        num_x_tiles = (n_x + x_tile_size - 1) // x_tile_size

        # Call the appropriate JIT-compiled implementation based on metric
        if metric == "lp" or metric == "l2":
            return HPIndex._knn_tiled_lp_jit(
                x, y, k, x_tile_size, y_batch_size,
                num_y_batches, y_remainder, num_x_tiles, n_x
            )
        elif metric == "l1":
            return HPIndex._knn_tiled_l1_jit(
                x, y, k, x_tile_size, y_batch_size,
                num_y_batches, y_remainder, num_x_tiles, n_x
            )
        elif metric == "linf":
            return HPIndex._knn_tiled_linf_jit(
                x, y, k, x_tile_size, y_batch_size,
                num_y_batches, y_remainder, num_x_tiles, n_x
            )
        elif metric == "cosine":
            return HPIndex._knn_tiled_cosine_jit(
                x, y, k, x_tile_size, y_batch_size,
                num_y_batches, y_remainder, num_x_tiles, n_x
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _knn_tiled_lp_jit(x, y, k, x_tile_size, y_batch_size,
                          num_y_batches, y_remainder, num_x_tiles, n_x):
        """JIT-compiled implementation for Lp metric."""
        n_y, d_y = y.shape
        _, d_x = x.shape

        # Initialize results
        all_indices = jnp.zeros((n_y, k), dtype=jnp.int32)
        all_distances = jnp.ones((n_y, k)) * jnp.finfo(jnp.float32).max

        # Define the scan function for processing y batches
        def process_y_batch(carry, y_batch_idx):
            curr_indices, curr_distances = carry

            # Get current batch of query points
            y_start = y_batch_idx * y_batch_size
            y_batch = jax.lax.dynamic_slice(y, (y_start, 0), (y_batch_size, d_y))

            # Initialize batch results
            batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            batch_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            # Define the scan function for processing x tiles within a y batch
            def process_x_tile(carry, x_tile_idx):
                batch_idx, batch_dist = carry

                # Get current tile of database points - use fixed size slices
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances using Lp metric
                tile_distances = _compute_batch_distances_lp(y_batch, x_tile)

                # Mask out invalid indices
                valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                # Adjust indices to account for tile offset
                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                # Merge with previous results
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                # Sort and get top k
                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            # Process all x tiles for this y batch
            (batch_indices, batch_distances), _ = jax.lax.scan(
                process_x_tile, (batch_indices, batch_distances), jnp.arange(num_x_tiles)
            )

            # Update overall results
            curr_indices = jax.lax.dynamic_update_slice(curr_indices, batch_indices, (y_start, 0))
            curr_distances = jax.lax.dynamic_update_slice(curr_distances, batch_distances, (y_start, 0))

            return (curr_indices, curr_distances), None

        # Process all full y batches
        (all_indices, all_distances), _ = jax.lax.scan(
            process_y_batch, (all_indices, all_distances), jnp.arange(num_y_batches)
        )

        # Handle remainder
        def handle_y_remainder(indices, distances):
            y_start = num_y_batches * y_batch_size
            remainder_y = jax.lax.dynamic_slice(y, (y_start, 0), (y_remainder, d_y))
            padded_y = jnp.pad(remainder_y, ((0, y_batch_size - y_remainder), (0, 0)))

            remainder_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            remainder_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            def process_x_tile_remainder(carry, x_tile_idx):
                batch_idx, batch_dist = carry
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                tile_distances = _compute_batch_distances_lp(padded_y, x_tile)

                x_valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    x_valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            (remainder_indices, remainder_distances), _ = jax.lax.scan(
                process_x_tile_remainder, (remainder_indices, remainder_distances), jnp.arange(num_x_tiles)
            )

            valid_remainder_indices = remainder_indices[:y_remainder]
            indices = jax.lax.dynamic_update_slice(indices, valid_remainder_indices, (y_start, 0))
            return indices, distances

        all_indices, all_distances = jax.lax.cond(
            y_remainder > 0,
            lambda args: handle_y_remainder(*args),
            lambda args: args,
            (all_indices, all_distances)
        )

        return all_indices, all_distances

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _knn_tiled_l1_jit(x, y, k, x_tile_size, y_batch_size,
                          num_y_batches, y_remainder, num_x_tiles, n_x):
        """JIT-compiled implementation for L1 metric - identical structure but uses _compute_batch_distances_l1."""
        # Same implementation as _knn_tiled_lp_jit but with _compute_batch_distances_l1
        n_y, d_y = y.shape
        _, d_x = x.shape
        all_indices = jnp.zeros((n_y, k), dtype=jnp.int32)
        all_distances = jnp.ones((n_y, k)) * jnp.finfo(jnp.float32).max

        def process_y_batch(carry, y_batch_idx):
            curr_indices, curr_distances = carry
            y_start = y_batch_idx * y_batch_size
            y_batch = jax.lax.dynamic_slice(y, (y_start, 0), (y_batch_size, d_y))
            batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            batch_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            def process_x_tile(carry, x_tile_idx):
                batch_idx, batch_dist = carry
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                tile_distances = _compute_batch_distances_l1(y_batch, x_tile)

                valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)
                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            (batch_indices, batch_distances), _ = jax.lax.scan(
                process_x_tile, (batch_indices, batch_distances), jnp.arange(num_x_tiles)
            )

            curr_indices = jax.lax.dynamic_update_slice(curr_indices, batch_indices, (y_start, 0))
            curr_distances = jax.lax.dynamic_update_slice(curr_distances, batch_distances, (y_start, 0))
            return (curr_indices, curr_distances), None

        (all_indices, all_distances), _ = jax.lax.scan(
            process_y_batch, (all_indices, all_distances), jnp.arange(num_y_batches)
        )

        def handle_y_remainder(indices, distances):
            y_start = num_y_batches * y_batch_size
            remainder_y = jax.lax.dynamic_slice(y, (y_start, 0), (y_remainder, d_y))
            padded_y = jnp.pad(remainder_y, ((0, y_batch_size - y_remainder), (0, 0)))
            remainder_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            remainder_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            def process_x_tile_remainder(carry, x_tile_idx):
                batch_idx, batch_dist = carry
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                tile_distances = _compute_batch_distances_l1(padded_y, x_tile)

                x_valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    x_valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)
                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            (remainder_indices, remainder_distances), _ = jax.lax.scan(
                process_x_tile_remainder, (remainder_indices, remainder_distances), jnp.arange(num_x_tiles)
            )

            valid_remainder_indices = remainder_indices[:y_remainder]
            indices = jax.lax.dynamic_update_slice(indices, valid_remainder_indices, (y_start, 0))
            return indices, distances

        all_indices, all_distances = jax.lax.cond(
            y_remainder > 0,
            lambda args: handle_y_remainder(*args),
            lambda args: args,
            (all_indices, all_distances)
        )

        return all_indices, all_distances

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _knn_tiled_linf_jit(x, y, k, x_tile_size, y_batch_size,
                            num_y_batches, y_remainder, num_x_tiles, n_x):
        """JIT-compiled implementation for L∞ metric."""
        # Same structure, uses _compute_batch_distances_linf
        n_y, d_y = y.shape
        _, d_x = x.shape
        all_indices = jnp.zeros((n_y, k), dtype=jnp.int32)
        all_distances = jnp.ones((n_y, k)) * jnp.finfo(jnp.float32).max

        def process_y_batch(carry, y_batch_idx):
            curr_indices, curr_distances = carry
            y_start = y_batch_idx * y_batch_size
            y_batch = jax.lax.dynamic_slice(y, (y_start, 0), (y_batch_size, d_y))
            batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            batch_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            def process_x_tile(carry, x_tile_idx):
                batch_idx, batch_dist = carry
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                tile_distances = _compute_batch_distances_linf(y_batch, x_tile)

                valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)
                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            (batch_indices, batch_distances), _ = jax.lax.scan(
                process_x_tile, (batch_indices, batch_distances), jnp.arange(num_x_tiles)
            )

            curr_indices = jax.lax.dynamic_update_slice(curr_indices, batch_indices, (y_start, 0))
            curr_distances = jax.lax.dynamic_update_slice(curr_distances, batch_distances, (y_start, 0))
            return (curr_indices, curr_distances), None

        (all_indices, all_distances), _ = jax.lax.scan(
            process_y_batch, (all_indices, all_distances), jnp.arange(num_y_batches)
        )

        def handle_y_remainder(indices, distances):
            y_start = num_y_batches * y_batch_size
            remainder_y = jax.lax.dynamic_slice(y, (y_start, 0), (y_remainder, d_y))
            padded_y = jnp.pad(remainder_y, ((0, y_batch_size - y_remainder), (0, 0)))
            remainder_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            remainder_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            def process_x_tile_remainder(carry, x_tile_idx):
                batch_idx, batch_dist = carry
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                tile_distances = _compute_batch_distances_linf(padded_y, x_tile)

                x_valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    x_valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)
                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            (remainder_indices, remainder_distances), _ = jax.lax.scan(
                process_x_tile_remainder, (remainder_indices, remainder_distances), jnp.arange(num_x_tiles)
            )

            valid_remainder_indices = remainder_indices[:y_remainder]
            indices = jax.lax.dynamic_update_slice(indices, valid_remainder_indices, (y_start, 0))
            return indices, distances

        all_indices, all_distances = jax.lax.cond(
            y_remainder > 0,
            lambda args: handle_y_remainder(*args),
            lambda args: args,
            (all_indices, all_distances)
        )

        return all_indices, all_distances

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _knn_tiled_cosine_jit(x, y, k, x_tile_size, y_batch_size,
                              num_y_batches, y_remainder, num_x_tiles, n_x):
        """JIT-compiled implementation for cosine metric."""
        # Same structure, uses _compute_batch_distances_cosine
        n_y, d_y = y.shape
        _, d_x = x.shape
        all_indices = jnp.zeros((n_y, k), dtype=jnp.int32)
        all_distances = jnp.ones((n_y, k)) * jnp.finfo(jnp.float32).max

        def process_y_batch(carry, y_batch_idx):
            curr_indices, curr_distances = carry
            y_start = y_batch_idx * y_batch_size
            y_batch = jax.lax.dynamic_slice(y, (y_start, 0), (y_batch_size, d_y))
            batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            batch_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            def process_x_tile(carry, x_tile_idx):
                batch_idx, batch_dist = carry
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                tile_distances = _compute_batch_distances_cosine(y_batch, x_tile)

                valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)
                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            (batch_indices, batch_distances), _ = jax.lax.scan(
                process_x_tile, (batch_indices, batch_distances), jnp.arange(num_x_tiles)
            )

            curr_indices = jax.lax.dynamic_update_slice(curr_indices, batch_indices, (y_start, 0))
            curr_distances = jax.lax.dynamic_update_slice(curr_distances, batch_distances, (y_start, 0))
            return (curr_indices, curr_distances), None

        (all_indices, all_distances), _ = jax.lax.scan(
            process_y_batch, (all_indices, all_distances), jnp.arange(num_y_batches)
        )

        def handle_y_remainder(indices, distances):
            y_start = num_y_batches * y_batch_size
            remainder_y = jax.lax.dynamic_slice(y, (y_start, 0), (y_remainder, d_y))
            padded_y = jnp.pad(remainder_y, ((0, y_batch_size - y_remainder), (0, 0)))
            remainder_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            remainder_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            def process_x_tile_remainder(carry, x_tile_idx):
                batch_idx, batch_dist = carry
                x_start = x_tile_idx * x_tile_size
                x_tile = jax.lax.dynamic_slice(x, (x_start, 0), (x_tile_size, d_x))
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                tile_distances = _compute_batch_distances_cosine(padded_y, x_tile)

                x_valid_mask = jnp.arange(x_tile_size, dtype=jnp.int32) < x_tile_actual_size
                tile_distances = jnp.where(
                    x_valid_mask[jnp.newaxis, :], tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                tile_indices = jnp.minimum(jnp.arange(x_tile_size, dtype=jnp.int32) + x_start, n_x - 1).astype(jnp.int32)
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)
                top_k_idx = jnp.argsort(combined_distances)[:, :k]
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            (remainder_indices, remainder_distances), _ = jax.lax.scan(
                process_x_tile_remainder, (remainder_indices, remainder_distances), jnp.arange(num_x_tiles)
            )

            valid_remainder_indices = remainder_indices[:y_remainder]
            indices = jax.lax.dynamic_update_slice(indices, valid_remainder_indices, (y_start, 0))
            return indices, distances

        all_indices, all_distances = jax.lax.cond(
            y_remainder > 0,
            lambda args: handle_y_remainder(*args),
            lambda args: args,
            (all_indices, all_distances)
        )

        return all_indices, all_distances