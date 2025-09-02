# hpindex.py

"""
A JAX-based implementation for efficient k-nearest neighbors.
"""

from functools import partial
import jax
import jax.numpy as jnp

#
# Double precision support
#
jax.config.update("jax_enable_x64", True)

class HPIndex:

    """
    A kernelized kNN index that uses batching / tiling to efficiently handle
    large datasets with limited memory usage.
    """

    def __init__(self):
        pass

    @staticmethod
    def knn_tiled(x, y, k=5, x_tile_size=8192, y_batch_size=1024, dtype=jnp.float64):
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
            dtype: desired floating-point dtype (e.g., jnp.float32 or jnp.float64)

        Returns:
            (m, k) array of indices of nearest neighbors
        """
        x = x.astype(dtype)
        y = y.astype(dtype)

        n_x, _ = x.shape
        n_y, _ = y.shape

        # Ensure batch sizes aren't larger than the data dimensions
        x_tile_size = min(x_tile_size, n_x)
        y_batch_size = min(y_batch_size, n_y)

        # Calculate batching parameters
        num_y_batches = n_y // y_batch_size
        y_remainder = n_y % y_batch_size
        num_x_tiles = (n_x + x_tile_size - 1) // x_tile_size

        # Call the JIT-compiled implementation with concrete values
        return HPIndex._knn_tiled_jit(
            x, y, k, x_tile_size, y_batch_size,
            num_y_batches, y_remainder, num_x_tiles, n_x
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _knn_tiled_jit(x, y, k, x_tile_size, y_batch_size,
                       num_y_batches, y_remainder, num_x_tiles, n_x):
        """
        JIT-compiled implementation of tiled KNN with concrete batch parameters.
        """
        n_y, d_y = y.shape
        _, d_x = x.shape
        
        # Infer dtype from input
        dtype = x.dtype

        # Initialize results
        all_indices = jnp.zeros((n_y, k), dtype=jnp.int64)
        all_distances = jnp.ones((n_y, k), dtype=dtype) * jnp.finfo(dtype).max

        # Define the scan function for processing y batches
        def process_y_batch(carry, y_batch_idx):
            curr_indices, curr_distances = carry

            # Get current batch of query points
            y_start = y_batch_idx * y_batch_size
            y_batch = jax.lax.dynamic_slice(y, (y_start, 0), (y_batch_size, d_y))

            # Initialize batch results
            batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int64)
            batch_distances = jnp.ones((y_batch_size, k), dtype=dtype) * jnp.finfo(dtype).max

            # Define the scan function for processing x tiles within a y batch
            def process_x_tile(carry, x_tile_idx):
                batch_idx, batch_dist = carry

                # Get current tile of database points - use fixed size slices
                x_start = x_tile_idx * x_tile_size

                # Use a fixed size for the slice and then mask invalid values
                x_tile = jax.lax.dynamic_slice(
                    x, (x_start, 0), (x_tile_size, d_x)
                )

                # Calculate how many elements are actually valid
                # (This is now done without dynamic shapes)
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances between y_batch and x_tile
                tile_distances = _compute_batch_distances_l2(y_batch, x_tile, dtype=dtype)

                # Mask out invalid indices (those beyond the actual data)
                valid_mask = jnp.arange(x_tile_size) < x_tile_actual_size
                tile_distances = jnp.where(
                    valid_mask[jnp.newaxis, :],
                    tile_distances,
                    jnp.ones_like(tile_distances, dtype=dtype) * jnp.finfo(dtype).max
                )

                # Adjust indices to account for tile offset
                # Make sure indices are within bounds
                tile_indices = jnp.minimum(
                    jnp.arange(x_tile_size) + x_start,
                    n_x - 1  # Ensure indices don't go beyond n_x
                )
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                # Merge current tile results with previous results
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                # Sort and get top k
                top_k_idx = jnp.argsort(combined_distances)[:, :k]

                # Gather top k distances and indices
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            # Process all x tiles for this y batch
            (batch_indices, batch_distances), _ = jax.lax.scan(
                process_x_tile,
                (batch_indices, batch_distances),
                jnp.arange(num_x_tiles)
            )

            # Update overall results for this batch
            curr_indices = jax.lax.dynamic_update_slice(
                curr_indices, batch_indices, (y_start, 0)
            )
            curr_distances = jax.lax.dynamic_update_slice(
                curr_distances, batch_distances, (y_start, 0)
            )

            return (curr_indices, curr_distances), None

        # Process all full y batches
        (all_indices, all_distances), _ = jax.lax.scan(
            process_y_batch,
            (all_indices, all_distances),
            jnp.arange(num_y_batches)
        )

        # Handle y remainder with similar changes if needed
        def handle_y_remainder(indices, distances):
            y_start = num_y_batches * y_batch_size

            # Get and pad remainder batch
            remainder_y = jax.lax.dynamic_slice(y, (y_start, 0), (y_remainder, d_y))
            padded_y = jnp.pad(remainder_y, ((0, y_batch_size - y_remainder), (0, 0)))

            # Initialize remainder results
            remainder_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int64)
            remainder_distances = jnp.ones((y_batch_size, k), dtype=dtype) * jnp.finfo(dtype).max

            # Process x tiles for the remainder batch (with same fix as above)
            def process_x_tile_remainder(carry, x_tile_idx):
                batch_idx, batch_dist = carry

                # Get current tile of database points - use fixed size slices
                x_start = x_tile_idx * x_tile_size

                # Use fixed size for the slice
                x_tile = jax.lax.dynamic_slice(
                    x, (x_start, 0), (x_tile_size, d_x)
                )

                # Calculate actual valid size
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances between padded_y and x_tile
                tile_distances = _compute_batch_distances_l2(padded_y, x_tile, dtype=dtype)

                # Mask out invalid indices (both for y padding and x overflow)
                x_valid_mask = jnp.arange(x_tile_size) < x_tile_actual_size
                tile_distances = jnp.where(
                    x_valid_mask[jnp.newaxis, :],
                    tile_distances,
                    jnp.ones_like(tile_distances, dtype=dtype) * jnp.finfo(dtype).max
                )

                # Adjust indices to account for tile offset
                tile_indices = jnp.minimum(
                    jnp.arange(x_tile_size) + x_start,
                    n_x - 1  # Ensure indices don't go beyond n_x
                )
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                # Merge current tile results with previous results
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                # Sort and get top k
                top_k_idx = jnp.argsort(combined_distances)[:, :k]

                # Gather top k distances and indices
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            # Process all x tiles for the remainder batch
            (remainder_indices, remainder_distances), _ = jax.lax.scan(
                process_x_tile_remainder,
                (remainder_indices, remainder_distances),
                jnp.arange(num_x_tiles)
            )

            # Extract valid remainder results and update both arrays
            valid_i = remainder_indices[:y_remainder]
            valid_d = remainder_distances[:y_remainder]

            indices = jax.lax.dynamic_update_slice(indices, valid_i, (y_start, 0))
            distances = jax.lax.dynamic_update_slice(distances, valid_d, (y_start, 0))

            return indices, distances

        # Conditionally handle remainder to avoid issues with remainder=0
        all_indices, all_distances = jax.lax.cond(
            y_remainder > 0,
            lambda args: handle_y_remainder(*args),
            lambda args: args,
            (all_indices, all_distances)
        )

        return all_indices, all_distances


# Globally define the _compute_batch_distances_l2 function for reuse
@partial(jax.jit, static_argnums=(2,))
def _compute_batch_distances_l2(y_batch, x, dtype=jnp.float64):
    """
    Compute the squared L2 distances between a batch of query points
    and all database points.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points
        dtype: data type for computation

    Returns:
        (batch_size, n) array of squared distances
    """
    # Ensure consistent dtype
    y_batch = y_batch.astype(dtype)
    x = x.astype(dtype)
    
    # Compute squared norms using more numerically stable method
    x_norm = jnp.sum(x * x, axis=1)
    y_norm = jnp.sum(y_batch * y_batch, axis=1)

    # Compute xy term with explicit dtype
    xy = jnp.dot(y_batch, x.T, precision=jax.lax.Precision.DEFAULT)

    # Complete squared distance: ||y||² + ||x||² - 2*<y,x>
    # Use broadcasting with consistent dtype
    two = jnp.array(2.0, dtype=dtype)
    dists2 = y_norm[:, jnp.newaxis] + x_norm[jnp.newaxis, :] - two * xy
    
    # Clip to valid range for the dtype
    zero = jnp.array(0.0, dtype=dtype)
    dists2 = jnp.maximum(dists2, zero)

    return dists2