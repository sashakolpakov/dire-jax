# hpindex.py

"""
A JAX-based implementation for efficient k-nearest neighbors.
"""

from functools import partial, lru_cache
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
        Single-kernel kNN implementation that compiles once and reuses efficiently.
        Uses a single JIT-compiled kernel with fixed tile/batch parameters.

        Args:
            x: (n, d) array of database points
            y: (m, d) array of query points
            k: number of nearest neighbors
            x_tile_size: size of database tiles
            y_batch_size: size of query batches
            dtype: desired floating-point dtype (e.g., jnp.float32 or jnp.float64)

        Returns:
            (m, k) array of indices and distances of nearest neighbors
        """
        # Get or compile the kernel for this configuration
        kernel = HPIndex._get_knn_kernel(k, x_tile_size, y_batch_size, dtype)

        # Call the compiled kernel
        return kernel(x, y)

    @staticmethod
    @lru_cache(maxsize=16)  # Cache compiled kernels for different configurations
    def _get_knn_kernel(k, x_tile_size, y_batch_size, dtype):
        """
        Get or create a cached JIT-compiled kNN kernel for the given configuration.
        This ensures we reuse compiled kernels across different datasets with same params.
        """

        @jax.jit
        def knn_kernel(x, y):
            # Ensure consistent dtypes
            x = x.astype(dtype)
            y = y.astype(dtype)

            n_x, d_x = x.shape
            n_y, d_y = y.shape

            # Pad data to tile/batch boundaries
            padded_n_x = ((n_x + x_tile_size - 1) // x_tile_size) * x_tile_size
            padded_n_y = ((n_y + y_batch_size - 1) // y_batch_size) * y_batch_size

            # Pad x if needed
            if padded_n_x > n_x:
                x_pad = jnp.full((padded_n_x - n_x, d_x), jnp.finfo(dtype).max / 2, dtype=dtype)
                x_padded = jnp.concatenate([x, x_pad], axis=0)
            else:
                x_padded = x

            # Pad y if needed
            if padded_n_y > n_y:
                y_pad = jnp.zeros((padded_n_y - n_y, d_y), dtype=dtype)
                y_padded = jnp.concatenate([y, y_pad], axis=0)
            else:
                y_padded = y

            # Calculate number of tiles/batches
            num_y_batches = padded_n_y // y_batch_size
            num_x_tiles = padded_n_x // x_tile_size

            # Initialize results
            all_indices = jnp.zeros((padded_n_y, k), dtype=jnp.int64)
            all_distances = jnp.ones((padded_n_y, k), dtype=dtype) * jnp.finfo(dtype).max

            # Get distance kernel for this dtype
            distance_kernel = _get_distance_kernel(dtype)

            # Main processing loop using scan for efficiency
            def process_y_batch(carry, y_batch_idx):
                curr_indices, curr_distances = carry
                y_start = y_batch_idx * y_batch_size
                y_batch = jax.lax.dynamic_slice(y_padded, (y_start, 0), (y_batch_size, d_y))

                batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int64)
                batch_distances = jnp.ones((y_batch_size, k), dtype=dtype) * jnp.finfo(dtype).max

                def process_x_tile(tile_carry, x_tile_idx):
                    batch_idx, batch_dist = tile_carry
                    x_start = x_tile_idx * x_tile_size
                    x_tile = jax.lax.dynamic_slice(x_padded, (x_start, 0), (x_tile_size, d_x))

                    # Compute distances
                    tile_distances = distance_kernel(y_batch, x_tile)

                    # Create tile indices
                    tile_indices = jnp.arange(x_tile_size) + x_start
                    tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                    # Merge and get top k
                    combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                    combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)
                    top_k_idx = jnp.argsort(combined_distances)[:, :k]

                    new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                    new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                    return (new_batch_idx, new_batch_dist), None

                # Process all x tiles for this y batch
                (batch_indices, batch_distances), _ = jax.lax.scan(
                    process_x_tile,
                    (batch_indices, batch_distances),
                    jnp.arange(num_x_tiles)
                )

                # Update results
                curr_indices = jax.lax.dynamic_update_slice(curr_indices, batch_indices, (y_start, 0))
                curr_distances = jax.lax.dynamic_update_slice(curr_distances, batch_distances, (y_start, 0))

                return (curr_indices, curr_distances), None

            # Process all y batches
            (all_indices, all_distances), _ = jax.lax.scan(
                process_y_batch,
                (all_indices, all_distances),
                jnp.arange(num_y_batches)
            )

            # Return only valid portion
            return all_indices[:n_y], all_distances[:n_y]

        return knn_kernel


# Globally define the _compute_batch_distances_l2 function for reuse
# Using lru_cache to avoid recompilation for different dtype combinations
@lru_cache(maxsize=8)  # Cache different dtype combinations
def _get_distance_kernel(dtype):
    """Get or create a JIT-compiled distance kernel for the given dtype."""
    @jax.jit
    def compute_distances(y_batch, x):
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

    return compute_distances

def _compute_batch_distances_l2(y_batch, x, dtype=jnp.float64):
    """
    Compute the squared L2 distances between a batch of query points
    and all database points. Uses cached kernels to avoid recompilation.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points
        dtype: data type for computation

    Returns:
        (batch_size, n) array of squared distances
    """
    # Get the cached kernel for this dtype
    distance_kernel = _get_distance_kernel(dtype)
    return distance_kernel(y_batch, x)