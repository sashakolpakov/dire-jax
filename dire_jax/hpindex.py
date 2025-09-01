# hpindex.py

"""
A JAX-based implementation for efficient k-nearest neighbors.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union, Callable

#
# Double precision support
#
jax.config.update("jax_enable_x64", True)

class HPIndex:

    """
    A high-performance kNN index that uses batching/tiling to efficiently handle
    large datasets with limited memory usage. Supports multiple distance metrics
    including Lp norms, cosine distance, and custom user-defined metrics.
    
    The index uses JAX for GPU acceleration and memory-efficient tiled computation
    to handle datasets that don't fit in memory.
    
    Supported distance metrics:
    - 'lp': p-th power of Lp distance (default p=2 for squared L2)
    - 'l1': Manhattan/L1 distance  
    - 'linf': Chebyshev/L-infinity distance
    - 'cosine': Cosine distance
    - Custom callable: User-defined metric function with signature my_metric(y_batch, x, **kwargs)
    
    Examples:
        # Default L2 squared distance
        indices, distances = HPIndex.knn_tiled(database, queries, k=10)
        
        # L1 Manhattan distance
        indices, distances = HPIndex.knn_tiled(database, queries, k=10, metric='l1')
        
        # L3 distance (3rd power)
        indices, distances = HPIndex.knn_tiled(database, queries, k=10, metric='lp', p=3)
        
        # Cosine distance
        indices, distances = HPIndex.knn_tiled(database, queries, k=10, metric='cosine')
        
        # Custom metric with kwargs
        def my_metric(y_batch, x, scale=1.0):
            diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
            return scale * jnp.sum(diff**2, axis=2)
        
        indices, distances = HPIndex.knn_tiled(database, queries, k=10, 
                                               metric=my_metric, scale=2.0)
    """

    def __init__(self):
        pass

    @staticmethod
    def knn_tiled(x, y, k=5, metric: Union[str, Callable]='lp', x_tile_size=8192, y_batch_size=1024, dtype=jnp.float64, **metric_kwargs):
        """
        Find k-nearest neighbors using tiled computation for memory efficiency.
        
        This method tiles both database and query points to handle large datasets
        that don't fit in memory. It uses JAX for GPU acceleration and supports
        multiple distance metrics.

        Args:
            x: (n, d) array of database points to search in
            y: (m, d) array of query points to find neighbors for
            k: number of nearest neighbors to find for each query point
            metric: distance metric to use. Options:
                - 'lp': p-th power of Lp distance (default p=2 for squared L2)
                - 'l1': Manhattan/L1 distance
                - 'linf': Chebyshev/L-infinity distance  
                - 'cosine': Cosine distance
                - callable: custom metric function with signature my_metric(y_batch, x, **kwargs)
            x_tile_size: number of database points to process in each tile (default 8192)
            y_batch_size: number of query points to process in each batch (default 1024)
            dtype: floating-point precision (jnp.float32 or jnp.float64)
            **metric_kwargs: additional metric parameters:
                - p: power for 'lp' metric (default 2, must be >= 2)

        Returns:
            tuple: (indices, distances) where:
                - indices: (m, k) array of indices of k nearest neighbors for each query point
                - distances: (m, k) array of distances to k nearest neighbors for each query point
            
        Raises:
            ValueError: If metric is unknown or if p < 2 for 'lp' metric
            
        Examples:
            # Find 10 nearest neighbors using default L2 squared distance
            indices, distances = HPIndex.knn_tiled(database, queries, k=10)
            
            # Use L1 Manhattan distance
            indices, distances = HPIndex.knn_tiled(database, queries, k=5, metric='l1')
            
            # Use L3 distance (3rd power)
            indices, distances = HPIndex.knn_tiled(database, queries, k=5, metric='lp', p=3)
            
            # Use cosine distance  
            indices, distances = HPIndex.knn_tiled(database, queries, k=5, metric='cosine')
            
            # Use custom metric
            def weighted_euclidean(y_batch, x, weights):
                diff = y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]
                return jnp.sum(weights * diff**2, axis=2)
            
            weights = jnp.array([1.0, 2.0, 0.5, ...])  # feature weights
            indices, distances = HPIndex.knn_tiled(database, queries, k=5, 
                                                   metric=weighted_euclidean, weights=weights)
            
        Note:
            For very large datasets, adjust x_tile_size and y_batch_size to fit
            your available memory. Smaller values use less memory but may be slower.
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

        # Handle different metric types
        if callable(metric):
            # Handle custom callable metric - create a partial function with kwargs bound
            metric_with_kwargs = partial(metric, **metric_kwargs) if metric_kwargs else metric
            indices, distances = HPIndex._knn_tiled_jit_custom(
                x, y, k, x_tile_size, y_batch_size,
                num_y_batches, y_remainder, num_x_tiles, n_x, metric_with_kwargs, dtype
            )
            return indices, distances
        elif metric == 'lp':
            p = metric_kwargs.get('p', 2)
            if p < 2:
                raise ValueError("For lp metric, p must be >= 2")
            indices, distances = HPIndex._knn_tiled_jit_lp(
                x, y, k, x_tile_size, y_batch_size,
                num_y_batches, y_remainder, num_x_tiles, n_x, p, dtype
            )
            return indices, distances
        else:
            # Call the JIT-compiled implementation with concrete values for builtin metrics
            indices, distances = HPIndex._knn_tiled_jit(
                x, y, k, x_tile_size, y_batch_size,
                num_y_batches, y_remainder, num_x_tiles, n_x, metric, dtype
            )
            return indices, distances

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10))
    def _knn_tiled_jit(x, y, k, x_tile_size, y_batch_size,
                       num_y_batches, y_remainder, num_x_tiles, n_x, metric, dtype=jnp.float64):
        """
        JIT-compiled implementation of tiled KNN search with concrete parameters.
        
        This function is JIT-compiled for performance and handles the core tiled
        computation logic. It processes query points in batches and database points
        in tiles to maintain memory efficiency.

        Args:
            x: (n_x, d) database points array
            y: (n_y, d) query points array  
            k: number of nearest neighbors to find
            x_tile_size: size of each database tile
            y_batch_size: size of each query batch
            num_y_batches: number of full query batches
            y_remainder: number of remaining query points after full batches
            num_x_tiles: number of database tiles
            n_x: total number of database points
            metric: distance metric string ('l1', 'linf', 'cosine')
            dtype: floating-point data type

        Returns:
            tuple: (indices, distances) where:
                - indices: (n_y, k) array of neighbor indices
                - distances: (n_y, k) array of distances to neighbors
                
        Note:
            This function is used internally by knn_tiled() and should not be
            called directly. Use HPIndex.knn_tiled() instead.
        """
        n_y, d_y = y.shape
        _, d_x = x.shape

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
                tile_distances = _compute_batch_distances(y_batch, x_tile, dtype, metric)

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
                tile_distances = _compute_batch_distances(padded_y, x_tile, dtype, metric)

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

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10))
    def _knn_tiled_jit_lp(x, y, k, x_tile_size, y_batch_size,
                          num_y_batches, y_remainder, num_x_tiles, n_x, p, dtype=jnp.float64):
        """
        JIT-compiled implementation of tiled KNN search using Lp metric.
        
        This specialized version handles the Lp distance metric where the p parameter
        is passed as a static argument for JIT compilation efficiency. It computes
        the p-th power of the Lp distance (without taking roots for p >= 2).

        Args:
            x: (n_x, d) database points array
            y: (n_y, d) query points array  
            k: number of nearest neighbors to find
            x_tile_size: size of each database tile
            y_batch_size: size of each query batch
            num_y_batches: number of full query batches
            y_remainder: number of remaining query points after full batches
            num_x_tiles: number of database tiles
            n_x: total number of database points
            p: power for Lp norm (must be >= 2)
            dtype: floating-point data type

        Returns:
            tuple: (indices, distances) where:
                - indices: (n_y, k) array of neighbor indices
                - distances: (n_y, k) array of p-th power of Lp distances
                
        Note:
            This function is used internally by knn_tiled() for the 'lp' metric
            and should not be called directly. Use HPIndex.knn_tiled() instead.
        """
        n_y, d_y = y.shape
        _, d_x = x.shape

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

                # Compute distances between y_batch and x_tile using Lp metric
                tile_distances = _compute_batch_distances_lp(y_batch, x_tile, dtype, p)

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

                # Compute distances between padded_y and x_tile using Lp metric
                tile_distances = _compute_batch_distances_lp(padded_y, x_tile, dtype, p)

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

    @staticmethod
    def _knn_tiled_jit_custom(x, y, k, x_tile_size, y_batch_size,
                              num_y_batches, y_remainder, num_x_tiles, n_x, metric_fn, dtype=jnp.float64):
        """
        Implementation of tiled KNN search using custom metric function.
        
        This function handles user-defined custom metrics. The custom metric is JIT-compiled
        within this function for efficiency.
        
        Args:
            x: (n_x, d) database points array
            y: (n_y, d) query points array  
            k: number of nearest neighbors to find
            x_tile_size: size of each database tile
            y_batch_size: size of each query batch
            num_y_batches: number of full query batches
            y_remainder: number of remaining query points after full batches
            num_x_tiles: number of database tiles
            n_x: total number of database points
            metric_fn: custom metric function (will be JIT-compiled internally)
            dtype: floating-point data type

        Returns:
            tuple: (indices, distances) where:
                - indices: (n_y, k) array of neighbor indices
                - distances: (n_y, k) array of distances from custom metric
                
        Note:
            This function is used internally by knn_tiled() for custom callable metrics
            and should not be called directly. Use HPIndex.knn_tiled() instead.
        """
        # JIT compile the custom metric function
        jit_metric_fn = jax.jit(metric_fn)
        n_y, d_y = y.shape
        _, d_x = x.shape

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
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances between y_batch and x_tile using custom metric
                tile_distances = jit_metric_fn(y_batch, x_tile)

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

            # Process x tiles for the remainder batch
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

                # Compute distances between padded_y and x_tile using custom metric
                tile_distances = jit_metric_fn(padded_y, x_tile)

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


# Built-in distance functions

@partial(jax.jit, static_argnums=(2, 3))
def _compute_batch_distances_lp(y_batch, x, dtype=jnp.float64, p=2):
    """
    Compute the p-th power of Lp distances between query points and database points.
    
    For p >= 2, we compute the p-th power directly to avoid expensive root operations.
    When p=2, this gives squared L2 distances. The Lp distance is defined as:
    ||y - x||_p = (sum(|y_i - x_i|^p))^(1/p), but we return (sum(|y_i - x_i|^p)).

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points
        dtype: floating-point data type for computation
        p: power for Lp norm (must be >= 2)

    Returns:
        jnp.ndarray: (batch_size, n) array of p-th power of Lp distances
        
    Note:
        For p=1, use _compute_batch_distances_l1() instead.
        For p=infinity, use _compute_batch_distances_linf() instead.
    """
    # Compute absolute differences
    diff = jnp.abs(y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :])
    
    # Compute p-th power of differences and sum
    dists_p = jnp.sum(diff**p, axis=2)
    
    return dists_p

@partial(jax.jit, static_argnums=(2,))
def _compute_batch_distances_l1(y_batch, x, dtype=jnp.float64):
    """
    Compute the L1 (Manhattan) distances between query points and database points.
    
    The L1 distance is defined as: ||y - x||_1 = sum(|y_i - x_i|).
    This is also known as Manhattan distance or taxicab distance.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points
        dtype: floating-point data type for computation

    Returns:
        jnp.ndarray: (batch_size, n) array of L1 distances
    """
    # Compute absolute differences and sum
    diff = jnp.abs(y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :])
    dists = jnp.sum(diff, axis=2)
    
    return dists

@partial(jax.jit, static_argnums=(2,))
def _compute_batch_distances_linf(y_batch, x, dtype=jnp.float64):
    """
    Compute the L-infinity (Chebyshev/max) distances between query and database points.
    
    The L-infinity distance is defined as: ||y - x||_∞ = max(|y_i - x_i|).
    This is also known as Chebyshev distance or maximum distance.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points
        dtype: floating-point data type for computation

    Returns:
        jnp.ndarray: (batch_size, n) array of L-infinity distances
    """
    # Compute absolute differences and take max
    diff = jnp.abs(y_batch[:, jnp.newaxis, :] - x[jnp.newaxis, :, :])
    dists = jnp.max(diff, axis=2)
    
    return dists

@partial(jax.jit, static_argnums=(2,))
def _compute_batch_distances_cosine(y_batch, x, dtype=jnp.float64):
    """
    Compute the cosine distances between query points and database points.
    
    Cosine distance is defined as: 1 - cos(θ) = 1 - (y·x)/(||y|| ||x||),
    where θ is the angle between vectors y and x. This distance is in [0, 2],
    with 0 for identical directions and 2 for opposite directions.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points
        dtype: floating-point data type for computation

    Returns:
        jnp.ndarray: (batch_size, n) array of cosine distances
        
    Note:
        Small epsilon (1e-8) is added to norms to prevent division by zero
        for zero vectors.
    """
    # Compute norms
    y_norm = jnp.linalg.norm(y_batch, axis=1, keepdims=True)
    x_norm = jnp.linalg.norm(x, axis=1, keepdims=True)
    
    # Normalize vectors
    y_normalized = y_batch / (y_norm + 1e-8)
    x_normalized = x / (x_norm + 1e-8)
    
    # Compute cosine similarity
    cosine_sim = jnp.dot(y_normalized, x_normalized.T)
    
    # Convert to cosine distance
    cosine_dist = 1.0 - cosine_sim
    
    return cosine_dist

# Built-in metrics dictionary
BUILTIN_METRICS = {
    'lp': _compute_batch_distances_lp,
    'l1': _compute_batch_distances_l1,
    'linf': _compute_batch_distances_linf,
    'cosine': _compute_batch_distances_cosine
}

def _compute_batch_distances(y_batch, x, dtype=jnp.float64, metric='lp'):
    """
    Compute distances between a batch of query points and database points.
    
    This function serves as a dispatcher to the appropriate distance computation
    function based on the specified metric. It is used internally by the JIT-compiled
    tiled KNN implementations.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points
        dtype: floating-point data type for computation
        metric: distance metric to use. Must be one of:
            - 'l1': Manhattan/L1 distance
            - 'linf': Chebyshev/L-infinity distance
            - 'cosine': Cosine distance
            
    Returns:
        jnp.ndarray: (batch_size, n) array of distances between each query point
                     and each database point
                     
    Raises:
        ValueError: If the specified metric is not supported
        
    Note:
        This function does not handle the 'lp' metric directly - that is handled
        by _knn_tiled_jit_lp() which calls _compute_batch_distances_lp() directly
        with the p parameter. Custom callable metrics are handled separately
        by _knn_tiled_jit_custom() and do not use this dispatcher function.
    """
    if metric not in BUILTIN_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available metrics: {list(BUILTIN_METRICS.keys())}")
    
    distance_fn = BUILTIN_METRICS[metric]
    return distance_fn(y_batch, x, dtype)
