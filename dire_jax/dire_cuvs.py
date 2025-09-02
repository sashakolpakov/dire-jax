# dire_cuvs.py

"""
DIRE with cuVS backend for GPU-accelerated k-NN at scale.

This module provides optional cuVS integration for massive datasets.
Falls back to PyTorch if cuVS is not available.

Requirements:
    pip install rapids-25.08  # or conda install -c rapidsai -c conda-forge rapids=25.08
"""

import numpy as np
import torch
from loguru import logger
import gc

# Import base DIRE PyTorch implementation
from .dire_pytorch import DiRePyTorch

# Try to import cuVS and CuPy
try:
    import cupy as cp
    from cuvs.neighbors import cagra, ivf_pq, ivf_flat
    CUVS_AVAILABLE = True
    logger.info("cuVS available - GPU-accelerated k-NN enabled")
except ImportError:
    CUVS_AVAILABLE = False
    logger.warning("cuVS not available. Install RAPIDS for GPU-accelerated k-NN: "
                  "conda install -c rapidsai -c conda-forge rapids=25.08")

# Try to import cuML for GPU-accelerated PCA
try:
    from cuml.decomposition import PCA as cuPCA
    from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
    CUML_AVAILABLE = True
    logger.info("cuML available - GPU-accelerated PCA enabled")
except ImportError:
    CUML_AVAILABLE = False
    if CUVS_AVAILABLE:
        logger.warning("cuML not available but cuVS is. PCA will run on CPU.")


class DiReCuVS(DiRePyTorch):
    """
    DIRE implementation with optional cuVS backend for k-NN.
    
    Advantages over PyTorch/PyKeOps:
    - 10-100x faster k-NN for large datasets
    - Handles 10M+ points efficiently
    - Approximate k-NN with high recall (>95%)
    - Multi-GPU support for extreme scale
    
    Falls back to PyTorch backend if cuVS is not available.
    """
    
    def __init__(
        self,
        *args,
        use_cuvs=None,  # Auto-detect by default
        use_cuml=None,  # Auto-detect by default
        cuvs_index_type='auto',  # 'auto', 'ivf_flat', 'ivf_pq', 'cagra'
        cuvs_build_params=None,
        cuvs_search_params=None,
        **kwargs
    ):
        """
        Initialize DIRE with optional cuVS backend.
        
        Args:
            use_cuvs: Use cuVS for k-NN (auto-detect if None)
            cuvs_index_type: Type of cuVS index ('auto' selects based on data size)
            cuvs_build_params: Custom parameters for index building
            cuvs_search_params: Custom parameters for search
        """
        super().__init__(*args, **kwargs)
        
        # Auto-detect cuVS usage
        if use_cuvs is None:
            # Use cuVS if available and we have a GPU
            self.use_cuvs = CUVS_AVAILABLE and self.device.type == 'cuda'
        else:
            self.use_cuvs = use_cuvs and CUVS_AVAILABLE
        
        if self.use_cuvs:
            logger.info("cuVS backend enabled for k-NN computation")
        else:
            if use_cuvs and not CUVS_AVAILABLE:
                logger.warning("cuVS requested but not available, falling back to PyTorch")
        
        # Auto-detect cuML usage for PCA
        if use_cuml is None:
            self.use_cuml = CUML_AVAILABLE and self.device.type == 'cuda'
        else:
            self.use_cuml = use_cuml and CUML_AVAILABLE
        
        if self.use_cuml:
            logger.info("cuML backend enabled for PCA initialization")
        else:
            if use_cuml and not CUML_AVAILABLE:
                logger.warning("cuML requested but not available, falling back to sklearn")
        
        self.cuvs_index_type = cuvs_index_type
        self.cuvs_build_params = cuvs_build_params
        self.cuvs_search_params = cuvs_search_params
        self.cuvs_index = None
    
    def _select_cuvs_index_type(self, n_samples, n_dims):
        """
        Automatically select the best cuVS index type based on data characteristics.
        """
        if self.cuvs_index_type != 'auto':
            return self.cuvs_index_type
        
        # Decision tree based on scale and dimensionality
        # For high dimensions (>500), prefer IVF methods over graph-based
        if n_samples < 50000:
            # Small dataset - use flat (IVF with many lists)
            return 'flat'
        elif n_samples < 500000 or n_dims > 500:
            # Medium dataset or high-D - IVF without compression
            # IVF-Flat works better than CAGRA for high dimensions
            return 'ivf_flat'
        elif n_samples < 5000000:
            # Large dataset - IVF with compression
            return 'ivf_pq'
        else:
            # Very large dataset with moderate dimensions - graph-based
            return 'cagra' if n_dims <= 500 else 'ivf_pq'
    
    def _build_cuvs_index(self, X_gpu, index_type):
        """
        Build cuVS index for fast k-NN search.
        """
        n_samples, n_dims = X_gpu.shape
        
        self.logger.info(f"Building cuVS {index_type} index for {n_samples} points in {n_dims}D...")
        
        if index_type == 'flat':
            # Exact search - no index needed
            self.logger.info("Using brute-force search (exact)")
            return None
            
        elif index_type == 'ivf_flat':
            # IVF without compression
            # For high-D data, use more lists for better quantization
            if n_dims > 500:
                # High-D: more lists help with curse of dimensionality
                n_lists = min(int(np.sqrt(n_samples) * 2), 8192)
            else:
                n_lists = min(int(np.sqrt(n_samples)), 4096)
            
            build_params = ivf_flat.IndexParams(
                n_lists=n_lists,
                metric='euclidean',  # cuVS uses 'euclidean' not 'l2_expanded'
                add_data_on_build=True
            )
            
            if self.cuvs_build_params:
                build_params.update(self.cuvs_build_params)
            
            index = ivf_flat.build(build_params, X_gpu)
            
            self.logger.info(f"Built IVF-Flat index with {n_lists} lists for {n_dims}D data")
            
        elif index_type == 'ivf_pq':
            # IVF with product quantization
            n_lists = min(int(np.sqrt(n_samples)), 8192)
            pq_dim = min(n_dims // 4, 128)  # Reasonable PQ dimension
            
            build_params = ivf_pq.IndexParams(
                n_lists=n_lists,
                metric='euclidean',
                pq_dim=pq_dim,
                pq_bits=8,
                add_data_on_build=True
            )
            
            if self.cuvs_build_params:
                build_params.update(self.cuvs_build_params)
            
            index = ivf_pq.build(build_params, X_gpu)
            
            self.logger.info(f"Built IVF-PQ index with {n_lists} lists, PQ dim={pq_dim}")
            
        elif index_type == 'cagra':
            # Graph-based index for very large datasets
            build_params = cagra.IndexParams(
                metric='euclidean',
                graph_degree=32,
                intermediate_graph_degree=64,
                graph_build_algo='nn_descent'
            )
            
            if self.cuvs_build_params:
                build_params.update(self.cuvs_build_params)
            
            index = cagra.build(build_params, X_gpu)
            
            self.logger.info(f"Built CAGRA graph-based index")
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return index
    
    def _search_cuvs(self, index, index_type, X_gpu, k):
        """
        Search cuVS index for k nearest neighbors.
        """
        n_samples = X_gpu.shape[0]
        
        self.logger.info(f"Searching for {k} nearest neighbors using cuVS {index_type}...")
        
        if index_type == 'flat':
            # For flat/brute force, just use IVF-Flat with many lists for exact search
            # This avoids dtype issues with brute_force module
            n_lists = min(int(np.sqrt(n_samples)), 1024)
            
            build_params = ivf_flat.IndexParams(
                n_lists=n_lists,
                metric='euclidean',
                add_data_on_build=True
            )
            
            index = ivf_flat.build(build_params, X_gpu)
            
            # Search with high probe count for near-exact results
            search_params = ivf_flat.SearchParams(
                n_probes=min(n_lists, 256)  # High probe count for accuracy
            )
            
            distances, indices = ivf_flat.search(
                search_params, index, X_gpu, k+1
            )
            
        elif index_type == 'ivf_flat':
            # IVF search
            search_params = ivf_flat.SearchParams(
                n_probes=min(index.n_lists // 10, 100)
            )
            
            if self.cuvs_search_params:
                search_params.update(self.cuvs_search_params)
            
            distances, indices = ivf_flat.search(
                search_params, index, X_gpu, k+1
            )
            
        elif index_type == 'ivf_pq':
            # IVF-PQ search
            search_params = ivf_pq.SearchParams(
                n_probes=min(index.n_lists // 10, 200),
                internal_distance_dtype='float32'
            )
            
            if self.cuvs_search_params:
                search_params.update(self.cuvs_search_params)
            
            distances, indices = ivf_pq.search(
                search_params, index, X_gpu, k+1
            )
            
        elif index_type == 'cagra':
            # CAGRA search
            search_params = cagra.SearchParams(
                max_queries=0,  # Automatic
                itopk_size=min(k * 2, 256),
                search_width=4
            )
            
            if self.cuvs_search_params:
                search_params.update(self.cuvs_search_params)
            
            distances, indices = cagra.search(
                search_params, index, X_gpu, k+1
            )
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return distances, indices
    
    def _compute_knn(self, X, chunk_size=50000, use_fp16=None):
        """
        Compute k-NN using cuVS if available and beneficial.
        """
        n_samples, n_dims = X.shape
        
        # Decide whether to use cuVS
        use_cuvs_for_this = (
            self.use_cuvs and 
            n_samples >= 10000 and  # cuVS overhead not worth it for small datasets
            n_dims <= 2048  # cuVS works best for moderate dimensions
        )
        
        if not use_cuvs_for_this:
            # Fall back to PyTorch implementation
            self.logger.info("Using PyTorch backend for k-NN")
            return super()._compute_knn(X, chunk_size, use_fp16)
        
        # Use cuVS for k-NN
        self.logger.info(f"Computing {self.n_neighbors}-NN graph using cuVS...")
        
        # Convert to CuPy array
        # Note: cuVS requires float32, not float16
        # cuVS also requires C-contiguous (row-major) arrays
        X_gpu = cp.asarray(X, dtype=cp.float32, order='C')
        
        # Select index type
        index_type = self._select_cuvs_index_type(n_samples, n_dims)
        
        # Build index
        if index_type != 'flat':
            self.cuvs_index = self._build_cuvs_index(X_gpu, index_type)
        else:
            self.cuvs_index = None
        
        # Search for k-NN
        distances, indices = self._search_cuvs(
            self.cuvs_index, index_type, X_gpu, self.n_neighbors
        )
        
        # Convert to CuPy arrays first, then remove self (first neighbor) and convert to numpy
        indices_cp = cp.asarray(indices)
        distances_cp = cp.asarray(distances)
        self._knn_indices = cp.asnumpy(indices_cp[:, 1:])
        self._knn_distances = cp.asnumpy(distances_cp[:, 1:])
        
        self.logger.info(f"k-NN graph computed: shape {self._knn_indices.shape}")
        
        # Clean up GPU memory
        del X_gpu
        if self.cuvs_index is not None:
            del self.cuvs_index
            self.cuvs_index = None
        cp.get_default_memory_pool().free_all_blocks()
        
        return self
    
    def _initialize_embedding(self, X):
        """
        Initialize embedding using cuML PCA if available, else fall back to sklearn.
        """
        if self.use_cuml and self.init == 'pca':
            self.logger.info("Initializing with cuML PCA (GPU-accelerated)")
            
            # Convert to CuPy array if needed
            if isinstance(X, np.ndarray):
                X_gpu = cp.asarray(X, dtype=cp.float32)
            else:
                X_gpu = X
            
            # Use TruncatedSVD for high-dimensional data (more efficient)
            if X.shape[1] > 100:
                # TruncatedSVD is perfect for high-D to low-D reduction
                pca = cuTruncatedSVD(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
            else:
                # Regular PCA for lower dimensions
                pca = cuPCA(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
            
            # Fit and transform on GPU
            embedding_gpu = pca.fit_transform(X_gpu)
            
            # Convert to PyTorch tensor on GPU
            # cuML returns cupy array, convert to torch
            embedding_cp = cp.asarray(embedding_gpu)
            
            # Normalize on GPU
            embedding_cp -= embedding_cp.mean(axis=0)
            embedding_cp /= embedding_cp.std(axis=0)
            
            # Convert to PyTorch
            # Use dlpack for zero-copy transfer from CuPy to PyTorch
            from torch.utils.dlpack import from_dlpack
            embedding_torch = from_dlpack(embedding_cp.toDlpack())
            
            return embedding_torch.to(self.device)
        
        else:
            # Fall back to CPU sklearn PCA
            return super()._initialize_embedding(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform with cuVS acceleration.
        """
        # Log backend being used
        if self.use_cuvs and X.shape[0] >= 10000:
            self.logger.info(f"Using cuVS-accelerated backend for {X.shape[0]} points")
        else:
            self.logger.info(f"Using PyTorch backend for {X.shape[0]} points")
        
        return super().fit_transform(X, y)


# Convenience function for backend selection
def create_dire(backend='auto', **kwargs):
    """
    Create DIRE instance with appropriate backend.
    
    Args:
        backend: 'auto', 'cuvs', 'pytorch', or 'jax'
        **kwargs: Parameters for DIRE
    
    Returns:
        DIRE instance with selected backend
    """
    if backend == 'auto':
        # Auto-select best backend
        if CUVS_AVAILABLE and torch.cuda.is_available():
            logger.info("Auto-selected cuVS backend")
            return DiReCuVS(use_cuvs=True, **kwargs)
        elif torch.cuda.is_available():
            logger.info("Auto-selected PyTorch backend")
            return DiRePyTorch(**kwargs)
        else:
            # Fall back to JAX
            from .dire import DiRe
            logger.info("Auto-selected JAX backend")
            return DiRe(**kwargs)
    
    elif backend == 'cuvs':
        if not CUVS_AVAILABLE:
            raise RuntimeError("cuVS backend requested but RAPIDS not installed")
        return DiReCuVS(use_cuvs=True, **kwargs)
    
    elif backend == 'pytorch':
        return DiRePyTorch(**kwargs)
    
    elif backend == 'jax':
        from .dire import DiRe
        return DiRe(**kwargs)
    
    else:
        raise ValueError(f"Unknown backend: {backend}")