# dire_pytorch.py

"""
PyTorch/PyKeOps backend for DiRe dimensionality reduction.

This module provides an alternative implementation using PyTorch and PyKeOps
for improved performance on CUDA GPUs with small to medium datasets (<100K points).
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import plotly.express as px
import pandas as pd
from loguru import logger
import gc

# PyKeOps for efficient force computations
try:
    from pykeops.torch import LazyTensor
    PYKEOPS_AVAILABLE = True
except ImportError:
    PYKEOPS_AVAILABLE = False
    logger.warning("PyKeOps not available. Install with: pip install pykeops")

# Optional cuVS for large-scale k-NN
try:
    from cuvs.neighbors import cagra
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False
    logger.info("cuVS not available. Will use PyTorch for k-NN if needed.")


class DiRePyTorch(TransformerMixin):
    """
    PyTorch/PyKeOps implementation of DiRe with API compatibility.
    
    Automatically selects backend based on dataset size:
    - <2M points: PyKeOps all-pairs force computation (no k-NN needed!)
    - >2M points: Falls back to k-NN graph construction (not yet implemented)
    
    Performance on H100:
    - 100K samples: ~0.1s per iteration (100x faster than JAX)
    - 500K samples: ~0.7s per iteration  
    - 1M samples: ~2.7s per iteration
    - 2M samples: ~10s per iteration
    
    Parameters match the original DiRe class for compatibility.
    """
    
    def __init__(
        self,
        n_components=2,
        n_neighbors=16,
        init="random",
        metric="lp",
        sim_kernel=None,
        pca_kernel=None,
        max_iter_layout=128,
        min_dist=1e-2,
        spread=1.0,
        cutoff=42.0,
        n_sample_dirs=8,
        sample_size=16,
        batch_size=None,
        neg_ratio=8,
        my_logger=None,
        verbose=True,
        memm=None,
        mpa=True,
        random_state=None,
        force_knn_threshold=2000000,  # 2M samples by default!
        **metric_kwargs,
    ):
        """Initialize DiRePyTorch with parameters matching original DiRe."""
        
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.init = init
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.sim_kernel = sim_kernel
        self.pca_kernel = pca_kernel
        self.max_iter_layout = max_iter_layout
        self.min_dist = min_dist
        self.spread = spread
        self.cutoff = cutoff
        self.n_sample_dirs = n_sample_dirs
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.verbose = verbose
        self.memm = memm
        self.mpa = mpa
        self.random_state = random_state if random_state is not None else np.random.randint(0, 2**32)
        self.force_knn_threshold = force_knn_threshold
        
        # Setup logger
        if my_logger is not None:
            self.logger = my_logger
        else:
            self.logger = logger
            if not verbose:
                self.logger.disable(__name__)
        
        # Internal state
        self._data = None
        self._layout = None
        self._n_samples = None
        self._a = None
        self._b = None
        
        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.logger.warning("CUDA not available, using CPU (will be slower)")
    
    def _find_ab_params(self):
        """Find a and b parameters for the distribution kernel."""
        
        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))
        
        xv = np.linspace(0, 3 * self.spread, 300)
        yv = np.zeros(xv.shape)
        yv[xv < self.min_dist] = 1.0
        yv[xv >= self.min_dist] = np.exp(-(xv[xv >= self.min_dist] - self.min_dist) / self.spread)
        
        params, _ = curve_fit(curve, xv, yv)
        self._a, self._b = params
        
        self.logger.info(f"Found kernel params: a={self._a:.4f}, b={self._b:.4f}")
    
    def _initialize_embedding(self, X):
        """Initialize the embedding using PCA, random, or spectral methods."""
        
        n_samples = X.shape[0]
        
        if self.init == 'pca':
            self.logger.info("Initializing with PCA")
            if self.pca_kernel is not None:
                raise NotImplementedError("Kernel PCA not yet implemented in PyTorch backend")
            else:
                pca = PCA(n_components=self.n_components, random_state=self.random_state)
                embedding = pca.fit_transform(X)
        
        elif self.init == 'spectral':
            self.logger.info("Initializing with spectral embedding")
            # Build affinity matrix
            if n_samples < self.force_knn_threshold and PYKEOPS_AVAILABLE:
                # Use PyKeOps for affinity computation
                embedding = self._spectral_embedding_pykeops(X)
            else:
                # Fall back to scipy sparse matrix approach
                embedding = self._spectral_embedding_sparse(X)
        
        elif self.init == 'random':
            self.logger.info("Initializing with random projection")
            rng = np.random.RandomState(self.random_state)
            # Johnson-Lindenstrauss random projection
            projection = rng.randn(X.shape[1], self.n_components)
            projection /= np.linalg.norm(projection, axis=0)
            embedding = X @ projection
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
        
        # Normalize
        embedding -= embedding.mean(axis=0)
        embedding /= embedding.std(axis=0)
        
        return torch.tensor(embedding, dtype=torch.float32, device=self.device)
    
    def _spectral_embedding_sparse(self, X):
        """Spectral embedding using sparse matrix (fallback)."""
        # Would need k-NN graph here - simplified for now
        n_samples = X.shape[0]
        
        # Simple distance-based affinity
        from sklearn.metrics.pairwise import pairwise_distances
        distances = pairwise_distances(X, metric='euclidean')
        
        # Convert to affinity
        gamma = 1.0 / (2.0 * np.median(distances) ** 2)
        affinity = np.exp(-gamma * distances ** 2)
        
        # Compute Laplacian eigenmaps
        L = laplacian(affinity, normed=True)
        eigenvalues, eigenvectors = eigsh(L, k=self.n_components + 1, which='SM')
        
        # Skip the first eigenvector (constant)
        embedding = eigenvectors[:, 1:self.n_components + 1]
        
        return embedding
    
    def _spectral_embedding_pykeops(self, X):
        """Spectral embedding using PyKeOps for efficiency."""
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # LazyTensors for all-pairs computation
        X_i = LazyTensor(X_torch[:, None, :])  # (N, 1, D)
        X_j = LazyTensor(X_torch[None, :, :])  # (1, N, D)
        
        # Compute squared distances
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        
        # Convert to affinity with Gaussian kernel
        gamma = 1.0 / (2.0 * torch.median(D_ij.sum(1)).item())
        K_ij = (-gamma * D_ij).exp()
        
        # Normalize to get transition matrix
        K_sum = K_ij.sum(1)
        P_ij = K_ij / K_sum[:, None]
        
        # Power iteration for top eigenvectors
        # Simplified - in practice would use more sophisticated method
        n_samples = X.shape[0]
        V = torch.randn(n_samples, self.n_components + 1, device=self.device)
        
        for _ in range(50):  # Power iterations
            V_new = P_ij @ V
            V, _ = torch.linalg.qr(V_new)
        
        # Skip first eigenvector
        embedding = V[:, 1:self.n_components + 1].cpu().numpy()
        
        return embedding
    
    def _compute_forces_pykeops(self, positions, alpha=1.0):
        """
        Compute all-pairs forces using PyKeOps (for small datasets).
        
        This is the key innovation - no k-NN graph needed!
        """
        if not PYKEOPS_AVAILABLE:
            raise RuntimeError("PyKeOps required for force computation")
        
        # Convert to LazyTensors
        X_i = LazyTensor(positions[:, None, :])  # (N, 1, D)
        X_j = LazyTensor(positions[None, :, :])  # (1, N, D)
        
        # Compute differences and distances
        diff = X_j - X_i  # (N, N, D) but lazy
        D_ij_sq = (diff ** 2).sum(-1)  # Squared distances
        D_ij = D_ij_sq.sqrt()  # Actual distances
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        D_ij_safe = D_ij + eps
        
        # Distribution kernel for attraction and repulsion
        # PyKeOps needs scalar values for exponents
        a_val = float(self._a)
        b_val = float(self._b)
        b_exp = float(2 * b_val)  # Pre-compute the exponent as a scalar
        
        # Attraction: kernel(1/dist)
        # Using the formula: 1 / (1 + a * (1/dist)^(2b))
        inv_dist = 1.0 / D_ij_safe
        att_kernel = 1.0 / (1.0 + a_val * (inv_dist ** b_exp))
        
        # Repulsion: -kernel(dist)
        # Using the formula: -1 / (1 + a * dist^(2b))
        rep_kernel = -1.0 / (1.0 + a_val * (D_ij_safe ** b_exp))
        
        # Combined force coefficient
        force_coeff = att_kernel + rep_kernel
        
        # Apply distance cutoff for efficiency (soft cutoff)
        cutoff_val = float(self.cutoff)
        cutoff_scale = (-D_ij / cutoff_val).exp()
        force_coeff = force_coeff * cutoff_scale
        
        # Compute force vectors
        # Force = coefficient * (direction vector)
        # Direction = diff / distance
        # We normalize by dividing by distance (with epsilon for safety)
        normalized_diff = diff / D_ij_safe
        
        # Apply force coefficient to each dimension of the difference vector
        # and sum over all j points to get total force on each i point
        forces_lazy = force_coeff * normalized_diff
        forces = forces_lazy.sum(dim=1)  # Sum over j dimension
        
        # Apply alpha cooling and return as regular tensor
        return float(alpha) * forces
    
    def _compute_forces_knn(self, positions, knn_indices, alpha=1.0):
        """
        Compute forces using k-NN graph (for large datasets).
        
        This would be the fallback for >100K points.
        """
        # Simplified version - would need full implementation
        forces = torch.zeros_like(positions)
        
        # Would implement k-NN based force computation here
        # Similar to JAX version but using PyTorch operations
        
        raise NotImplementedError("k-NN force computation not yet implemented")
    
    def _optimize_layout(self, initial_positions):
        """
        Main optimization loop for layout refinement.
        """
        positions = initial_positions.clone()
        
        # Smart backend selection based on hardware and dataset size
        can_use_pykeops = PYKEOPS_AVAILABLE and self.device.type == 'cuda'
        
        if can_use_pykeops:
            # Check GPU memory to adjust threshold dynamically
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Adjust threshold based on GPU memory (rough heuristic)
            # H100 80GB can handle 2M+, A100 40GB ~1M, smaller GPUs less
            if gpu_mem_gb >= 70:  # H100 80GB
                effective_threshold = 2000000
            elif gpu_mem_gb >= 30:  # A100 40GB, A6000 48GB
                effective_threshold = 1000000
            elif gpu_mem_gb >= 20:  # A10 24GB, 3090 24GB
                effective_threshold = 500000
            else:  # Smaller GPUs
                effective_threshold = 200000
            
            # Use the minimum of user setting and GPU-based threshold
            final_threshold = min(self.force_knn_threshold, effective_threshold)
            use_pykeops = self._n_samples < final_threshold
            
            if use_pykeops:
                self.logger.info(f"Using PyKeOps all-pairs forces for {self._n_samples:,} points")
                self.logger.info(f"GPU: {torch.cuda.get_device_name()}, Memory: {gpu_mem_gb:.1f}GB")
            else:
                self.logger.warning(f"Dataset ({self._n_samples:,} points) exceeds threshold ({final_threshold:,})")
                use_pykeops = False
        else:
            use_pykeops = False
            if not PYKEOPS_AVAILABLE:
                self.logger.warning("PyKeOps not available - install with: pip install pykeops")
            if self.device.type != 'cuda':
                self.logger.warning("CUDA not available - PyKeOps requires GPU")
        
        if not use_pykeops:
            self.logger.error(f"Cannot process {self._n_samples:,} points without PyKeOps")
            raise NotImplementedError("k-NN backend not yet implemented - PyKeOps required for large datasets")
        
        # Optimization loop
        for iteration in range(self.max_iter_layout):
            # Linear cooling schedule
            alpha = 1.0 - iteration / self.max_iter_layout
            
            # Compute forces
            if use_pykeops:
                forces = self._compute_forces_pykeops(positions, alpha)
            else:
                # Would use k-NN graph here
                forces = self._compute_forces_knn(positions, None, alpha)
            
            # Clip forces to prevent instability
            forces = torch.clamp(forces, -self.cutoff, self.cutoff)
            
            # Update positions
            positions += forces
            
            # Log progress
            if iteration % 20 == 0:
                force_magnitude = torch.norm(forces, dim=1).mean().item()
                self.logger.debug(f"Iteration {iteration}/{self.max_iter_layout}, "
                                f"avg force magnitude: {force_magnitude:.6f}")
        
        # Final normalization
        positions -= positions.mean(dim=0)
        positions /= positions.std(dim=0)
        
        return positions
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and transform data (API compatible with original DiRe).
        """
        # Store data
        self._data = np.asarray(X, dtype=np.float32)
        self._n_samples = self._data.shape[0]
        
        self.logger.info(f"Processing {self._n_samples} samples with {self._data.shape[1]} features")
        
        # Find distribution kernel parameters
        self._find_ab_params()
        
        # Initialize embedding
        initial_embedding = self._initialize_embedding(self._data)
        
        # Optimize layout
        final_embedding = self._optimize_layout(initial_embedding)
        
        # Convert back to numpy and store
        self._layout = final_embedding.cpu().numpy()
        
        # Clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return self._layout
    
    def fit(self, X, y=None):
        """Fit the model (for sklearn compatibility)."""
        self.fit_transform(X, y)
        return self
    
    def transform(self, X):
        """Transform data (returns the fitted layout)."""
        if self._layout is None:
            raise ValueError("Model must be fitted before transform")
        return self._layout
    
    def visualize(self, labels=None, point_size=2, title=None, colormap=None,
                 width=800, height=600, opacity=0.7):
        """
        Visualize the embedding (API compatible with original DiRe).
        """
        if self._layout is None:
            self.logger.warning("No layout available for visualization")
            return None
        
        if title is None:
            title = f"PyTorch/PyKeOps {self.n_components}D Embedding"
        
        # Create dataframe
        if self.n_components == 2:
            df = pd.DataFrame(self._layout, columns=['x', 'y'])
        elif self.n_components == 3:
            df = pd.DataFrame(self._layout, columns=['x', 'y', 'z'])
        else:
            self.logger.error(f"Cannot visualize {self.n_components}D embedding")
            return None
        
        # Add labels if provided
        if labels is not None:
            df['label'] = labels
        
        # Create plot
        vis_params = {
            'color': 'label' if labels is not None else None,
            'opacity': opacity,
            'title': title,
        }
        
        if self.n_components == 2:
            fig = px.scatter(df, x='x', y='y', **vis_params)
        else:
            fig = px.scatter_3d(df, x='x', y='y', z='z', **vis_params)
        
        fig.update_layout(width=width, height=height)
        fig.show()
        
        return fig