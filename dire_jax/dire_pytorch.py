# dire_pytorch.py

"""
PyTorch/PyKeOps backend for DiRe dimensionality reduction.

This implementation features:
- Memory-efficient chunked k-NN computation for large datasets (>100K points)
- Attraction forces applied only between k-NN neighbors  
- Repulsion forces computed from random samples
- Automatic GPU memory management with adaptive chunk sizing
- Designed for high-performance processing on CUDA GPUs

Performance characteristics:
- Best for datasets >50K points on CUDA GPUs
- Memory-aware processing up to millions of points
- Chunked computation prevents GPU out-of-memory errors
"""

import numpy as np
import torch
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
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


class DiRePyTorch(TransformerMixin):
    """
    Memory-efficient PyTorch/PyKeOps implementation of DiRe.
    
    Features adaptive memory management for large datasets:
    - Chunked k-NN computation prevents GPU out-of-memory errors
    - Memory-aware force computation with automatic chunk sizing  
    - Attraction forces between k-NN neighbors only
    - Repulsion forces from random sampling for efficiency
    
    Best suited for:
    - Large datasets (>50K points) on CUDA GPUs
    - Production environments requiring reliable memory usage
    - High-performance dimensionality reduction workflows
    """

    def __init__(
            self,
            n_components=2,
            n_neighbors=16,
            init="pca",
            max_iter_layout=128,
            min_dist=1e-2,
            spread=1.0,
            cutoff=42.0,
            n_sample_dirs=8,
            sample_size=16,
            neg_ratio=8,
            verbose=True,
            random_state=None,
            use_exact_repulsion=False,  # If True, use all-pairs repulsion (for testing)
    ):
        """Initialize with parameters matching original DiRe."""

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.init = init
        self.max_iter_layout = max_iter_layout
        self.min_dist = min_dist
        self.spread = spread
        self.cutoff = cutoff
        self.n_sample_dirs = n_sample_dirs
        self.sample_size = sample_size
        self.neg_ratio = neg_ratio
        self.verbose = verbose
        self.random_state = random_state if random_state is not None else np.random.randint(0, 2 ** 32)
        self.use_exact_repulsion = use_exact_repulsion

        # Setup logger
        self.logger = logger
        if not verbose:
            self.logger.disable(__name__)

        # Internal state
        self._data = None
        self._layout = None
        self._n_samples = None
        self._a = None
        self._b = None
        self._knn_indices = None
        self._knn_distances = None

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.logger.warning("CUDA not available, using CPU")

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

    def _compute_knn(self, X, chunk_size=50000):
        """
        Compute k-nearest neighbors with memory-efficient chunking.
        """
        if not PYKEOPS_AVAILABLE:
            raise RuntimeError("PyKeOps required for k-NN computation. Install with: pip install pykeops")
        
        n_samples = X.shape[0]
        self.logger.info(f"Computing {self.n_neighbors}-NN graph for {n_samples} points...")

        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Adaptive chunk sizing based on available GPU memory and dataset size
        if self.device.type == 'cuda':
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            # Estimate memory for k-NN: chunk_size * n_samples * 4 bytes for distances
            memory_per_chunk = chunk_size * n_samples * 4
            
            # Use 30% of available memory for k-NN computation
            max_memory = gpu_mem_free * 0.3
            if memory_per_chunk > max_memory:
                chunk_size = int(max_memory / (n_samples * 4))
                chunk_size = max(1000, chunk_size)  # Minimum chunk size
            
            self.logger.info(f"Using chunk size: {chunk_size} (GPU memory: {gpu_mem_free/1024**3:.1f}GB)")
        
        # Initialize arrays for results
        all_knn_indices = []
        all_knn_distances = []
        
        # Process in chunks to avoid memory issues
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            
            self.logger.info(f"Processing chunk {start_idx//chunk_size + 1}/{(n_samples + chunk_size - 1)//chunk_size}")
            
            # Get chunk data
            X_chunk = X_torch[start_idx:end_idx]  # (chunk_size, D)
            
            if PYKEOPS_AVAILABLE:
                # Use PyKeOps for this chunk vs all points
                X_i = LazyTensor(X_chunk[:, None, :])  # (chunk_size, 1, D)
                X_j = LazyTensor(X_torch[None, :, :])   # (1, N, D)
                
                # Compute squared distances
                D_ij = ((X_i - X_j) ** 2).sum(-1)  # (chunk_size, N) LazyTensor
                
                # Find k+1 nearest neighbors (including self)
                knn_dists, knn_indices = D_ij.Kmin_argKmin(K=self.n_neighbors + 1, dim=1)
                
                # Remove self and convert to actual distances
                chunk_indices = knn_indices[:, 1:].cpu().numpy()
                chunk_distances = torch.sqrt(knn_dists[:, 1:]).cpu().numpy()
            else:
                # Fallback to PyTorch (slower but more memory friendly)
                distances = torch.cdist(X_chunk, X_torch, p=2)
                knn_dists, knn_indices = torch.topk(distances, k=self.n_neighbors + 1, 
                                                   dim=1, largest=False)
                
                # Remove self
                chunk_indices = knn_indices[:, 1:].cpu().numpy()  
                chunk_distances = knn_dists[:, 1:].cpu().numpy()
            
            all_knn_indices.append(chunk_indices)
            all_knn_distances.append(chunk_distances)
            
            # Clear GPU memory for this chunk
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Concatenate results
        self._knn_indices = np.vstack(all_knn_indices)
        self._knn_distances = np.vstack(all_knn_distances)

        self.logger.info(f"k-NN graph computed: shape {self._knn_indices.shape}")

    def _initialize_embedding(self, X):
        """Initialize the embedding using PCA or random."""

        if self.init == 'pca':
            self.logger.info("Initializing with PCA")
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            embedding = pca.fit_transform(X)

        elif self.init == 'random':
            self.logger.info("Initializing with random projection")
            rng = np.random.RandomState(self.random_state)
            projection = rng.randn(X.shape[1], self.n_components)
            projection /= np.linalg.norm(projection, axis=0)
            embedding = X @ projection

        else:
            raise ValueError(f"Unknown init method: {self.init}")

        # Normalize
        embedding -= embedding.mean(axis=0)
        embedding /= embedding.std(axis=0)

        return torch.tensor(embedding, dtype=torch.float32, device=self.device)

    def _compute_forces(self, positions, iteration, max_iterations, chunk_size=5000):
        """
        Memory-efficient force computation with chunked processing:
        - Attraction: only between k-NN neighbors
        - Repulsion: random sampling
        """
        if not PYKEOPS_AVAILABLE:
            raise RuntimeError("PyKeOps required for efficient force computation")

        n_samples = positions.shape[0]
        forces = torch.zeros_like(positions)

        # Linear cooling schedule
        alpha = 1.0 - iteration / max_iterations

        # Parameters
        a_val = float(self._a)
        b_val = float(self._b)
        b_exp = float(2 * b_val)
        
        # Adjust chunk size based on available memory
        # Estimate memory usage: chunk_size * (k + n_neg) * D * 4 bytes
        n_neg_samples = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)
        memory_per_point = (self.n_neighbors + n_neg_samples) * positions.shape[1] * 4  # bytes
        
        if self.device.type == 'cuda':
            # Get available GPU memory and use 20% for force computation (more conservative)
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            max_chunk_size = int(gpu_mem_free * 0.2 / memory_per_point)
            chunk_size = min(chunk_size, max_chunk_size, n_samples)
            # For very large datasets, be extra conservative
            if n_samples > 500000:
                chunk_size = min(chunk_size, 2000)
        else:
            chunk_size = min(chunk_size, n_samples)

        # Process in chunks to manage memory
        knn_indices_torch = torch.tensor(self._knn_indices, device=self.device)
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_indices = slice(start_idx, end_idx)
            
            # ============ ATTRACTION FORCES (k-NN only) ============
            # Get chunk data
            chunk_positions = positions[chunk_indices]  # (chunk, D)
            chunk_knn_indices = knn_indices_torch[chunk_indices]  # (chunk, k)
            
            # Get neighbor positions for this chunk
            neighbor_positions = positions[chunk_knn_indices]  # (chunk, k, D)
            current_positions = chunk_positions.unsqueeze(1)  # (chunk, 1, D)
            
            # Compute differences and distances
            diff = neighbor_positions - current_positions  # (chunk, k, D)
            dist = torch.norm(diff, dim=2, keepdim=True) + 1e-10  # (chunk, k, 1)
            
            # Attraction kernel
            att_coeff = 1.0 / (1.0 + a_val * (1.0 / dist) ** b_exp)  # (chunk, k, 1)
            
            # Compute attraction forces and sum over neighbors
            att_forces = (att_coeff * diff / dist).sum(dim=1)  # (chunk, D)
            forces[chunk_indices] += att_forces

            # ============ REPULSION FORCES (Random Sampling) ============
            if n_neg_samples > 0:
                chunk_size_actual = end_idx - start_idx
                
                # Generate random samples for this chunk
                neg_indices = torch.randint(0, n_samples, (chunk_size_actual, n_neg_samples + 5), 
                                          device=self.device)
                
                # Create mask to exclude points from the current chunk
                chunk_range = torch.arange(start_idx, end_idx, device=self.device)
                self_mask = neg_indices == chunk_range.unsqueeze(1)
                
                # Replace self indices with valid random ones
                replacement_indices = torch.randint(0, n_samples, (chunk_size_actual, n_neg_samples + 5), 
                                                  device=self.device)
                neg_indices = torch.where(self_mask, replacement_indices, neg_indices)
                
                # Take first n_neg_samples
                neg_indices = neg_indices[:, :n_neg_samples]
                
                # Get negative sample positions
                neg_positions = positions[neg_indices]  # (chunk, n_neg, D)
                current_positions = chunk_positions.unsqueeze(1)  # (chunk, 1, D)
                
                # Compute differences and distances
                diff = neg_positions - current_positions  # (chunk, n_neg, D)
                dist = torch.norm(diff, dim=2, keepdim=True) + 1e-10  # (chunk, n_neg, 1)
                
                # Repulsion kernel
                rep_coeff = -1.0 / (1.0 + a_val * (dist ** b_exp))  # (chunk, n_neg, 1)
                
                # Apply distance cutoff
                cutoff_scale = torch.exp(-dist / self.cutoff)
                rep_coeff = rep_coeff * cutoff_scale
                
                # Compute repulsion forces and sum over negative samples
                rep_forces = (rep_coeff * diff / dist).sum(dim=1)  # (chunk, D)
                forces[chunk_indices] += rep_forces

        # Apply cooling and clipping
        forces = alpha * forces
        forces = torch.clamp(forces, -self.cutoff, self.cutoff)

        return forces

    def _optimize_layout(self, initial_positions):
        """
        Main optimization loop using force computation.
        """
        positions = initial_positions.clone()

        self.logger.info(f"Optimizing layout for {self._n_samples} points...")

        # Optimization loop
        for iteration in range(self.max_iter_layout):
            # Compute forces correctly
            forces = self._compute_forces(positions, iteration, self.max_iter_layout)

            # Update positions
            positions += forces

            # Log progress
            self.logger.info(f"Iteration {iteration}/{self.max_iter_layout}")

        # Final normalization
        positions -= positions.mean(dim=0)
        positions /= positions.std(dim=0)

        return positions

    def fit_transform(self, X, y=None):
        """
        Fit the model and transform data (API compatible with JAX backend).
        """
        # Store data
        self._data = np.asarray(X, dtype=np.float32)
        self._n_samples = self._data.shape[0]

        self.logger.info(f"Processing {self._n_samples} samples with {self._data.shape[1]} features")

        # Find distribution kernel parameters
        self._find_ab_params()

        # Compute k-NN graph
        self._compute_knn(self._data)

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

    def visualize(self, labels=None, point_size=2, title=None, **kwargs):
        """
        Visualize the embedding (API compatible).
        """
        if self._layout is None:
            self.logger.warning("No layout available for visualization")
            return None

        if title is None:
            title = f"PyTorch {self.n_components}D Embedding"

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
            'title': title,
        }
        vis_params.update(kwargs)

        if self.n_components == 2:
            fig = px.scatter(df, x='x', y='y', **vis_params)
        else:
            fig = px.scatter_3d(df, x='x', y='y', z='z', **vis_params)

        fig.update_traces(marker=dict(size=point_size))

        return fig
