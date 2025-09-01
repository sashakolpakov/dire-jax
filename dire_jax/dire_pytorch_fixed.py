# dire_pytorch_fixed.py

"""
Fixed PyTorch/PyKeOps backend for DiRe dimensionality reduction.

This version correctly implements attraction forces only between k-NN neighbors
and repulsion forces from random samples, matching the JAX implementation.
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


class DiRePyTorchFixed(TransformerMixin):
    """
    FIXED PyTorch/PyKeOps implementation that correctly handles k-NN attraction.
    
    Key fix: Attraction forces are applied ONLY between k-NN neighbors,
    not all pairs. Repulsion uses random sampling.
    """
    
    def __init__(
        self,
        n_components=2,
        n_neighbors=16,
        init="random",
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
        **kwargs,
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
        self.random_state = random_state if random_state is not None else np.random.randint(0, 2**32)
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
    
    def _compute_knn(self, X):
        """
        Compute k-nearest neighbors using PyTorch.
        For simplicity, using brute force here - could use FAISS/cuVS for larger datasets.
        """
        self.logger.info(f"Computing {self.n_neighbors}-NN graph...")
        
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        # For large datasets, this should be done in chunks
        if n_samples < 50000:
            # Compute all pairwise distances
            dist_matrix = torch.cdist(X_torch, X_torch, p=2)
            
            # Get k+1 nearest neighbors (including self)
            knn_dists, knn_indices = torch.topk(dist_matrix, k=self.n_neighbors + 1, 
                                                dim=1, largest=False)
            
            # Remove self (first neighbor)
            self._knn_indices = knn_indices[:, 1:].cpu().numpy()
            self._knn_distances = knn_dists[:, 1:].cpu().numpy()
        else:
            # For larger datasets, process in chunks
            self.logger.warning("Large dataset - using chunked k-NN computation")
            batch_size = 5000
            all_indices = []
            all_distances = []
            
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                batch = X_torch[i:end_i]
                
                # Compute distances from batch to all points
                dist_batch = torch.cdist(batch, X_torch, p=2)
                
                # Get k+1 nearest
                knn_dists, knn_indices = torch.topk(dist_batch, k=self.n_neighbors + 1,
                                                    dim=1, largest=False)
                
                all_indices.append(knn_indices[:, 1:].cpu().numpy())
                all_distances.append(knn_dists[:, 1:].cpu().numpy())
            
            self._knn_indices = np.vstack(all_indices)
            self._knn_distances = np.vstack(all_distances)
        
        self.logger.info(f"k-NN graph computed: shape {self._knn_indices.shape}")
    
    def _initialize_embedding(self, X):
        """Initialize the embedding using PCA or random."""
        
        n_samples = X.shape[0]
        
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
    
    def _compute_forces_correct(self, positions, iteration, max_iterations):
        """
        CORRECTLY compute forces:
        - Attraction: ONLY between k-NN neighbors
        - Repulsion: From random samples (or all-pairs if use_exact_repulsion=True)
        """
        
        n_samples = positions.shape[0]
        forces = torch.zeros_like(positions)
        
        # Linear cooling schedule
        alpha = 1.0 - iteration / max_iterations
        
        # Parameters
        a_val = float(self._a)
        b_val = float(self._b)
        b_exp = float(2 * b_val)
        
        # ============ ATTRACTION FORCES (k-NN only) ============
        # Use PyTorch for k-NN attraction (simpler and works)
        if True:  # Always use this path for now
            # Fallback to PyTorch
            for i in range(n_samples):
                neighbor_ids = self._knn_indices[i]
                pos_i = positions[i:i+1]  # Keep dimensions
                pos_neighbors = positions[neighbor_ids]
                
                # Compute differences and distances
                diff = pos_neighbors - pos_i
                dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
                
                # Attraction kernel
                att_coeff = 1.0 / (1.0 + a_val * (1.0 / dist) ** b_exp)
                
                # Apply force
                forces[i] += (att_coeff * diff / dist).sum(0)
        
        # ============ REPULSION FORCES ============
        if self.use_exact_repulsion and PYKEOPS_AVAILABLE and self.device.type == 'cuda':
            # All-pairs repulsion (for testing)
            X_i = LazyTensor(positions[:, None, :])
            X_j = LazyTensor(positions[None, :, :])
            
            diff = X_j - X_i
            D_ij = (diff ** 2).sum(-1).sqrt() + 1e-10
            
            # Repulsion kernel: -1 / (1 + a * d^(2b))
            rep_kernel = -1.0 / (1.0 + a_val * (D_ij ** b_exp))
            
            # Apply distance cutoff
            cutoff_scale = (-D_ij / self.cutoff).exp()
            rep_kernel = rep_kernel * cutoff_scale
            
            # Compute repulsion forces
            force_dir = diff / D_ij
            rep_forces = (rep_kernel * force_dir).sum(1)
            forces += rep_forces
        else:
            # Random sampling for repulsion (more efficient and often better)
            n_neg_samples = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)
            
            for i in range(n_samples):
                # Random sample for repulsion
                neg_samples = np.random.choice(n_samples, n_neg_samples, replace=False)
                neg_samples = neg_samples[neg_samples != i]  # Exclude self
                
                pos_i = positions[i:i+1]
                pos_neg = positions[neg_samples]
                
                # Compute differences and distances
                diff = pos_neg - pos_i
                dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
                
                # Repulsion kernel
                rep_coeff = -1.0 / (1.0 + a_val * (dist ** b_exp))
                
                # Apply distance cutoff
                cutoff_scale = torch.exp(-dist / self.cutoff)
                rep_coeff = rep_coeff * cutoff_scale
                
                # Apply force
                forces[i] += (rep_coeff * diff / dist).sum(0)
        
        # Apply cooling and clipping
        forces = alpha * forces
        forces = torch.clamp(forces, -self.cutoff, self.cutoff)
        
        return forces
    
    def _optimize_layout(self, initial_positions):
        """
        Main optimization loop using CORRECT force computation.
        """
        positions = initial_positions.clone()
        
        self.logger.info(f"Optimizing layout for {self._n_samples} points...")
        
        # Optimization loop
        for iteration in range(self.max_iter_layout):
            # Compute forces correctly
            forces = self._compute_forces_correct(positions, iteration, self.max_iter_layout)
            
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
        
        # Compute k-NN graph (CRUCIAL!)
        self._compute_knn(self._data)
        
        # Initialize embedding
        initial_embedding = self._initialize_embedding(self._data)
        
        # Optimize layout with CORRECT forces
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
            title = f"PyTorch {self.n_components}D Embedding (Fixed)"
        
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
        fig.show()
        
        return fig