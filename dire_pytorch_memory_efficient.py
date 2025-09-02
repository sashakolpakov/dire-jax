# dire_pytorch_memory_efficient.py

"""
Memory-efficient PyTorch/PyKeOps backend for DiRe.

Key improvements:
1. Uses PyKeOps LazyTensors to avoid materializing large matrices
2. Processes k-NN attraction point-by-point to avoid memory issues
3. Offers choice between exact all-pairs repulsion (via PyKeOps) or random sampling
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


class DiRePyTorchMemoryEfficient(TransformerMixin):
    """
    Memory-efficient PyTorch/PyKeOps implementation.
    
    Key features:
    - k-NN attraction without materializing large tensors
    - PyKeOps LazyTensors for all-pairs repulsion
    - Automatic memory management
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
        neg_ratio=8,
        verbose=True,
        random_state=None,
        use_pykeops_repulsion=True,  # Use PyKeOps for repulsion when possible
        pykeops_threshold=50000,     # Max points for PyKeOps all-pairs
    ):
        """Initialize with memory-efficient defaults."""
        
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.init = init
        self.max_iter_layout = max_iter_layout
        self.min_dist = min_dist
        self.spread = spread
        self.cutoff = cutoff
        self.neg_ratio = neg_ratio
        self.verbose = verbose
        self.random_state = random_state if random_state is not None else np.random.randint(0, 2**32)
        self.use_pykeops_repulsion = use_pykeops_repulsion
        self.pykeops_threshold = pykeops_threshold
        
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
        
        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            # Log available memory
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU memory: {mem_gb:.1f} GB")
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
        Compute k-nearest neighbors in a memory-efficient way.
        """
        self.logger.info(f"Computing {self.n_neighbors}-NN graph...")
        
        n_samples = X.shape[0]
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # For small datasets, compute all at once
        if n_samples <= 10000:
            dist_matrix = torch.cdist(X_torch, X_torch, p=2)
            knn_dists, knn_indices = torch.topk(dist_matrix, k=self.n_neighbors + 1, 
                                                dim=1, largest=False)
            self._knn_indices = knn_indices[:, 1:].cpu().numpy()
        else:
            # Process in chunks for larger datasets
            self.logger.info("Using chunked k-NN computation for large dataset")
            chunk_size = min(5000, n_samples // 10)
            all_indices = []
            
            for i in range(0, n_samples, chunk_size):
                end_i = min(i + chunk_size, n_samples)
                batch = X_torch[i:end_i]
                
                # Compute distances from batch to all points
                dist_batch = torch.cdist(batch, X_torch, p=2)
                
                # Get k+1 nearest
                knn_dists, knn_indices = torch.topk(dist_batch, k=self.n_neighbors + 1,
                                                    dim=1, largest=False)
                
                all_indices.append(knn_indices[:, 1:].cpu().numpy())
                
                # Clear GPU cache periodically
                if self.device.type == 'cuda' and i % (chunk_size * 10) == 0:
                    torch.cuda.empty_cache()
            
            self._knn_indices = np.vstack(all_indices)
        
        self.logger.info(f"k-NN graph computed: shape {self._knn_indices.shape}")
    
    def _initialize_embedding(self, X):
        """Initialize the embedding using PCA or random projection."""
        
        n_samples = X.shape[0]
        
        if self.init == 'pca':
            self.logger.info("Initializing with PCA")
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            embedding = pca.fit_transform(X)
        else:
            self.logger.info("Initializing with random projection")
            rng = np.random.RandomState(self.random_state)
            projection = rng.randn(X.shape[1], self.n_components)
            projection /= np.linalg.norm(projection, axis=0)
            embedding = X @ projection
        
        # Normalize
        embedding -= embedding.mean(axis=0)
        embedding /= embedding.std(axis=0)
        
        return torch.tensor(embedding, dtype=torch.float32, device=self.device)
    
    def _compute_forces_memory_efficient(self, positions, iteration, max_iterations):
        """
        Memory-efficient force computation.
        """
        n_samples = positions.shape[0]
        forces = torch.zeros_like(positions)
        
        # Linear cooling
        alpha = 1.0 - iteration / max_iterations
        
        # Parameters
        a_val = float(self._a)
        b_val = float(self._b)
        b_exp = float(2 * b_val)
        
        # ============ ATTRACTION (k-NN only) ============
        # Process point-by-point to avoid memory issues
        for i in range(n_samples):
            neighbor_ids = self._knn_indices[i]
            pos_i = positions[i:i+1]
            pos_neighbors = positions[neighbor_ids]
            
            diff = pos_neighbors - pos_i
            dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
            
            att_coeff = 1.0 / (1.0 + a_val * (1.0 / dist) ** b_exp)
            forces[i] += (att_coeff * diff / dist).sum(0)
            
            # Clear cache periodically on GPU
            if self.device.type == 'cuda' and i % 1000 == 0:
                torch.cuda.empty_cache()
        
        # ============ REPULSION ============
        use_pykeops = (PYKEOPS_AVAILABLE and 
                      self.use_pykeops_repulsion and 
                      n_samples < self.pykeops_threshold and
                      self.device.type == 'cuda')
        
        if use_pykeops:
            # Use PyKeOps for efficient all-pairs repulsion
            self.logger.debug("Using PyKeOps for repulsion")
            
            X_i = LazyTensor(positions[:, None, :])
            X_j = LazyTensor(positions[None, :, :])
            
            diff = X_j - X_i
            D_ij = (diff ** 2).sum(-1).sqrt() + 1e-10
            
            # Repulsion kernel
            rep_kernel = -1.0 / (1.0 + a_val * (D_ij ** b_exp))
            
            # Apply cutoff
            cutoff_scale = (-D_ij / self.cutoff).exp()
            rep_kernel = rep_kernel * cutoff_scale
            
            # Compute forces (reduction happens in PyKeOps)
            force_dir = diff / D_ij
            rep_forces = (rep_kernel * force_dir).sum(1)
            forces += rep_forces
        else:
            # Random sampling for large datasets or CPU
            self.logger.debug("Using random sampling for repulsion")
            n_neg = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)
            
            for i in range(n_samples):
                # Sample negative points
                neg_idx = np.random.choice(n_samples, n_neg, replace=False)
                neg_idx = neg_idx[neg_idx != i]
                
                pos_i = positions[i:i+1]
                pos_neg = positions[neg_idx]
                
                diff = pos_neg - pos_i
                dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
                
                rep_coeff = -1.0 / (1.0 + a_val * (dist ** b_exp))
                cutoff_scale = torch.exp(-dist / self.cutoff)
                rep_coeff = rep_coeff * cutoff_scale
                
                forces[i] += (rep_coeff * diff / dist).sum(0)
        
        # Apply cooling and clipping
        forces = alpha * forces
        forces = torch.clamp(forces, -self.cutoff, self.cutoff)
        
        return forces
    
    def _optimize_layout(self, initial_positions):
        """Main optimization loop."""
        positions = initial_positions.clone()
        
        self.logger.info(f"Optimizing layout for {self._n_samples} points...")
        
        # Check memory usage
        if self.device.type == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU memory usage: {mem_used:.1f}/{mem_total:.1f} GB")
        
        for iteration in range(self.max_iter_layout):
            forces = self._compute_forces_memory_efficient(positions, iteration, self.max_iter_layout)
            positions += forces
            
            if iteration % 20 == 0:
                force_mag = torch.norm(forces, dim=1).mean().item()
                self.logger.debug(f"Iteration {iteration}/{self.max_iter_layout}, "
                                f"avg force: {force_mag:.6f}")
                
                # Monitor memory on GPU
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Final normalization
        positions -= positions.mean(dim=0)
        positions /= positions.std(dim=0)
        
        return positions
    
    def fit_transform(self, X, y=None):
        """Fit and transform the data."""
        # Store data
        self._data = np.asarray(X, dtype=np.float32)
        self._n_samples = self._data.shape[0]
        
        self.logger.info(f"Processing {self._n_samples} samples with {self._data.shape[1]} features")
        
        # Decide on strategy based on size
        if self._n_samples > self.pykeops_threshold:
            self.logger.warning(f"Dataset has {self._n_samples} samples, using random sampling for repulsion")
        elif PYKEOPS_AVAILABLE and self.device.type == 'cuda':
            self.logger.info("Will use PyKeOps for efficient all-pairs repulsion")
        
        # Find kernel parameters
        self._find_ab_params()
        
        # Compute k-NN graph
        self._compute_knn(self._data)
        
        # Initialize embedding
        initial_embedding = self._initialize_embedding(self._data)
        
        # Optimize layout
        final_embedding = self._optimize_layout(initial_embedding)
        
        # Convert back to numpy
        self._layout = final_embedding.cpu().numpy()
        
        # Clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return self._layout
    
    def visualize(self, labels=None, point_size=2, title=None, **kwargs):
        """Visualize the embedding."""
        if self._layout is None:
            self.logger.warning("No layout available for visualization")
            return None
        
        if title is None:
            title = f"PyTorch {self.n_components}D Embedding (Memory Efficient)"
        
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