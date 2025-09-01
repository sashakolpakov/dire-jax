
# dire-jax

"""
A JAX-based dimensionality reducer.
"""

from .dire import DiRe  # core class import

# Attempt to import PyTorch backend
try:
    from .dire_pytorch import DiRePyTorch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Attempt to import optional utilities, set a flag accordingly
try:
    from . import dire_utils
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

# Optionally inform users that optional modules aren't available unless explicitly installed
if not HAS_UTILS:
    import warnings
    warnings.warn(
        "Optional module 'dire_utils' not found. "
        "If you need utility functions, install dire-jax with extras: pip install dire-jax[utils]",
        UserWarning
    )

if not HAS_PYTORCH:
    import warnings
    warnings.warn(
        "PyTorch backend not available. "
        "For high-performance PyTorch backend with PyKeOps, install: pip install dire-jax[torch]",
        UserWarning
    )

# Build __all__ based on available modules
__all__ = ['DiRe']
if HAS_PYTORCH:
    __all__.append('DiRePyTorch')
if HAS_UTILS:
    __all__.append('dire_utils')
