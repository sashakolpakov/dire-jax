
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

# Optionally inform users that utilities aren't available unless explicitly installed
if not HAS_UTILS:
    import warnings
    warnings.warn(
        "Optional module 'dire_utils' not found. "
        "If you need utility functions, install dire-jax with extras: pip install dire-jax[utils]",
        UserWarning
    )

# Build __all__ based on available modules
__all__ = ['DiRe']
if HAS_PYTORCH:
    __all__.append('DiRePyTorch')
if HAS_UTILS:
    __all__.append('dire_utils')
