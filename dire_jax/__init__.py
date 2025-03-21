
# dire-jax

"""
A JAX-based dimensionality reducer.
"""

from .dire import DiRe  # core class import

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

__all__ = ['DiRe', 'dire_utils'] if HAS_UTILS else ['DiRe']
