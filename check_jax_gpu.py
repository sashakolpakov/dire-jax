#!/usr/bin/env python3

import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

# Test computation
x = jnp.ones((1000, 1000))
y = x @ x
y.block_until_ready()
print(f"Test computation device: {y.devices()}")

# Check if GPU is being used
try:
    from jax.lib import xla_bridge
    print(f"XLA backend: {xla_bridge.get_backend().platform}")
except:
    pass