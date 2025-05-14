"""SciDS: Scientific Data Structures and file formats"""

import jax
# Enable 64-bit floating point precision for JAX
# See: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
jax.config.update("jax_enable_x64", True)

from scids import field, grid, pointcloud, volume


__all__ = [
    "field",
    "grid",
    "pointcloud",
    "volume",
]
