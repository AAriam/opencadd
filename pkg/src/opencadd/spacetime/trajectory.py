
import jax.numpy as jnp

from opencadd._typing import ArrayLike
from opencadd.spacetime.spatial import Spatial


class Trajectory:
    def __init__(self, coordinates: ArrayLike):
        self._coordinates: jnp.ndarray
        self._spatials: list[Spatial]

        coordinates_array = jnp.asarray(coordinates)
        return
