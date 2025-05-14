from __future__ import annotations

from typing import TYPE_CHECKING
import itertools

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from scids.grid import Grid
    from scids.typing import ArrayLike


class RectangularCuboid:
    """An n-dimensional rectangular cuboid.

    This can be a line, rectangle, rectangular cuboid,
    hyper-rectangular cuboid, etc.,
    sampled at one or several instances.
    """

    def __init__(
        self,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
    ):
        self._lower_bounds = jnp.asarray(lower_bounds)
        self._upper_bounds = jnp.asarray(upper_bounds)
        if self._lower_bounds.ndim != 2 or self._upper_bounds.ndim != 2:
            raise ValueError("Input arrays must be 2-dimensional.")
        if self._lower_bounds.shape != self._upper_bounds.shape:
            raise ValueError("Input arrays must have the same shape.")
        return

    @property
    def lower_bounds(self) -> jnp.ndarray:
        return self._lower_bounds

    @property
    def upper_bounds(self) -> jnp.ndarray:
        return self._upper_bounds

    @property
    def volume(self) -> jnp.ndarray:
        """Volume of the cuboid."""
        return jnp.prod(self.upper_bounds - self.lower_bounds, axis=1)

    @property
    def corners(self) -> np.ndarray:
        """Coordinates of all corners of the cuboid.

        Returns
        -------
        np.ndarray
            Array of shape (n_instances, 2 ** n_dimensions, n_dimensions) containing the coordinates
            of all corners for each cuboid.
        """
        n_instances, n_dimensions = self.lower_bounds.shape
        n_corners = 2 ** n_dimensions
        # Generate binary selector for corners, shape (n_corners, n_dimensions)
        corner_mask = np.array(list(itertools.product([0, 1], repeat=n_dimensions)), dtype=int)[None, :, :]
        # Broadcast lower and upper bounds to shape (n_instances, n_corners, n_dimensions)
        lower = np.broadcast_to(self.lower_bounds[:, None, :], (n_instances, n_corners, n_dimensions))
        upper = np.broadcast_to(self.upper_bounds[:, None, :], (n_instances, n_corners, n_dimensions))
        # Use corner_mask to choose between lower and upper bounds
        corners = np.where(corner_mask, upper, lower)
        return corners

    def __repr__(self) -> str:
        lines = [
            "RectangularCuboid(",
            f"    lower_bounds={self.lower_bounds},",
            f"    upper_bounds={self.upper_bounds},",
            ")"
        ]
        return "\n".join(lines)
