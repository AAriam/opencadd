from __future__ import annotations

from typing import TYPE_CHECKING
import itertools

import jax.numpy as jnp
import numpy as np

from scids import exception

if TYPE_CHECKING:
    from scids.typing import ArrayLike


class AxisAlignedRectangularCuboid:
    """An axis-aligned rectangular cuboid form in n-dimensional Euclidean space.

    This can be a line, rectangle, rectangular cuboid,
    hyper-rectangular cuboid, etc.,
    sampled at one or several instances.

    Parameters
    ----------
    lower_bounds
        Lower bounds of the cuboid(s)
        as an array of shape `(n_dimensions,)`
        or `(n_instances, n_dimensions)`.
        This is the coordinates of the vertex/vertices
        with the smallest value in each dimension.
    upper_bounds
        Upper bounds of the cuboid(s)
        as an array of shape `(n_dimensions,)`
        or `(n_instances, n_dimensions)`.
        This is the coordinates of the vertex/vertices
        with the largest value in each dimension.
    """

    def __init__(
        self,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
    ):
        self._lower_bounds = jnp.asarray(lower_bounds)
        self._upper_bounds = jnp.asarray(upper_bounds)
        if self._lower_bounds.shape != self._upper_bounds.shape:
            raise exception.InputError(
                name="upper_bounds",
                message="Lower and upper bounds must have the same shape, "
                        f"but got {self._lower_bounds.shape} and {self._upper_bounds.shape}."
            )
        if self._lower_bounds.ndim not in (1, 2):
            raise exception.InputError(
                name="lower_bounds",
                message=f"Bounds must be 1D or 2D, but got {self._lower_bounds.ndim}D."
            )
        return

    @property
    def lower_bounds(self) -> jnp.ndarray:
        """Lower bounds of the cuboid(s)."""
        return self._lower_bounds

    @property
    def upper_bounds(self) -> jnp.ndarray:
        """Upper bounds of the cuboid(s)."""
        return self._upper_bounds

    @property
    def size(self) -> jnp.ndarray:
        """Size of the cuboid(s)."""
        return jnp.abs(self.upper_bounds - self.lower_bounds)

    @property
    def volume(self) -> jnp.ndarray:
        """Volume of the cuboid(s)."""
        return jnp.prod(self.size, axis=-1)

    @property
    def corners(self) -> np.ndarray:
        """Coordinates of all corners of the cuboid(s).

        Returns
        -------
        Array of shape `(2 ** n_dimensions, n_dimensions)`
        or `(n_instances, 2 ** n_dimensions, n_dimensions)`
        containing the coordinates of all corners for each cuboid.
        """
        # Generalize to 2D case
        lower_bounds = jnp.atleast_2d(self.lower_bounds)
        upper_bounds = jnp.atleast_2d(self.upper_bounds)
        # Calculate dimensions
        n_instances, n_dimensions = lower_bounds.shape
        n_corners = 2 ** n_dimensions
        # Broadcast lower and upper bounds to shape (n_instances, n_corners, n_dimensions)
        lower, upper = [
            np.broadcast_to(bounds[:, None, :], (n_instances, n_corners, n_dimensions))
            for bounds in (lower_bounds, upper_bounds)
        ]
        # Generate binary selector for corners, shape (n_corners, n_dimensions)
        corner_mask = np.array(list(itertools.product([0, 1], repeat=n_dimensions)), dtype=int)[None, :, :]
        # Use corner_mask to choose between lower and upper bounds
        corners = np.where(corner_mask, upper, lower)
        # Return corners in the original shape
        if self.lower_bounds.ndim == 1:
            return corners[0]
        return corners

    def __repr__(self) -> str:
        lines = [
            "RectangularCuboid(",
            f"    lower_bounds={self.lower_bounds},",
            f"    upper_bounds={self.upper_bounds},",
            ")"
        ]
        return "\n".join(lines)
