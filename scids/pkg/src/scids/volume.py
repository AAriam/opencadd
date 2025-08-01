from __future__ import annotations

from typing import TYPE_CHECKING
import itertools

import jax.numpy as jnp
import numpy as np

from scids import dataset, exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from jax.typing import ArrayLike


class AxisAlignedRectangularCuboid(dataset.DataSet):
    """An axis-aligned rectangular cuboid form in n-dimensional Euclidean space.

    This can be a line, rectangle, rectangular cuboid,
    hyper-rectangular cuboid, etc.,
    sampled at one or several instances.

    Parameters
    ----------
    lower_bounds
        Lower bounds of the cuboid(s)
        as an array of shape `(..., n_dimensions)`.
        This is the coordinates of the vertex/vertices
        with the smallest value in each dimension.
    upper_bounds
        Upper bounds of the cuboid(s)
        as an array of shape `(..., n_dimensions)`.
        This is the coordinates of the vertex/vertices
        with the largest value in each dimension.
    """

    def __init__(
        self,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
    ):
        lower_bounds = jnp.asarray(lower_bounds)
        upper_bounds = jnp.asarray(upper_bounds)
        if lower_bounds.shape != upper_bounds.shape:
            raise exception.InputError(
                name="upper_bounds",
                message="Lower and upper bounds must have the same shape, "
                        f"but got {lower_bounds.shape} and {upper_bounds.shape}."
            )
        data = jnp.stack([lower_bounds, upper_bounds], axis=-2)
        super().__init__(data=data, batch=batch or lower_bounds.ndim - 1)
        return

    @property
    def lower_bounds(self) -> jnp.ndarray:
        """Lower bounds of the cuboid(s)."""
        return self._data[..., 0, :]

    @property
    def upper_bounds(self) -> jnp.ndarray:
        """Upper bounds of the cuboid(s)."""
        return self._data[..., 1, :]

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
        lower_bounds = self.lower_bounds.reshape(-1, self.lower_bounds.shape[-1])
        upper_bounds = self.upper_bounds.reshape(-1, self.upper_bounds.shape[-1])
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
        return corners.reshape(*self.lower_bounds.shape[:-1], n_corners, n_dimensions)

    def __repr__(self) -> str:
        if self._batch_instance_labels:
            batch = []
            for batch_dim_label, batch_instance_labels in self._batch_instance_labels.items():
                batch.append((batch_dim_label, batch_instance_labels.tolist()))
        else:
            batch = self.batch_ndim
        lines = [
            "RectangularCuboid(",
            f"    lower_bounds={self.lower_bounds},",
            f"    upper_bounds={self.upper_bounds},",
            f"    batch={batch},",
            ")"
        ]
        return "\n".join(lines)
