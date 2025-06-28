
from typing import Sequence

import jax.numpy as jnp
from caddpy.typing import ArrayLike
from scids.field import Field
from scids.grid import Grid


class Pocket(Field):
    """Binding pocket."""

    def __init__(
        self,
        tensor: ArrayLike,
        grid: Grid,
        batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
    ):
        tensor = jnp.asarray(tensor, dtype=bool)
        if tensor.ndim < 3:
            raise ValueError(
                "Excepted at least a 3D array, "
                f"but got a {tensor.ndim}D array with shape {tensor.shape}: {tensor}"
            )
        if batch is None:
            batch = batch_ndim = tensor.ndim - 3
        else:
            batch_ndim = len(batch)
            if batch_ndim + 3 != tensor.ndim:
                raise ValueError(
                    "The number of batch dimensions must be exactly 3 less "
                    "than the number of dimensions of the tensor, "
                    f"but got {batch_ndim} batch dimensions for a {tensor.ndim}D tensor."
                )
        super().__init__(tensor=tensor, grid=grid, batch=batch)
        return
