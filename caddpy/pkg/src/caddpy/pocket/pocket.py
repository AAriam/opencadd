
from typing import Sequence

import jax.numpy as jnp

import arrayer
import scids
from scids.field import Field
from scids.grid import Grid
import scishow

from caddpy.typing import ArrayLike

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
        tensor, deltas = arrayer.tensor.ensure_padding(
            tensor=tensor,
            axes=tuple(range(batch_ndim, tensor.ndim)),
            padding=0,
            pad_value=False,
        )
        origin_shift = jnp.array([delta[0] for delta in deltas]) * grid.spacings
        new_origin = grid.lower_bounds + origin_shift
        grid = scids.grid.from_anchor_shape_spacing(
            shape=tensor.shape[batch_ndim:],
            spacing=grid.spacings,
            anchor_type="lower",
            anchor_coord=new_origin,
        )
        super().__init__(tensor=tensor, grid=grid, batch=batch)
        tensor_dialated, deltas = arrayer.tensor.ensure_padding(
            tensor=tensor,
            axes=tuple(range(batch_ndim, tensor.ndim)),
            padding=3,
            pad_value=False,
        )
        self._tensor_dialated = tensor_dialated.astype(jnp.uint8)
        self._grid_dialated = scids.grid.from_anchor_shape_spacing(
            shape=self._tensor_dialated.shape[batch_ndim:],
            spacing=grid.spacings,
            anchor_type="lower",
            anchor_coord=grid.lower_bounds - 3 * grid.spacings,
        )
        return

    def display(
        self,
        nglwidget=None,
        name: str = "Pocket",
        contour: bool = True,
        visible: bool = True,
        color: tuple[float, float, float] = (0.8, 0.2, 0.2)
    ):
        nv = nglwidget or scishow.nglview.NGLWidget()
        nv.add_volume(
            data=self._tensor_dialated,
            basis=self._grid_dialated.unit_vectors,
            origin=self._grid_dialated.lower_bounds,
            name=name,
            representation_params=scishow.nglview.SurfaceRepresentationParameters(
                isolevel=0.5,
                isolevel_type="value",
                contour=contour,
                color=color,
                visible=visible,
            )
        )
        return nv
