
from typing import Sequence, Any, Literal

import jax.numpy as jnp

import arrayer
import scids
from scids.field import Field
from scids.grid import Grid
import scishow
from caddpy.chemsys import ChemicalSystem
from caddpy.typing import ArrayLike


class Pocket(Field):
    """Binding pocket."""

    def __init__(
        self,
        tensor: ArrayLike,
        grid: Grid,
        batch: Sequence[str | tuple[str, Sequence[str]]] | None = None,
        receptor: ChemicalSystem | None = None,
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
        new_origin = grid.lower_bounds - origin_shift
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
        self._receptor = receptor
        return

    def point_coverage(self, points: ArrayLike):
        points = jnp.asarray(points)
        if points.ndim != 2:
            raise ValueError(
                "Expected points to be a 2D array with shape (n_points, n_dims), "
                f"but got a {points.ndim}D array with shape {points.shape}."
            )
        indices, distances, is_inside = self.grid.nearest_point(points)
        idx_tuple = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
        return jnp.logical_and(is_inside, self.tensor[..., *idx_tuple])

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        show_box: bool = False,
        name: str = "Pocket",
        box_name: str = "BBox",
        contour: bool = False,
        wireframe: bool = True,
        visible: bool = True,
        lazy: bool = True,
        opacity: float = 0.8,
        color: tuple[float, float, float] = (0.8, 0.2, 0.2),
        receptor: Any | Literal[False] | None = None,
    ):
        nv = nglwidget or scishow.nglview.NGLWidget()
        if receptor is not False:
            if receptor is not None:
                nv.add_trajectory(receptor)
            elif self._receptor is not None:
                nv.add_trajectory(self._receptor)
        if show_box:
            nv.add_box(
                lower_bounds=self.grid.lower_bounds,
                upper_bounds=self.grid.upper_bounds,
                name=box_name,
            )
        nv.add_volume(
            data=self._tensor_dialated,
            basis=self._grid_dialated.unit_vectors,
            origin=self._grid_dialated.lower_bounds,
            name=name,
            representation_params=scishow.nglview.SurfaceRepresentationParameters(
                isolevel=0.5,
                isolevel_type="value",
                contour=contour,
                wireframe=wireframe,
                color=color,
                opacity=opacity,
                visible=visible,
                lazy=lazy,
            )
        )
        return nv
