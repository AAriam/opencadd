
from typing import Sequence, Any, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd

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
        pocket_atom_serials: Sequence[int] | None = None,
        trim: bool = True
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
            batch_ndim = batch if isinstance(batch, int | jnp.integer) else len(batch)
            if batch_ndim + 3 != tensor.ndim:
                raise ValueError(
                    "The number of batch dimensions must be exactly 3 less "
                    "than the number of dimensions of the tensor, "
                    f"but got {batch_ndim} batch dimensions for a {tensor.ndim}D tensor."
                )
        if trim:
            tensor, deltas = arrayer.tensor.ensure_padding(
                tensor=tensor,
                axes=tuple(range(batch_ndim, tensor.ndim)),
                padding=0,
                pad_value=False,
            )
            origin_shift = jnp.array([delta[0] for delta in deltas]) * grid.spacings
            grid = scids.grid.from_anchor_shape_spacing(
                shape=tensor.shape[batch_ndim:],
                spacing=grid.spacings,
                anchor_type="lower",
                anchor=grid.lower_bounds - origin_shift,
            )
        super().__init__(tensor=tensor, grid=grid, batch=batch)
        tensor_dialated, deltas = arrayer.tensor.ensure_padding(
            tensor=tensor,
            axes=tuple(range(batch_ndim, tensor.ndim)),
            padding=3,
            pad_value=False,
        )
        origin_shift = jnp.array([delta[0] for delta in deltas]) * grid.spacings
        self._tensor_dialated = tensor_dialated.astype(jnp.uint8)
        self._grid_dialated = scids.grid.from_anchor_shape_spacing(
            shape=self._tensor_dialated.shape[batch_ndim:],
            spacing=grid.spacings,
            anchor_type="lower",
            anchor=grid.lower_bounds - origin_shift,
        )
        self._receptor = receptor
        self._pocket_atom_serials = jnp.asarray(pocket_atom_serials) if pocket_atom_serials is not None else None
        if self._pocket_atom_serials is not None and self._receptor is not None:
            serials = set(self._pocket_atom_serials.tolist())
            atoms = self._receptor.composition.atoms
            mask = atoms["serial"].isin(serials)
            self._pocket_atoms = atoms[mask]
            self._pocket_atom_indices = jnp.asarray(mask.to_numpy().nonzero()[0])
        else:
            self._pocket_atoms = None
            self._pocket_atom_indices = None
        return

    @property
    def receptor(self) -> ChemicalSystem | None:
        """Receptor associated with the pocket."""
        return self._receptor

    @property
    def atom_indices(self) -> jnp.ndarray | None:
        """Indices of atoms that make up the pocket."""
        return self._pocket_atom_indices

    @property
    def atom_serials(self) -> jnp.ndarray | None:
        """Serials of atoms that make up the pocket."""
        return self._pocket_atom_serials

    @property
    def atoms(self) -> pd.DataFrame | None:
        """Atoms that make up the pocket."""
        if self._pocket_atoms is not None:
            return self._pocket_atoms.copy()
        return None

    def point_coverage(self, points: ArrayLike):
        points = jnp.asarray(points)
        if points.ndim != 2:
            raise ValueError(
                "Expected points to be a 2D array with shape (n_points, n_dims), "
                f"but got a {points.ndim}D array with shape {points.shape}."
            )
        indices, distances, is_inside = self.grid.nearest_point(points)
        idx_tuple = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
        return np.logical_and(is_inside, self.tensor[..., *idx_tuple])

    def nearest_point(self, points: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Find the nearest pocket point for each point in the input.

        Parameters
        ----------
        points
            An array of shape `(..., 3)`,
            containing the coordinates of points in the same space as the pocket.

        Returns
        -------
        indices
            An array of shape `(*self.batch_shape, ..., 3)`
            containing the indices of the nearest pocket point in each pocket instance
            for each point in the input.
        distances
            An array of shape `(*self.batch_shape, ...)` containing the distances
            from each point in the input to its nearest pocket point in each pocket instance.
        """
        points = jnp.asarray(points)
        if points.ndim < 1 or points.shape[-1] != self.grid.dimension:
            raise ValueError(
                f"Input points must have at least one dimension and "
                f"the last dimension must have size {self.grid.dimension}, "
                f"but input had shape {points.shape}."
            )

        # Grid coordinates: (n_x, n_y, n_z, 3)
        coords = jnp.asarray(self.grid.coordinates)
        if coords.shape[-1] != points.shape[-1]:
            raise ValueError(
                f"Coordinate dimensionality mismatch: grid has {coords.shape[-1]}, "
                f"points have {points.shape[-1]}."
            )
        n_x, n_y, n_z, _ = coords.shape

        # Sanity: each batch slice must have at least one True voxel.
        has_any = jnp.any(self.tensor, axis=(-3, -2, -1))  # shape: *B
        if bool(np.asarray(~has_any).any()):
            raise ValueError("At least one batch instance has no pocket voxels (no True values).")

        B_shape = self.tensor.shape[:-3]        # batch shape (may be empty)
        num_B = len(B_shape)

        # Pairwise squared distances between points and all grid coords.
        # Result shape before adding batch: (..., n_x, n_y, n_z)
        diffs = points[..., None, None, None, :] - coords[None, ...]
        d2 = jnp.sum(diffs * diffs, axis=-1)

        # Add leading batch dims so d2 matches mask's leading batch dims.
        if num_B:
            d2 = jnp.reshape(d2, (1,) * num_B + d2.shape)  # (*B, ..., n_x, n_y, n_z)

        # Correct broadcasting of the boolean mask:
        # insert singleton axes for the point dims BETWEEN batch and spatial dims.
        mask = jnp.asarray(self.tensor)  # (*B, n_x, n_y, n_z)
        extra_point_dims = d2.ndim - mask.ndim
        if extra_point_dims < 0:
            raise RuntimeError("Unexpected rank mismatch while broadcasting mask.")
        mask_b = mask.reshape((*B_shape, *(1,) * extra_point_dims, n_x, n_y, n_z))  # (*B, ..., n_x, n_y, n_z)

        # Mask non-pocket voxels with +inf so they can't win the argmin.
        masked_d2 = jnp.where(mask_b, d2, jnp.inf)

        # Argmin over flattened spatial dimension.
        point_shape = masked_d2.shape[num_B:-3]
        flat = jnp.reshape(masked_d2, (*B_shape, *point_shape, n_x * n_y * n_z))   # (*B, ..., N)
        argmin_flat = jnp.argmin(flat, axis=-1)                                     # (*B, ...)
        min_d2 = jnp.take_along_axis(flat, argmin_flat[..., None], axis=-1)[..., 0] # (*B, ...)

        # Unravel flat indices back to (i, j, k).
        ii, jj, kk = jnp.unravel_index(argmin_flat, (n_x, n_y, n_z))                # each (*B, ...)
        indices = jnp.stack((ii, jj, kk), axis=-1)                                   # (*B, ..., 3)
        distances = jnp.sqrt(min_d2)                                                # (*B, ...)

        return indices, distances

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        show_box: bool = True,
        show_pocket_atoms: bool = False,
        name: str = "Pocket",
        box_name: str = "BBox",
        contour: bool = False,
        wireframe: bool = True,
        visible: bool = True,
        lazy: bool = True,
        opacity: float = 0.8,
        color: tuple[float, float, float] = (0.8, 0.2, 0.2),
        receptor: Any | Literal[False] | None = None,
        gui: bool = True,
    ):
        nv = nglwidget or scishow.nglview.NGLWidget()
        if receptor is not False:
            if receptor is not None:
                nv.add_trajectory(receptor)
            elif self._receptor is not None:
                nv.add_trajectory(self._receptor)
            atom_indices = self._pocket_atom_indices
            if atom_indices is not None:
                nv.add_representation(
                    repr_type="spacefill",
                    selection=atom_indices,
                    visible=show_pocket_atoms,
                )
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
        if gui:
            nv.display(gui=True)
        return nv

    def to_dict(self) -> dict[str, list]:
        """Convert the pocket to a serializable dictionary representation.

        The dictionary can be used to recreate the `Field` object
        using the `from_data()` function.

        Returns
        -------
        Dictionary contains the following keys:
        - "shape": shape of the grid as a list of integers.
        - "size": size of the grid as a list of floats.
        - "spacing": spacing between grid points as a list of floats.
        - "lower": lower bounds of the grid as a list of floats.
        - "upper": upper bounds of the grid as a list of floats.
        - "dtype": data type of the field values as a string.
        - "batch": information about the batch dimensions.
        - "tensor": Field values as a list of lists.
        """
        dictionary = super().to_dict(dtype=jnp.uint8) | {
            "pocket_atom_serials": self._pocket_atom_serials.tolist() if self._pocket_atom_serials is not None else None,
        }
        dictionary.pop("dtype")
        return dictionary

    def to_npz(
        self,
        filepath: str | None = None,
        compress: bool = False,
    ) -> dict[str, Any]:
        """Save the pocket to a .npz file."""
        kwds = {"pocket_atom_serials": self._pocket_atom_serials} if self._pocket_atom_serials is not None else {}
        return super().to_npz(
            filepath=filepath,
            kwds=kwds,
            compress=compress,
        )
