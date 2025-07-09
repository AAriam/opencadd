
from typing import Sequence, Any, Literal

import jax.numpy as jnp
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
            anchor=new_origin,
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
            anchor=grid.lower_bounds - 3 * grid.spacings,
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
        return jnp.logical_and(is_inside, self.tensor[..., *idx_tuple])

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        show_box: bool = False,
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
