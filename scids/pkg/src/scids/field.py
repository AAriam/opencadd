"""Toxel field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import scids
from scids import exception
from scids.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal
    import numpy.typing as npt
    from pathlib import Path
    from scids.grid import Grid


class ToxelField:
    """Toxel field.

    A collection of n scalar fields, or one m-dimensional vector field,
    sampled over time (or any other dimension)
    at regularly spaced points on a n-dimensional grid in Euclidean space.

    Parameters
    ----------
    tensor
        An (n + 2)-dimensional array (where n is the number of grid dimensions)
        representing the field values for each grid point at different times.
        The first dimension represents time, the next n dimensions represent
        the grid points along their respective axes,
        and the last dimension represents the field values.
        In each dimension, the elements should be ordered from the smallest index to largest.
    grid
        A Grid object representing the grid on which the field is sampled.
    names
        Optional labels for each field value.
        If given, the grid can then be indexed using labels as well.
    """

    def __init__(
        self,
        tensor: ArrayLike,
        grid: Grid,
        names: Sequence[Any] | None = None,
        autodock_map_file_headers: Sequence[Sequence[scids.file.autodock_map.AutodockMapFileOptionalHeader]] | None = None,
    ):
        # Check for errors in input arguments.
        # _exceptions.raise_for_type(self.__class__.__name__, ("grid", grid, Grid))
        tensor = jnp.asarray(tensor)
        # _exceptions.raise_array(
        #     parent_name=self.__class__.__name__, param_name="tensor", array=tensor, ndim_gt=2
        # )
        if grid.dimension != tensor.ndim - 2:
            raise ValueError(
                "Dimension Mismatch:\n"
                "An n-dimensional `tensor` requires an (n - 2)-dimensional `grid`, "
                f"but the input tensor was {tensor.ndim}-dimensional, while the grid "
                f"was {grid.dimension}-dimensional."
            )
        if np.any(grid.shape != tensor.shape[1:-1]):
            raise ValueError(
                "Shape Mismatch:\n"
                "The spatial shape of the input tensor (i.e. `tensor.shape[1:-1]`) "
                f"must be equal to the shape of the grid, but the input tensor had "
                f"a spatial shape of {tensor.shape[1:-1]}, while the shape of the grid "
                f"was {grid.shape}."
            )
        if names is None:
            names = np.arange(tensor.shape[-1])
        else:
            names = np.asarray(names)
            # _exceptions.raise_array(
            #     parent_name=self.__class__.__name__, param_name="names", array=names, ndim_eq=1
            # )
            if names.size != tensor.shape[-1]:
                raise ValueError(
                    "Shape Mismatch:\n"
                    "The size of `names` must be equal to the shape of `tensor` along its last dimension, "
                    f"but the input array had {names.size} elements, while the last dimension of `tensor` "
                    f"had {tensor.shape[-1]} elements."
                )

        self._tensor: jnp.ndarray = jnp.asarray(tensor)
        self._grid: Grid = grid
        self._field_names = names
        self._autodock_map_file_headers = autodock_map_file_headers
        return

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def tensor(self):
        """The 5-dimensional array representing the field values for each grid point at different times.

        Returns
        -------
        numpy.ndarray
            A 5-dimensional array of shape (n_t, n_x, n_y, n_z, n_f), with
            n_t: temporal length of the field.
            n_x, n_y, n_z: number of grid points along x, y, and z directions.
            n_f: number of field values for each point.
        """
        return self._tensor

    def spatial_direction_vectors(self, dimensions=None):
        return np.pad(
            self._grid.direction_vectors(dimensions=dimensions),
            pad_width=((0, 0), (1, 0)),
            mode="constant",
            constant_values=0,
        )

    def __call__(self, name=None, instance=slice(None)):
        idx_fields = slice(None) if name is None else self.index_field(name)
        return self._tensor[instance, ..., idx_fields]

    @property
    def temporal_length(self) -> int:
        """
        Number of times the field values were sampled, e.g. at different times or different
        environments.
        """
        return self._tensor.shape[0]

    @property
    def field_names(self):
        return self._field_names

    @property
    def fields_count(self) -> int:
        return self._tensor.shape[-1]

    def index_field(self, name: Any | Sequence[Any]) -> np.ndarray:
        names = np.asarray(name).reshape(-1, 1)
        indices = np.argwhere(self._field_names == names)
        if indices.shape[0] != names.size:
            ind_bad_names = np.setdiff1d(np.arange(names.size), indices[:, 0])
            raise IndexError(
                f"Following field names are not valid: {names[ind_bad_names]}. "
                f"Valid field names are: {self.field_names}."
            )
        return np.squeeze(indices[:, 1])

    def calculate_vacancy(
        self,
        energy_cutoff: float = +0.6,
        mode: Literal["max", "min", "avg", "sum"] | None = "min",
    ) -> np.ndarray:
        """
        Calculate whether each grid point is vacant, or occupied by a target atom.

        Parameters
        ----------
        energy_cutoff : float, Optional, default: +0.6
            Cutoff value for energy; grid points with energies lower than cutoff are considered
            vacant.
        mode: Literal["max", "min", "avg", "sum"], Optional, default: "min"
            If the energy of more than one ligand type is to be compared, this parameter defines
            how those different energy values must be processed, before comparing with the cutoff.
        ligand_types : Sequence[opencadd.consts.autodock.Autodock4AtomType], Optional, default: None
            A subset of ligand types that were used to initialize the object, whose energy values
            are to be taken as reference for calculating the vacancy of each grid point. If not
            set to None, then all ligand interaction energies are considered.

        Returns
        -------
        vacancy : numpy.ndarray[dtype=numpy.bool_, shape=T2FPharm.grid.shape[:-1]]
            A 4-dimensional boolean array matching the first four dimensions of `T2FPharm.grid`,
            indicating whether each grid point is vacant (True), or occupied (False).
            Vacant grid points can easily be indexed by `T2FPharm.grid[vacancy]`.
        """
        # The reducing operations corresponding to each `mode`:
        red_fun = {"max": np.max, "min": np.min, "avg": np.mean, "sum": np.sum}
        # Get index of input ligand types
        # if ligand_types is None:
        #     ind = slice(None)
        # else:
        #     ind = np.argwhere(np.expand_dims(ligand_types, axis=-1) == self._probe_types)[:, 1]
        #     # Verify that all input ligand types are valid
        #     if len(ind) != len(ligand_types):
        #         raise ValueError(f"Some of input energies were not calculated.")
        # Reduce the given references using the given operation.
        energy_vals = red_fun[mode](self._interaction_field.van_der_waals, axis=-1)
        # Apply cutoff and return
        self._vacancy = energy_vals < energy_cutoff
        return self._vacancy

    def __getitem__(self, item):
        return self._tensor.__getitem__(item)
        # if isinstance(item, int) or (isinstance(item, tuple) and isinstance(item[-1], int)):
        #
        # if isinstance(item, str):
        #     return self._tensor.__getitem__(..., index_of_label(item))
        # elif isinstance(item, tuple) and isinstance(item[-1], (str, Sequence, np.ndarray)):
        #     return self._tensor.__getitem__(*item[:-1], index_of_label(item))
        # else:

    def visualize(self):
        pass


def from_autodock_map(
    files: list[list[str | bytes | Path]],
    field_dtype: np.dtype = np.single,
    field_names: Sequence[Any] | None = None,
    strict: bool = True,
    file_labels: list[list[str]] | None = None,
) -> ToxelField:
    """Create a ToxelField from one or several AutoDock MAP files.

    Parameters
    ----------
    files
        MAP file contents or paths.
    field_dtype
        Numpy datatype of the output array.
        Default is 32-bit float (numpy.single).
    field_names
        Labels for the fields.
    strict
        Treat any parsing problems as errors.
        If False, only critical problems are raised as errors,
        and all other problems are reported as warnings.
    filepath
        Path to the MAP file.
        This is used for error reporting only.
    """
    # Parse the first file to get the grid shape first
    first_map = scids.file.autodock_map.parse(
        file=files[0][0],
        field_dtype=field_dtype,
        strict=strict,
        file_label=file_labels[0][0] if file_labels else None,
    )

    # Create the grid
    grid_shape = first_map.nelements + 1
    grid = scids.grid.from_center_spacing_shape(
        center=first_map.center,
        spacings=first_map.spacing,
        shape=grid_shape
    )

    # Create the field tensor
    time_point_count = len(files)
    field_count = len(files[0])
    toxel_field_shape = (time_point_count, *grid_shape, field_count)
    fields = np.empty(shape=toxel_field_shape, dtype=field_dtype)
    fields[0, ..., 0] = first_map.field

    # Create array for unique headers
    headers_shape = (time_point_count, field_count)
    headers = np.empty(shape=headers_shape, dtype=object)

    # Parse the rest of the files
    for idx_instance, instance in enumerate(files):
        for idx_map, file in enumerate(instance[1:], start=1):
            file_label = file_labels[idx_instance][idx_map] if file_labels else f"[{idx_instance}, {idx_map}]"
            mapfile = scids.file.autodock_map.parse(
                file=file,
                field_dtype=field_dtype,
                strict=strict,
                file_label=file_label,
            )
            fields[idx_instance, ..., idx_map] = mapfile.field
            headers[idx_instance, idx_map] = scids.file.autodock_map.AutodockMapFileOptionalHeader(
                grid_parameter_file=mapfile.grid_parameter_file,
                grid_data_file=mapfile.grid_data_file,
                macromolecule=mapfile.macromolecule,
            )
            # Check for consistency in the header values
            for key in ("center", "nelements", "spacing"):
                if getattr(mapfile, key) != getattr(first_map, key):
                    err_msg = (
                        f"Header '{key.upper()}' values do not match across MAP files. "
                        f"Expected {getattr(first_map, key)}, but got {getattr(mapfile, key)} "
                        f"for MAP file {file_label}."
                    )
                    exception.ScidsReadError(
                        file_type="autodock_map",
                        message=err_msg,
                        filepath=file_label,
                        content=mapfile,
                    )
    return ToxelField(
        tensor=fields,
        grid=grid,
        names=field_names,
    )


def from_tensor_grid(
    tensor,
    grid,
    names,
):
    return ToxelField(tensor=tensor, grid=grid, names=names)


def from_tensor(
    tensor: ArrayLike,
    grid: Grid,
    prefix: int | Sequence[str | tuple[str, Sequence[str]]],
    axis_order: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
):
    """Create a field from an array-like object.

    Parameters
    ----------
    tensor
        An `(n_prefix + n_dim + n_field)`-dimensional array containing the field values.
        The first `n_prefix >= 0` dimensions represent prefix dimensions,
        along which different instances of the field are sampled.
        The next `n_dim >= 1` dimensions represent spatial dimensions of the field,
        which must match the dimensions of the grid.
        The last `n_field >= 0` dimensions represent the field values for each grid point.
        In each dimension, the elements should be ordered from the smallest index to largest.
    grid
        The grid on which the field is sampled.
    prefix
        Information about the prefix dimensions.
        This can either be the number of prefix dimensions as an integer,
        or a sequence of dimension data for each prefix dimension.
        If a sequence is provided, its length must match the number of prefix dimensions.
        Each element of the sequence can be:
        - A string representing the label of the dimension.
        - A 2-tuple, where the first element is a string representing the label of the dimension,
          and the second element is a sequence of strings
          representing the labels of the prefix dimension's instances.
    axis_order
        Order of the axes in the input tensor.
        This is a sequence of integers, where each integer
        represents the index of the axis in the input tensor.
        The order of the integers in the sequence determines
        the new order of the axes in the output tensor.
        The length of the sequence must be equal
        to the number of dimensions in the input tensor.
        If not provided, the axes will be ordered as they are in the input tensor.
    dtype
        Datatype of the field values.
        If not provided, the datatype will be inferred from the input tensor.
    """
    tensor = jnp.asarray(tensor, dtype=dtype)
    if axis_order:
        if len(axis_order) > tensor.ndim:
            raise exception.InputError(
                name="order_axes",
                message="Expected a sequence of integers with maximum length equal to the number of dimensions of the tensor, "
                        f"but got {axis_order} with length {len(axis_order)} and tensor with {tensor.ndim} dimensions."
            )
        destination = tuple(range(len(axis_order)))
        tensor = jnp.moveaxis(tensor, source=tuple(axis_order), destination=destination)
    return Field(tensor=tensor, grid=grid, prefix=prefix)
