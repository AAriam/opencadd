"""Toxel field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from scids import exception
from scids.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal
    from jax.typing import ArrayLike, DTypeLike
    from scids.grid import Grid


class Field:
    """One or several tensor fields in Euclidean space.

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
    """

    def __init__(
        self,
        tensor: ArrayLike,
        grid: Grid,
        prefix: int | Sequence[str | tuple[str, Sequence[str]]],
    ):
        self._tensor = jnp.asarray(tensor)
        self._grid = grid
        self._prefix_ndim = prefix if isinstance(prefix, int) else len(prefix)
        self._prefix_shape = self.tensor.shape[:self.prefix_ndim]
        self._prefix_size = np.prod(self.prefix_shape)
        if self.tensor.ndim < (self.prefix_ndim + self.grid.dimension):
            raise exception.InputError(
                name="tensor",
                message="Tensor dimension must be greater than or equal to the sum of grid and prefix dimensions, "
                        f"but got a {self.tensor.ndim}D tensor for a {self._prefix_ndim}D prefix and a {self.grid.dimension}D grid."
            )
        if np.any(self.grid.shape != self.tensor.shape[self.prefix_ndim:self.prefix_ndim + self.grid.dimension]):
            raise exception.InputError(
                name="tensor",
                message="The spatial shape of the tensor must be equal to the shape of the grid, "
                        f"but the tensor has a spatial shape of {self.tensor.shape[self.prefix_ndim:self.prefix_ndim + self.grid.dimension]}, "
                        f"while the shape of the grid is {self.grid.shape}."
            )
        self._field_ndim = self.tensor.ndim - self.prefix_ndim - self.grid.dimension
        self._field_shape = self.tensor.shape[self.prefix_ndim + self.grid.dimension:]
        self._field_size = np.prod(self._field_shape)
        self._prefix_dim_labels = []
        self._prefix_instance_labels = {}
        if isinstance(prefix, int):
            return
        for prefix_idx, prefix_data in enumerate(prefix):
            if isinstance(prefix_data, str):
                self._prefix_dim_labels.append(prefix_data)
                continue
            prefix_dim_label, prefix_instance_labels = prefix_data
            self._prefix_dim_labels.append(prefix_dim_label)
            if len(prefix_instance_labels) != self.prefix_shape[prefix_idx]:
                raise exception.InputError(
                    name="prefix",
                    message="The number of prefix instances must match the shape of the tensor along the prefix dimension, "
                            f"but got {len(prefix_instance_labels)} instances for prefix dimension {prefix_idx} with size {self.prefix_shape[prefix_idx]}."
                )
            self._prefix_instance_labels[prefix_dim_label] = np.array(prefix_instance_labels)
        self._prefix_dim_labels = np.array(self._prefix_dim_labels)
        return

    @property
    def grid(self) -> Grid:
        """The grid on which the field is sampled."""
        return self._grid

    @property
    def tensor(self) -> jnp.ndarray:
        """The tensor containing the entire field values."""
        return self._tensor

    @property
    def prefix_ndim(self) -> int:
        """Number of prefix dimensions."""
        return self._prefix_ndim

    @property
    def prefix_shape(self) -> np.ndarray:
        """Shape of the prefix dimensions."""
        return np.array(self._prefix_shape)

    @property
    def prefix_size(self) -> int:
        """Size of the prefix dimensions.

        This represents the total number of field instances.
        """
        return self._prefix_size

    @property
    def field_ndim(self) -> int:
        """Number of field dimensions."""
        return self._field_ndim

    @property
    def field_shape(self) -> np.ndarray:
        """Shape of the field dimensions."""
        return np.array(self._field_shape)

    @property
    def field_size(self) -> int:
        """Size of the field dimensions.

        This represents the total number of values for each grid point.
        """
        return self._field_size

    def nearest_target_distances(
        self,
        predicate: callable[[np.ndarray, np.ndarray], np.ndarray],
        direction_vectors: np.ndarray | None = None,
        vector_multipliers: Sequence[int] | int | None = None,
    ) -> np.ndarray:
        """Get distances to nearest neighbor elements within the tensor.

        For each element in the tensor, this method caclulates
        the minimum number of steps `m >= 0` for each direction vector
        such that moving `m` times along that direction leads to an element
        `target` within bounds of the tensor where `predicate(tensor[current], tensor[target])`
        is True. If no such position is found, the step count is 0.

        Parameters
        ----------
        predicate
            A function `predicate(a, b)` that takes two NumPy arrays `a` and `b`
            of the same shape and returns a boolean NumPy array of the same shape.
            The function should return True if `b` is considered a target for `a`,
            and False otherwise.
            Examples of such functions include:
            - `numpy.logical_xor`: to find the nearest opposite element in a binary array.
        direction_vectors
            A 2D integer array of shape (k, n),
            containing k direction vectors in an n-dimensional space,
            where n is the number of tensor dimensions.
            Each row of the array represents a direction vector,
            i.e. the number of elements to travers along each dimension
            to find the first neighboring point in that direction.
            They must be (positive or negative) integers.
            For example in 3D space, [1,0,0] is along the positive x-axis
            and [0,-1, 0] is along the negative y-axis,
            whereas [1,1,0] goes to the next diagonal point along positive xy-plane.
        vector_multipliers
            Maximum multipliers for direction vectors,
            i.e. maximum number of times to travel along each direction
            to find the neighbor of an element, before terminating the search.
            This can be a single integer used for all direction vectors,
            or a sequence of k integers, one for each direction vector.
            If not provided, search will continue until one edge of the array is reached.

        Returns
        -------
        An (n+1)D array of integers, where the first n dimensions match the shape of
        the tensor, and the last dimension has k elements, each describing the
        distance to the nearest neighbor along the corresponding direction vector in input.
        The values are all integers, and correspond to the number of times to
        travel along the corresponding direction vector to reach the nearest element where predicate is True.
        The value is 0 for directions where no such element was found
        (because either the end of the dimension or `vector_multipliers` was reached).

        Notes
        -----
        This method can be useful in various applications, such as:
        - morphological path-based filtering in image processing (e.g. geodesic transforms)
        - directional distance transforms in computational geometry
        - vector-distance map computation in voxel/grid data analysis
        - anisotropic BFS/flood-fills along constrained axes
        """

        def slicer(vec: np.ndarray) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
            """Calculate start and end slices for a given displacement vector.

            For a given displacement vector (i.e. direction vector times a multiplier),
            calculate two tuples of slices, which index the starting elements
            and end elements of that displacement on the array.
            """
            start_slices, end_slices = [], []
            for val in vec:
                if val > 0:
                    start_slices.append(slice(None, -val))
                    end_slices.append(slice(val, None))
                elif val < 0:
                    start_slices.append(slice(-val, None))
                    end_slices.append(slice(None, val))
                else:
                    for lis in [start_slices, end_slices]:
                        lis.append(slice(None, None))
            return tuple(start_slices), tuple(end_slices)

        if direction_vectors is None:
            direction_vectors = self.grid_direction_vectors()

        dists = np.zeros(shape=(*self.tensor.shape, direction_vectors.shape[0]), dtype=np.uintc)

        # Calculate the maximum multiplier along each direction:
        # First, calculate the maximum possible multipliers
        with np.errstate(divide="ignore", invalid="ignore"):
            max_mult_axis = (np.array(self.tensor.shape) - 1) / np.abs(direction_vectors)
            max_mult_axis[np.isnan(max_mult_axis)] = np.inf
            max_mult_dir = np.min(max_mult_axis, axis=-1)
        # Then, compare with user-input multipliers and take the smaller one in each direction.
        if vector_multipliers is None:
            vector_multipliers = np.ones(direction_vectors.shape[0]) * np.max(self.tensor.shape)
        elif isinstance(vector_multipliers, int):
            vector_multipliers = np.ones(direction_vectors.shape[0]) * vector_multipliers
        max_mult = np.min((max_mult_dir, vector_multipliers), axis=0).astype(int) + 1

        # Loop through directions, and for each direction through multipliers, and calculate
        # distances between starting elements and end elements.
        for idx_dir, direction in enumerate(direction_vectors):
            curr_mask = np.ones_like(self.tensor)
            for mult in range(1, max_mult[idx_dir]):
                start_slice, end_slice = slicer(mult * direction)
                reached_target = predicate(self.tensor[start_slice], self.tensor[end_slice])
                dists[(*start_slice, idx_dir)][curr_mask[start_slice]] = reached_target[curr_mask[start_slice]] * mult
                curr_mask[start_slice][reached_target] = 0
        return dists

    def grid_direction_vectors(self, dimensions: Sequence[int] | None = None) -> np.ndarray:
        """Get (a subset of) direction vectors in the grid.

        This returns the same direction vectors as
        `self.grid.direction_vectors(dimensions=dimensions)`,
        but with each vector padded with leading zeros
        to match the prefix dimensions of the tensor.
        """
        return np.pad(
            self._grid.direction_vectors(dimensions=dimensions),
            pad_width=((0, 0), (self.prefix_ndim, self.field_ndim)),
            mode="constant",
            constant_values=0,
        )

    def __call__(self, **kwargs) -> jnp.ndarray:
        if not self._prefix_instance_labels:
            raise exception.InputError(
                name="prefix",
                message="Prefix dimension labels are not set. "
                        "Please provide a prefix dimension label to index the tensor."
            )
        index = []
        for prefix_label in self._prefix_dim_labels:
            if prefix_label not in kwargs:
                index.append(slice(None))
                continue
            instance_labels = self._prefix_instance_labels.get(prefix_label, [])
            selection = kwargs[prefix_label]
            if isinstance(selection, str):
                selection_idx = np.argwhere(instance_labels == selection)
                if selection_idx.size == 0:
                    raise exception.InputError(
                        name="prefix",
                        message=f"Prefix instance label '{selection}' is not valid for prefix dimension '{prefix_label}'. "
                                f"Valid labels are: {instance_labels}."
                    )
                index.append(selection_idx[0][0])
            else:
                index.append(selection)
        return self._tensor[*index]

    # @property
    # def field_names(self):
    #     return self._field_names

    # def index_field(self, name: Any | Sequence[Any]) -> np.ndarray:
    #     names = np.asarray(name).reshape(-1, 1)
    #     indices = np.argwhere(self._field_names == names)
    #     if indices.shape[0] != names.size:
    #         ind_bad_names = np.setdiff1d(np.arange(names.size), indices[:, 0])
    #         raise IndexError(
    #             f"Following field names are not valid: {names[ind_bad_names]}. "
    #             f"Valid field names are: {self.field_names}."
    #         )
    #     return np.squeeze(indices[:, 1])

    # def calculate_vacancy(
    #     self,
    #     energy_cutoff: float = +0.6,
    #     mode: Literal["max", "min", "avg", "sum"] | None = "min",
    # ) -> np.ndarray:
    #     """
    #     Calculate whether each grid point is vacant, or occupied by a target atom.

    #     Parameters
    #     ----------
    #     energy_cutoff : float, Optional, default: +0.6
    #         Cutoff value for energy; grid points with energies lower than cutoff are considered
    #         vacant.
    #     mode: Literal["max", "min", "avg", "sum"], Optional, default: "min"
    #         If the energy of more than one ligand type is to be compared, this parameter defines
    #         how those different energy values must be processed, before comparing with the cutoff.
    #     ligand_types : Sequence[opencadd.consts.autodock.Autodock4AtomType], Optional, default: None
    #         A subset of ligand types that were used to initialize the object, whose energy values
    #         are to be taken as reference for calculating the vacancy of each grid point. If not
    #         set to None, then all ligand interaction energies are considered.

    #     Returns
    #     -------
    #     vacancy : numpy.ndarray[dtype=numpy.bool_, shape=T2FPharm.grid.shape[:-1]]
    #         A 4-dimensional boolean array matching the first four dimensions of `T2FPharm.grid`,
    #         indicating whether each grid point is vacant (True), or occupied (False).
    #         Vacant grid points can easily be indexed by `T2FPharm.grid[vacancy]`.
    #     """
    #     # The reducing operations corresponding to each `mode`:
    #     red_fun = {"max": np.max, "min": np.min, "avg": np.mean, "sum": np.sum}
    #     # Get index of input ligand types
    #     # if ligand_types is None:
    #     #     ind = slice(None)
    #     # else:
    #     #     ind = np.argwhere(np.expand_dims(ligand_types, axis=-1) == self._probe_types)[:, 1]
    #     #     # Verify that all input ligand types are valid
    #     #     if len(ind) != len(ligand_types):
    #     #         raise ValueError(f"Some of input energies were not calculated.")
    #     # Reduce the given references using the given operation.
    #     energy_vals = red_fun[mode](self._interaction_field.van_der_waals, axis=-1)
    #     # Apply cutoff and return
    #     self._vacancy = energy_vals < energy_cutoff
    #     return self._vacancy

    # def __getitem__(self, item):
    #     return self._tensor.__getitem__(item)
    #     # if isinstance(item, int) or (isinstance(item, tuple) and isinstance(item[-1], int)):
    #     #
    #     # if isinstance(item, str):
    #     #     return self._tensor.__getitem__(..., index_of_label(item))
    #     # elif isinstance(item, tuple) and isinstance(item[-1], (str, Sequence, np.ndarray)):
    #     #     return self._tensor.__getitem__(*item[:-1], index_of_label(item))
    #     # else:

    # # def display_nglview(
    #     self,
    #     widget: NGLWidget,

    #     representation_type: Literal["surface", "dot", "slice"] = "surface",
    #     representation_params: SurfaceRepresentationParameters | None = None,
    # ) -> NGLWidget:
    #     widget.add_volume(
    #         data=self.tensor[],
    #         basis=self.grid.unit_vectors,
    #         origin=self.grid.lower_bounds,
    #         representation_type=representation_type,
    #         representation_params=representation_params,
    #     )
    #     return widget


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
