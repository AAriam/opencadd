"""Toxel field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import scipy as sp

from scids import dataset, exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from jax.typing import ArrayLike, DTypeLike
    from scids.grid import Grid


class Field(dataset.DataSet):
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
    batch
        Information about the batch dimensions.
        This can either be the number of batch dimensions as an integer,
        or a sequence of dimension data for each batch axis.
        If a sequence is provided, its length must match the number of batch axes.
        Each element of the sequence can be:
        - A string representing the label of the axis.
        - A 2-tuple, where the first element is a string
          representing the label of the axis,
          and the second element is a sequence of strings
          representing the labels for each instance along that axis.
    """

    def __init__(
        self,
        tensor: ArrayLike,
        grid: Grid,
        batch: int | Sequence[str | tuple[str, Sequence[str]]],
    ):
        super().__init__(data=tensor, batch=batch)
        self._grid = grid
        if self.tensor.ndim < (self.batch_ndim + self.grid.dimension):
            raise exception.InputError(
                name="tensor",
                message="Tensor dimension must be greater than or equal "
                        "to the sum of grid and prefix dimensions, "
                        f"but got a {self.tensor.ndim}D tensor for "
                        f"a {self._batch_ndim}D batch and a {self.grid.dimension}D grid."
            )
        if np.any(self.grid.shape != self.tensor.shape[self.batch_ndim:self.batch_ndim + self.grid.dimension]):
            raise exception.InputError(
                name="tensor",
                message="The spatial shape of the tensor must be equal to the shape of the grid, "
                        f"but the tensor has a spatial shape of {self.tensor.shape[self.batch_ndim:self.batch_ndim + self.grid.dimension]}, "
                        f"while the shape of the grid is {self.grid.shape}."
            )
        self._field_ndim = self.tensor.ndim - self.batch_ndim - self.grid.dimension
        self._field_shape = self.tensor.shape[self.batch_ndim + self.grid.dimension:]
        self._field_size = np.prod(self._field_shape)
        return

    @property
    def grid(self) -> Grid:
        """The grid on which the field is sampled."""
        return self._grid

    @property
    def tensor(self) -> jnp.ndarray:
        """The tensor containing the entire field values."""
        return self._data

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

    def holes(self) -> jnp.ndarray:
        """Get the holes in the field.

        This treats the field as a binary mask,
        where `True` values represent filled points
        and `False` values represent holes.
        A hole is then defined as a region of `False` values
        that is fully surrounded by `True` values.

        Returns
        -------
        A boolean array of the same shape as the field,
        where `True` values represent holes in the field.
        """
        volume = self.tensor.astype(bool)
        volume_filled = sp.ndimage.binary_fill_holes(volume)
        volume_empty = jnp.logical_not(volume)
        return jnp.logical_and(volume_empty, volume_filled)

    def nearest_target_distances(
        self,
        predicate: callable[[np.ndarray, np.ndarray], np.ndarray],
        direction_vectors: np.ndarray | None = None,
        vector_multipliers: Sequence[int] | int | None = None,
    ) -> np.ndarray:
        """Get distances to nearest neighbor elements within the tensor.

        For each element in the tensor, this method caclulates
        the minimum number of steps `m` for each direction vector
        such that moving `m` times along that direction leads to a target element
        within bounds of the tensor where `predicate(element, target)` is True.
        If no such position is found, the step count is 0.

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

        dists = np.zeros(
            shape=(*self.tensor.shape, direction_vectors.shape[0]),
            dtype=np.uintc
        )

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
            curr_mask = np.ones(shape=self.tensor.shape, dtype=bool)
            for mult in range(1, max_mult[idx_dir]):
                start_slice, end_slice = slicer(mult * direction)
                reached_target = predicate(self.tensor[start_slice], self.tensor[end_slice])
                dists[(*start_slice, idx_dir)][curr_mask[start_slice]] = reached_target[curr_mask[start_slice]] * mult
                curr_mask[start_slice][reached_target] = False
        return dists

    def grid_direction_vectors(self, dimensions: int | Sequence[int] | None = None) -> np.ndarray:
        """Get (a subset of) direction vectors in the grid.

        This returns the same direction vectors as
        `self.grid.direction_vectors(dimensions=dimensions)`,
        but with each vector padded with leading zeros
        to match the prefix dimensions of the tensor.
        """
        return np.pad(
            self._grid.direction_vectors(dimensions=dimensions),
            pad_width=((0, 0), (self.batch_ndim, self.field_ndim)),
            mode="constant",
            constant_values=0,
        )


def from_tensor(
    tensor: ArrayLike,
    grid: Grid,
    batch: int | Sequence[str | tuple[str, Sequence[str]]],
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
    batch
        Information about the batch dimensions.
        This can either be the number of batch dimensions as an integer,
        or a sequence of dimension data for each batch axis.
        If a sequence is provided, its length must match the number of batch axes.
        Each element of the sequence can be:
        - A string representing the label of the axis.
        - A 2-tuple, where the first element is a string
          representing the label of the axis,
          and the second element is a sequence of strings
          representing the labels for each instance along that axis.
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
                message="Expected a sequence of integers with maximum length "
                        "equal to the number of dimensions of the tensor, "
                        f"but got {axis_order} with length {len(axis_order)} "
                        f"and tensor with {tensor.ndim} dimensions."
            )
        destination = tuple(range(len(axis_order)))
        tensor = jnp.moveaxis(tensor, source=tuple(axis_order), destination=destination)
    return Field(tensor=tensor, grid=grid, batch=batch)
