"""An n-dimensional grid of points in euclidean space."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import scids

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import ArrayLike


class Grid:
    """An n-dimensional grid of points in euclidean space.

    Parameters
    ----------
    shape
        Shape of the grid, i.e. number of points in each dimension.
    size
        Length of the grid in each dimension.
    lower_bounds
        Coordinates of the point with minimum values in all dimensions.
    center
        Coordinates of the geometric center of the grid.
    upper_bounds
        Coordinates of the point with maximum values in all dimensions.
    spacings
        Spacing between grid points in each dimension.
        grid delta, i.e. distance between two adjacent grid points
    mgrid
        Fleshed out meshgrid of grid point coordinates.
    """

    def __init__(
        self,
        shape: np.ndarray,
        size: np.ndarray,
        spacings: np.ndarray,
        lower_bounds: np.ndarray,
        center: np.ndarray,
        upper_bounds: np.ndarray,
        mgrid: np.ndarray,
    ):
        self._shape = shape
        self._size = size
        self._lower_bounds = lower_bounds
        self._center = center
        self._upper_bounds = upper_bounds
        self._spacings = spacings
        self._mgrid = jnp.asarray(mgrid)

        self._coordinates: jnp.ndarray = jnp.stack(mgrid, axis=-1)
        self._point_count = np.prod(self._shape)
        self._indices: np.ndarray = np.array(list(np.ndindex(*self._shape))).reshape(
            *self._shape, -1
        )
        self._pointcloud = scids.pointcloud.PointCloud(
            points=self._coordinates.reshape(self._point_count, self.dimension)
        )
        # All possible combinations of -1, 0, and 1 in n dimensions
        self._direction_vectors = np.array(
            list(itertools.product([-1, 0, 1], repeat=self.dimension))
        )
        # Manhattan distance for each direction vector
        self._direction_vectors_dimension = np.count_nonzero(self._direction_vectors, axis=-1)
        return

    @property
    def center(self) -> np.ndarray:
        """Coordinates of the center of the grid."""
        return np.array(self._center)

    @property
    def coordinates(self) -> jnp.ndarray:
        """Coordinates of all grid points.

        For an n-dimensional grid, this is an (n+1)-dimensional array,
        where the first n dimensions have the same shape as the grid,
        and the last dimension has size n, containing the coordinates of each grid point.
        The array is ordered in the same way as the grid points,
        and thus the individual grid points can be indexed
        using their coordinates in unit vectors.
        That is, coordinates [i, j, k] gives the coordinates of the point
        located at position (i, j, k) on the grid.
        """
        return self._coordinates

    @property
    def coordinates_2d(self) -> jnp.ndarray:
        """Coordinates of all grid points as a 2D array.

        For an n-dimensional grid of shape (x, y, z),
        this is a 2D array of shape (x * y * z, n),
        where each row contains the coordinates of a grid point.
        This is a flattened version of the `coordinates` property,
        where the first n dimensions are flattened into a single dimension.
        """
        return self.points.points_2d

    @property
    def dimension(self) -> int:
        """Mumber of axes in the grid."""
        return self._shape.size

    @property
    def indices(self) -> np.ndarray:
        """Indices of all grid points.

        This is similar to the `coordinates` property, but instead of coordinates,
        it contains the indices of each grid point in the grid.
        That is, indices [i, j, k] returns (i, j, k).
        This is useful, for example, to quickly obtain the indices of
        some grid points using a boolean mask.
        """
        return self._indices

    @property
    def lower_bounds(self) -> np.ndarray:
        """Coordinates of the origin point of the grid.

        This is the point where all indices are zero.
        """
        return np.array(self._lower_bounds)

    @property
    def point_count(self) -> int:
        """Total number of points in the grid."""
        return self._point_count

    @property
    def points(self) -> scids.pointcloud.PointCloud:
        """DynamicPointCloud object representing the grid points."""
        return self._pointcloud

    @property
    def shape(self) -> np.ndarray:
        """Number of grid points in each dimension."""
        return np.array(self._shape)

    @property
    def size(self) -> np.ndarray:
        """Length of the grid in each dimension."""
        return np.array(self._size)

    @property
    def spacings(self) -> np.ndarray:
        """Distance between two adjacent points along each dimension."""
        return np.array(self._spacings)

    @property
    def unit_vectors(self) -> np.ndarray:
        """Unit vectors along each dimension.

        For an n-dimensional grid, this is a 2D array of shape (n, n),
        where i-th row contains the unit vector along the i-th dimension.
        It corresponds to a diagonal matrix with `spacings` as diagonal elements.
        """
        return self.coordinates[tuple(np.eye(self.dimension, dtype=int))] - self.coordinates_2d[0]

    @property
    def upper_bounds(self) -> np.ndarray:
        """Coordinates of the point with maximum index values in all dimensions."""
        return np.array(self._upper_bounds)

    def index_coordinates(self, indices: ArrayLike) -> jnp.ndarray:
        """Get coordinates of grid points given their indices.

        Parameters
        ----------
        indices
            Indices of the grid points to get coordinates for.
            This can be a single index or an array of indices.
            The indices need not be integers.

        Returns
        -------
        Coordinates of the grid points with the given indices.
        """
        return self.lower_bounds + indices * self.spacings

    def direction_vectors(self, dimensions: int | Sequence[int] | None = None) -> np.ndarray:
        """Get (a subset of) direction vectors in the grid.

        These vectors represent possible relative movement directions
        in the n-dimensional discrete grid,
        from one grid point to an adjacent grid point.

        For example, in a 2D grid, the full set of direction vectors and
        their corresponding 'dimension' values are:

        - Dimension 0:
            - `[0, 0]` (no movement)
        - Dimension 1:
            - `[-1, 0]` (left)
            - `[0, -1]` (down)
            - `[0, 1]` (up)
            - `[1, 0]` (right)
        - Dimension 2:
            - `[-1, -1]` (left-down)
            - `[-1, 1]` (left-up)
            - `[1, -1]` (right-down)
            - `[1, 1]` (right-up)

        Parameters
        ----------
        dimensions
            Desired counts of non-zero components
            in the direction vectors to include.
            This filters direction vectors from the set {-1, 0, 1}^d
            by their discrete L₀ norm.
            If None, defaults to the range [1, n],
            which includes all directions with at least one non-zero component and
            at most n non-zero components (i.e., all non-zero directions).

        Returns
        -------
        An array of shape (a, n), where each row is an n-dimensional vector
        from the set {-1, 0, 1}^d that has a number of non-zero components
        in the given set `dimensions`.
        """
        if dimensions is None:
            dimensions = np.arange(1, self.dimension + 1)
        return self._direction_vectors[np.isin(self._direction_vectors_dimension, dimensions)]

    def __repr__(self):
        rep = (
            f"Grid(\n  shape={self._shape},\n  size={self._size},\n  spacings={self._spacings},\n  "
            f"lower_bounds={self._lower_bounds},\n  center={self._center},\n  upper_bounds={self._upper_bounds}\n)"
        )
        return rep


def from_anchor_shape_size(
    shape: Sequence[float],
    size: Sequence[float],
    anchor_coord: Sequence[float] = None,
    anchor: Literal["lower", "center", "upper"] | Sequence[int] = "center",
):
    shape = np.asarray(shape)
    size = np.asarray(size)
    num_spacings = shape - 1
    spacings = size / num_spacings
    return from_anchor_shape_spacing(
        shape=shape, spacing=spacings, anchor_coord=anchor_coord, anchor_type=anchor
    )


def from_anchor_shape_spacing(
    shape: Sequence[float],
    spacing: Sequence[float] | float,
    anchor_type: Literal["lower", "center", "upper"] | Sequence[int] = "center",
    anchor_coord: Sequence[float] = None,
):
    shape = np.asarray(shape)
    spacing = np.array([spacing] * shape.size) if np.isscalar(spacing) else np.asarray(spacing)
    num_spacings = shape - 1
    size = num_spacings * spacing
    anchor_coord = np.zeros(shape=shape.size) if anchor_coord is None else np.asarray(anchor_coord)
    if anchor_coord.size != shape.size:
        raise ValueError(
            f"Parameter `anchor_coord` must have the same size as `shape`, "
            f"but input argument had size {anchor_coord.size} and shape {shape.size}. "
            f"Input was: {anchor_coord}"
        )
    if anchor_type == "center":
        lower_bounds = anchor_coord - size / 2
        upper_bounds = anchor_coord + size / 2
    elif anchor_type == "lower":
        lower_bounds = anchor_coord
        upper_bounds = anchor_coord + size
    elif anchor_type == "upper":
        lower_bounds = anchor_coord - size
        upper_bounds = anchor_coord
    else:
        anchor_type = np.asarray(anchor_type)
        lower_bounds = anchor_coord - anchor_type * spacing
        upper_bounds = lower_bounds + size
    return from_bounds_shape(lower_bounds=lower_bounds, upper_bounds=upper_bounds, shape=shape)


def from_anchor_size_spacing(
    size: Sequence[float],
    spacings: Sequence[float],
    anchor_coord: Sequence[float] = None,
    anchor: Literal["lower", "center", "upper"] | Sequence[int] = "center",
    shrink_to_fit: bool = False,
):
    size = np.asarray(size)
    spacings = np.asarray(spacings)
    num_spacings = size / spacings
    fit_func = np.floor if shrink_to_fit else np.ceil
    return from_anchor_shape_spacing(
        shape=fit_func(num_spacings + 1).astype(int),
        spacing=spacings,
        anchor_coord=anchor_coord,
        anchor_type=anchor,
    )


def from_bounds_shape(
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    shape: Sequence[int],
) -> Grid:
    """Create a Grid from its bounds and shape.

    All arguments must be 1D arrays/sequences of the same size.

    Parameters
    ----------
    lower_bounds
        Coordinates of the point with minimum index values in all dimensions.
    upper_bounds
        Coordinates of the point with maximum index values in all dimensions.
    shape
        Shape of the grid, i.e. number of points in each dimension.
    """
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    shape = np.asarray(shape)
    for bound, arg_name in zip(
        (lower_bounds, upper_bounds, shape),
        ("lower_bounds", "upper_bounds", "shape"),
        strict=True,
    ):
        if bound.ndim != 1:
            raise ValueError(
                f"Parameter `{arg_name}` expects a 1D array, "
                f"but input argument had {bound.ndim} dimensions. Input was: {bound}"
            )
    for bound, arg_name in zip(
        (lower_bounds, upper_bounds),
        ("lower_bounds", "upper_bounds"),
        strict=True,
    ):
        if not (np.issubdtype(bound.dtype, np.floating) or np.issubdtype(bound.dtype, np.integer)):
            raise ValueError(
                f"Parameter `{arg_name}` expects an array of real numbers, "
                f"but input argument had elements of type {bound.dtype}. Input was: {bound}"
            )
    if not np.issubdtype(shape.dtype, np.integer):
        raise ValueError(
            f"Parameter `shape` expects an array of integers, "
            f"but input argument had elements of type {shape.dtype}. Input was: {shape}"
        )
    for arg, arg_name in zip((upper_bounds, shape), ("upper_bounds", "shape"), strict=True):
        if lower_bounds.size != arg.size:
            raise ValueError(
                f"Parameters `lower_bound` and `{arg_name}` expect 1D arrays of same size, "
                f"but input argument `lower_bounds` had size {lower_bounds.size}, "
                f"while `{arg_name}` had size {arg.size}. "
                f"Inputs were: `lower_bounds` = {lower_bounds}\n`{arg_name}` = {arg}."
            )
    size = upper_bounds - lower_bounds
    size_is_invalid = size <= 0
    if np.any(size_is_invalid):
        raise ValueError(
            "All values in `lower_bounds` must be strictly smaller "
            f"than corresponding values in `upper_bounds`, but at indices {np.where(size_is_invalid)[0]} "
            f"`lower_bounds` had values {lower_bounds[size_is_invalid]} and `upper_bounds` had "
            f"values {upper_bounds[size_is_invalid]}."
        )
    shape_is_invalid = shape <= 0
    if np.any(shape_is_invalid):
        raise ValueError(
            "All values in `shape` must be positive integers, "
            f"but at indices {np.where(shape_is_invalid)[0]} "
            f"`shape` had values {shape[shape_is_invalid]}."
        )
    slices = tuple(
        slice(start, end, complex(num_points))
        for start, end, num_points in zip(lower_bounds, upper_bounds, shape, strict=True)
    )
    return Grid(
        shape=shape,
        size=size,
        lower_bounds=lower_bounds,
        center=(lower_bounds + upper_bounds) / 2,
        upper_bounds=upper_bounds,
        spacings=size / (shape - 1),
        mgrid=np.mgrid[slices],
    )


def from_bounds_spacing(
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    spacings: Sequence[float],
    shrink_to_fit: bool = False,
) -> Grid:
    """
    Create a `Grid` from its lower- and upper bounds, and spacings.

    Parameters
    ----------
    lower_bounds : sequence of float
        Coordinates of the point with minimum values in all dimensions.
    upper_bounds : sequence of float
        Coordinates of the point with maximum values in all dimensions.
        This must have the same length as `lower_bounds`.
    spacings : sequence of int
        Spacing between grid points in each dimension.
        This must have the same length as `lower_bounds` and `upper_bounds`.

    Returns
    -------
    Grid
    """
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    spacings = np.asarray(spacings)
    size = upper_bounds - lower_bounds
    num_spacings = size / spacings
    fit_func = np.floor if shrink_to_fit else np.ceil
    return from_bounds_shape(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        shape=fit_func(num_spacings + 1).astype(int),
    )
