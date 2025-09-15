"""An n-dimensional grid of points in euclidean space."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import scids

if TYPE_CHECKING:
    from typing import Literal, Self
    from numpy.typing import ArrayLike


class Grid:
    """An n-dimensional grid of points in Euclidean space.

    Parameters
    ----------
    shape
        Shape of the grid, i.e. number of points in each dimension.
    size
        Length of the grid in each dimension.
    spacing
        Spacing between grid points in each dimension.
        grid delta, i.e. distance between two adjacent grid points
    lower
        Coordinates of the point with minimum values in all dimensions.
    upper
        Coordinates of the point with maximum values in all dimensions.
    """

    def __init__(
        self,
        shape: np.ndarray,
        size: np.ndarray,
        spacing: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ):
        self._shape = shape
        self._size = size.astype(np.float64)
        self._spacing = spacing.astype(np.float64)
        self._lower = lower.astype(np.float64)
        self._upper = upper.astype(np.float64)

        self._center = (self._lower + self._upper) / 2

        slices = tuple(
            slice(start, end, complex(num_points))
            for start, end, num_points in zip(self._lower, self._upper, self._shape, strict=True)
        )
        self._mgrid = jnp.asarray(np.mgrid[slices])
        self._coordinates: jnp.ndarray = jnp.stack(self._mgrid, axis=-1)

        self._point_count = np.prod(self._shape)
        self._point_volume = np.prod(self._spacing)
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
        """Number of axes in the grid."""
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
        return np.array(self._lower)

    @property
    def point_count(self) -> int:
        """Total number of points in the grid."""
        return self._point_count

    @property
    def point_volume(self) -> float:
        """Volume of a single grid point.

        This is the volume of the hypercube defined by the grid spacings.
        For an n-dimensional grid, this is the product of the spacings in all dimensions.
        """
        return self._point_volume

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
        return np.array(self._spacing)

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
        return np.array(self._upper)

    @property
    def unique_distances(self) -> jnp.ndarray:
        """Unique distances between all pairs of grid points, sorted from lowest to highest."""
        shape = self.shape
        spacings = self.spacings
        dimension = self.dimension

        # Build weighted squared-distance grid
        weighted_sq = np.zeros(tuple(shape), dtype=float)
        for axis in range(dimension):
            idx = np.arange(shape[axis], dtype=float)
            contrib = (idx**2) * (spacings[axis]**2)
            bshape = [1]*dimension
            bshape[axis] = shape[axis]
            weighted_sq += contrib.reshape(bshape)

        # Flatten and drop zero
        flat = weighted_sq.ravel()
        flat = flat[flat > 0.0]

        # Sort values
        flat_sorted = np.sort(flat)

        # Determine tolerance for grouping duplicates
        eps = np.finfo(float).eps
        tol = 10 * dimension * eps * flat_sorted.max()

        # Group values within tol
        unique_vals = [flat_sorted[0]]
        for val in flat_sorted[1:]:
            if val - unique_vals[-1] > tol:
                unique_vals.append(val)

        uniq_sq = np.array(unique_vals)
        return np.sqrt(uniq_sq)

    def nearest_point(self, points: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the nearest grid point for each point in the input.

        Parameters
        ----------
        points
            An array of shape `(..., self.dimension)`,
            containing the coordinates of points in the same space as the grid.

        Returns
        -------
        indices
            An array of shape `(..., self.dimension)`
            containing the indices of the nearest grid point
            for each point in the input.
        distances
            An array of shape `(...)` containing the distances
            from each point in the input to the nearest grid point.
        is_inside
            A boolean array of shape `(...)` indicating whether each point
            is inside the grid bounds.
        """
        points = np.asarray(points)
        if points.ndim < 1 or points.shape[-1] != self.dimension:
            raise ValueError(
                f"Input points must have at least one dimension and "
                f"the last dimension must have size {self.dimension}, "
                f"but input had shape {points.shape}."
            )
        uvw = (points - self.lower_bounds) / self.spacings
        is_inside = np.all((uvw >= -0.5) & (uvw <= (self.shape - 0.5)), axis=-1)
        indices = np.clip(np.rint(uvw).astype(int), min=0, max=self.shape - 1)
        self_coords = self.index_coordinates(indices)
        distances = np.linalg.norm(points - self_coords, axis=-1)
        return indices, distances, is_inside

    def index_coordinates(self, indices: ArrayLike) -> np.ndarray:
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

    def common_neighbors_count(
        self,
        neighborhood_radius: float,
        point1: Sequence[int],
        point2: Sequence[int] | None = None,
    ) -> int:
        """Calculate the number of common neighbors for two grid points.

        This method calculates the number of grid points that lie
        within `neighborhood_radius` of both of two grid points
        at indices `point1` and `point2`.

        Parameters
        ----------
        neighborhood_radius
            Maximum radius for a point to be considered a neighbor.
        point1
            Grid index of the first point.
        point2
            Grid index of the second point.
            If not provided, the grid is assumed to be infinite,
            with `point1` being the index offset from the first grid point to the second.
            That is, the number of common neighbors is calculated
            for two grid points at indices `(i, j, k, ...)` and
            `(i + point1[0], j + point1[1], k + point1[2], ...)`.
            If `point2` is provided, `point1` and `point2` must
            be valid grid indices, and the number of common neighbors
            is calculated on the finite grid,
            i.e., two points at grid boundaries will have less common neighbors
            than two points with the same distance but in the middle of the grid.

        Returns
        -------
        Number of grid points (excluding the two seed points) whose Euclidean distance
        to both seeds is less than or equal to `neighborhood_radius`.

        Raises
        ------
        ValueError
            If `neighborhood_radius` is negative,
            any spacing is non-positive,
            lengths of `spacing` and `offset` differ,
            or `offset` is all zeros.
        """
        if neighborhood_radius < 0:
            raise ValueError(f"`neighborhood_radius` must be non-negative, but got {neighborhood_radius}.")

        dim = self.dimension
        spacings = self.spacings

        p1 = np.asarray(point1)
        if p1.ndim != 1 or p1.size != dim or p1.dtype.kind not in "iu":
            raise ValueError(
                f"`point1` must be a 1D integer array of length {dim}, "
                f"but got {p1} with shape {p1.shape} and dtype {p1.dtype}."
            )

        # Figure out offset and whether we clip to a finite grid
        if point2 is None:
            # infinite grid: point1 is the offset
            offset = p1.copy()
            clip_min = None
            clip_max = None
        else:
            p2 = np.asarray(point2)
            if p2.ndim != 1 or p2.size != dim or p2.dtype.kind not in "iu":
                raise ValueError(
                    f"When provided, `point2` must be a 1D integer array of length {dim}, "
                    f"but got {p2} with shape {p2.shape} and dtype {p2.dtype}."
                )
            shape = self.shape
            # both seeds must lie inside the grid
            if np.any(p1 < 0) or np.any(p1 >= shape):
                raise ValueError(f"`point1` {tuple(p1)} out of bounds for grid shape {tuple(shape)}.")
            if np.any(p2 < 0) or np.any(p2 >= shape):
                raise ValueError(f"`point2` {tuple(p2)} out of bounds for grid shape {tuple(shape)}.")
            offset = p2 - p1
            # To keep grid‐offsets k so that (p1 + k) stays in [0..shape-1]:
            clip_min = -p1
            clip_max = shape - 1 - p1

        if np.all(offset == 0):
            raise ValueError("Seed points must not coincide (offset ≠ 0).")

        # Calculate per-axis discrete distance bounds.
        # These are the maximum grid‑index offsets
        # along each axis needed to reach the radius when moved only along that axis.
        # We'll use these to enumerate integer grid offsets
        # inside each sphere’s bounding hypercube.
        r_discrete = np.floor(neighborhood_radius / spacings).astype(int)

        # Determine bounding box indices for each dimension
        # to fully enclose both hyperspheres.
        lower_bounds = np.minimum(-r_discrete, offset - r_discrete)
        upper_bounds = np.maximum(r_discrete, offset + r_discrete)

        # If finite, clip to grid boundaries
        if clip_min is not None:
            lower_bounds = np.maximum(lower_bounds, clip_min)
            upper_bounds = np.minimum(upper_bounds, clip_max)

        # Generate all integer grid indices within the bounding box.
        # For each axis, grid_point_idx_ranges[i] is all integer indices
        # from lower_bounds[i] to upper_bounds[i].
        grid_point_idx_ranges = [
            np.arange(low, high + 1) for low, high in zip(lower_bounds, upper_bounds)
        ]

        # Generate one n-dimensional array per axis,
        # where each array holds the coordinate along that axis
        # for every point in the hyper-rectangle.
        grid_point_indices_per_axis = np.meshgrid(*grid_point_idx_ranges, indexing='ij')
        # Stack these arrays along a new last axis
        # to get a single array of shape (n_points, n_dimensions).
        grid_point_indices = np.stack([g.flatten() for g in grid_point_indices_per_axis], axis=-1)

        # Compute squared distances to both seeds
        dist1_squared = np.sum((grid_point_indices * spacings) ** 2, axis=1)
        dist2_squared = np.sum(((grid_point_indices - offset) * spacings) ** 2, axis=1)

        # Calculate squared distance to avoid computing square roots
        r_squared = neighborhood_radius ** 2
        # Boolean mask for points within both spheres
        mask = (dist1_squared <= r_squared) & (dist2_squared <= r_squared)
        n_points = int(np.count_nonzero(mask))

        # Both seed points (at indices all‐zeros and at offset)
        # always satisfy the distance test, so they’re in the mask.
        # We subtract 2 to exclude them,
        # and use max(0, ...) to guard against negative counts
        # if, for example, the offset is farther apart than distance
        # (in which case only one or zero seeds lie in the intersection).
        return max(0, n_points - 2)

    def footprint_spherical(self, radius: float) -> np.ndarray:
        """Create a spherical footprint (a.k.a. structuring element).

        This method generates a centrosymmetric
        binary array representing a filled n-sphere
        (i.e., a disc in 2D, a sphere in 3D, and a hypersphere in nD)
        with a given radius.

        The footprint can be used in morphological operations
        on fields corresponding to this grid.

        Parameters
        ----------
        radius
            Radius of the sphere in the same units as the grid's coordinates.
        """
        if radius <= 0:
            raise ValueError("`radius` must be positive.")

        # Calculate integer radius per axis,
        # i.e., number of grid spacings from center to edge along each axis.
        r_discrete = np.rint(radius / self.spacings).astype(int)

        # Calculate output shape, ensuring an odd number of points in each dimension
        # to maintain centrosymmetry.
        shape = tuple(2 * r_discrete + 1)

        # Create open grid of indices for each dimension.
        # coords[i] has shape with length = shape[i] in acis i, 1 elsewhere.
        coords = np.ogrid[tuple(slice(0, n) for n in shape)]

        # For each point, calculate physical (squared) distance from center
        dist2 = np.zeros(shape, dtype=float)
        for grid_coord_i, r_i, spacing_i in zip(coords, r_discrete, self.spacings):
            # shift index so center is at zero, multiply by spacing to get physical
            dist2 += ((grid_coord_i - r_i) * spacing_i) ** 2

        # Threshold against (squared) physical radius
        return dist2 <= (radius ** 2)

    def to_dict(self, array_to_list: bool = True) -> dict[str, list[int] | list[float]]:
        """Convert the grid to a serializable dictionary representation.

        The dictionary can be used to recreate the `Grid` object
        using the `from_data()` function.

        Returns
        -------
        Dictionary contains the following keys:
        - "shape": shape of the grid as a list of integers.
        - "size": size of the grid as a list of floats.
        - "spacing": spacing between grid points as a list of floats.
        - "lower": lower bounds of the grid as a list of floats.
        - "upper": upper bounds of the grid as a list of floats.
        """
        return {
            k: v.tolist() if array_to_list else v
            for k, v in (
                ("shape", self.shape),
                ("size", self.size),
                ("spacing", self.spacings),
                ("lower", self.lower_bounds),
                ("upper", self.upper_bounds),
            )
        }

    def new_aligned_grid(
        self,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        *,
        rounding: Literal["nearest", "expand", "shrink"] = "nearest",
        atol: float = 1e-12,
    ) -> Self:
        """Create a new grid aligned to this grid's infinite lattice.

        The new grid uses **identical spacings and phase** as this grid.
        The requested bounds are snapped to lattice points defined by
        ``self.lower_bounds + k * self.spacings`` (elementwise integer k).
        This ensures that, if both grids extended infinitely, **all points overlap**.

        Parameters
        ----------
        lower
            Requested lower bounds (min corner) as a 1D array of shape (ndim,).
            Values will be snapped to the existing lattice per `rounding`.
            If None, defaults to this grid's lower bounds.
        upper
            Requested upper bounds (max corner) as a 1D array of shape (ndim,).
            Must satisfy ``upper >= lower`` elementwise before snapping.
            Values will be snapped to the existing lattice per `rounding`.
            If None, defaults to this grid's upper bounds.
        rounding
            Lattice snapping rule per dimension:
            - ``"nearest"``: snap each bound to the nearest lattice point.
            - ``"expand"``: choose the **outer** lattice points so the resulting box
              fully contains the requested interval (lower rounds down, upper rounds up).
            - ``"shrink"``: choose the **inner** lattice points so the result is fully
              contained in the requested interval (lower rounds up, upper rounds down).
        atol
            Absolute tolerance used for numerical safety when validating integrality
            of step counts and nonnegativity of extents.

        Returns
        -------
        Grid
            A new grid with the same spacings and phase as this grid, and bounds
            snapped to lattice points. The shape is chosen so that
            ``upper = lower + (shape - 1) * spacings`` holds exactly.

        Raises
        -------
        ValueError
            If inputs have incompatible shapes, if any spacing is non-positive,
            if the requested interval is invalid, or if snapping results in an
            empty dimension.

        Notes
        ------
        - The relationship ``size = (shape - 1) * spacings`` is enforced exactly.
        - Snapping uses ``rint/floor/ceil`` in units of the lattice step to avoid
          floating drift.
        """
        lb = np.asarray(lower, dtype=float) if lower is not None else self.lower_bounds
        ub = np.asarray(upper, dtype=float) if upper is not None else self.upper_bounds
        s = np.asarray(self.spacings, dtype=float)
        origin = np.asarray(self.lower_bounds, dtype=float)

        if lb.ndim != 1 or ub.ndim != 1 or s.ndim != 1 or origin.ndim != 1:
            raise ValueError("All inputs must be 1D arrays.")
        if not (lb.shape == ub.shape == origin.shape):
            raise ValueError("All inputs must have the same shape (ndim,).")
        if np.any(ub < lb - atol):
            raise ValueError("upper must be >= lower elementwise.")

        # Compute integer lattice indices for requested bounds in step units.
        # Work in step coordinates to avoid cumulative float error.
        t_lower = (lb - origin) / s
        t_upper = (ub - origin) / s

        if rounding == "nearest":
            k_lower = np.rint(t_lower)
            k_upper = np.rint(t_upper)
        elif rounding == "expand":
            k_lower = np.floor(t_lower)
            k_upper = np.ceil(t_upper)
        elif rounding == "shrink":
            k_lower = np.ceil(t_lower)
            k_upper = np.floor(t_upper)
        else:
            raise ValueError("rounding must be 'nearest', 'expand', or 'shrink'.")

        # Aligned bounds on the lattice.
        aligned_lower = origin + k_lower * s
        aligned_upper = origin + k_upper * s

        # Ensure non-empty extents after snapping.
        if np.any(aligned_upper < aligned_lower - atol):
            raise ValueError("Snapped bounds are invalid (upper < lower).")

        # Compute number of intervals and points: n = (upper - lower)/s + 1
        intervals = (aligned_upper - aligned_lower) / s
        # Guard against tiny negative zeros due to floating noise.
        intervals = np.where(np.abs(intervals) < atol, 0.0, intervals)

        k_intervals = np.rint(intervals)
        if np.any(np.abs(intervals - k_intervals) > 1e-9):
            # This should not happen due to snapping to lattice; keep a clear error if it does.
            raise ValueError("Non-integer interval count after snapping; check inputs/tolerances.")

        shape_new = (k_intervals.astype(np.int64) + 1)
        if np.any(shape_new < 1):
            raise ValueError("Snapped shape would be empty along at least one dimension.")

        # Recompute size and upper exactly from integer shape to guarantee consistency.
        size_new = (shape_new - 1) * s
        upper_new = aligned_lower + size_new

        # Return a new Grid with identical spacings/phase and aligned bounds.
        return Grid(
            shape=shape_new,
            size=size_new,
            spacing=s.copy(),
            lower=aligned_lower,
            upper=upper_new,
        )

    def overlap_slice(
        self,
        other: Self,
        *,
        atol: float = 1e-12,
        strict_phase: bool = True,
    ) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        """Return numpy indexers extracting the overlapping region between two grids.

        The returned indexing tuples, when applied to arrays shaped like the respective
        grids (``self.shape`` and ``other.shape``), select the **exact** set of points
        that overlap in Euclidean coordinates. If there is no overlap, the slices
        will produce empty arrays (stop ≤ start in at least one dimension).

        Assumes both grids are samples of the same underlying infinite lattice
        (identical spacings and phase). If `strict_phase=True`, this is enforced.

        Parameters
        ----------
        other
            The grid to intersect with.
        atol
            Absolute tolerance for floating comparisons and integrality checks.
        strict_phase
            If True, require that the two grids share identical lattice phase:
            ``(self.lower_bounds - other.lower_bounds) / spacings`` is integer
            (within `atol`). If False, indices are computed geometrically assuming
            equal spacings; non-integer phase may yield empty overlap.

        Returns
        -------
        (self_indexer, other_indexer)
            Two tuples of Python ``slice`` objects, one per dimension, suitable for
            indexing arrays shaped like the corresponding grids.

        Raises
        -------
        ValueError
            If spacings differ; if dimension mismatches; or if `strict_phase` is True
            and lattice phases don't match within tolerance.

        Notes
        ------
        - Coordinates at index i are ``lower_bounds + i * spacings`` (0-based).
        - Each grid includes both lower and upper bounds: upper = lower + (shape-1)*spacing.
        - The slices are half-open [start, stop) as per NumPy semantics.
        """
        # Basic checks
        if self.spacings.shape != other.spacings.shape:
            raise ValueError("Grid dimensionality mismatch.")
        if not np.allclose(self.spacings, other.spacings, rtol=0.0, atol=atol):
            raise ValueError("Grids must have identical spacings for index overlap.")
        s = np.asarray(self.spacings, dtype=float)

        lb1 = np.asarray(self.lower_bounds, dtype=float)
        ub1 = np.asarray(self.upper_bounds, dtype=float)
        n1 = np.asarray(self.shape, dtype=int)

        lb2 = np.asarray(other.lower_bounds, dtype=float)
        ub2 = np.asarray(other.upper_bounds, dtype=float)
        n2 = np.asarray(other.shape, dtype=int)

        if strict_phase:
            phase = (lb1 - lb2) / s
            if not np.allclose(phase, np.rint(phase), rtol=0.0, atol=1e3 * atol):
                raise ValueError(
                    "Grids are not phase-aligned: (lower1 - lower2)/spacing must be integer."
                )

        self_slices: list[slice] = []
        other_slices: list[slice] = []

        # For each dimension, compute inclusive index overlap [i0, i1] and [j0, j1]
        for d in range(s.size):
            sd = s[d]

            # Map other's coord range into self's index space
            t0 = (lb2[d] - lb1[d]) / sd
            t1 = (ub2[d] - lb1[d]) / sd
            # Inclusive bounds in self
            i0 = int(max(0, np.ceil(t0 - atol)))
            i1 = int(min(n1[d] - 1, np.floor(t1 + atol)))

            # Map self's coord range into other's index space
            u0 = (lb1[d] - lb2[d]) / sd
            u1 = (ub1[d] - lb2[d]) / sd
            # Inclusive bounds in other
            j0 = int(max(0, np.ceil(u0 - atol)))
            j1 = int(min(n2[d] - 1, np.floor(u1 + atol)))

            # If either side has no overlap in this dim, return empty slices for both
            if i1 < i0 or j1 < j0:
                self_slices.append(slice(0, 0, 1))
                other_slices.append(slice(0, 0, 1))
                continue

            # Convert inclusive [i0, i1] to half-open [i0, i1+1)
            self_slices.append(slice(i0, i1 + 1, 1))
            other_slices.append(slice(j0, j1 + 1, 1))

        return tuple(self_slices), tuple(other_slices)

    def __eq__(self, other: object) -> bool:
        """Check if two grids are equal.

        Two grids are considered equal if they have the same shape, size,
        lower bounds, center, upper bounds, and spacings.
        """
        if not isinstance(other, Grid):
            return False
        return (
            np.array_equal(self.shape, other.shape)
            and np.array_equal(self.lower_bounds, other.lower_bounds)
            and np.array_equal(self.upper_bounds, other.upper_bounds)
        )

    def __repr__(self):
        rep = (
            f"Grid(\n  shape={self._shape},\n  size={self._size},\n  spacings={self._spacing},\n  "
            f"lower_bounds={self._lower},\n  center={self._center},\n  upper_bounds={self._upper}\n)"
        )
        return rep


def from_anchor_shape_size(
    anchor: Sequence[float],
    shape: Sequence[int] | int,
    size: Sequence[float] | float,
    *,
    anchor_type: Literal["lower", "center", "upper"] | Sequence[int] = "lower",
):
    """Create a `Grid` from an anchor point, shape, and size.

    Parameters
    ----------
    anchor
        Coordinates of the anchor point of the grid.
        The type of anchor point is determined by `anchor_type`.
    shape
        Shape of the grid, i.e. number of points in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `anchor`.
    size
        Physical length of the grid in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `anchor`.
    anchor_type
        Type of anchor point; either the index of a grid point (e.g. [0, 1, 2])
        or one of the following keywords:
        - "lower": the anchor point is the lower bound of the grid.
        - "center": the anchor point is the center of the grid.
        - "upper": the anchor point is the upper bound of the grid.
    """
    anchor = np.asarray(anchor, dtype=np.float128)
    shape = _expand_arg("shape", shape, size=anchor.size)
    size = _expand_arg("size", size, size=anchor.size)
    num_spacings = shape - 1
    spacing = size / num_spacings
    return from_anchor_shape_spacing(
        anchor=anchor, shape=shape, spacing=spacing, anchor_type=anchor_type
    )


def from_anchor_shape_spacing(
    anchor: Sequence[float],
    shape: Sequence[int] | int,
    spacing: Sequence[float] | float,
    *,
    anchor_type: Literal["lower", "center", "upper"] | Sequence[int] = "lower",
):
    """Create a `Grid` from an anchor point, shape, and nominal spacings.

    Parameters
    ----------
    anchor
        Coordinates of the anchor point of the grid.
        The type of anchor point is determined by `anchor_type`.
    shape
        Shape of the grid, i.e. number of points in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `anchor`.
    spacing
        Spacing between grid points in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `anchor`.
    anchor_type
        Type of anchor point; either the index of a grid point (e.g. [0, 1, 2])
        or one of the following keywords:
        - "lower": the anchor point is the lower bound of the grid.
        - "center": the anchor point is the center of the grid.
        - "upper": the anchor point is the upper bound of the grid.
    """
    anchor = np.asarray(anchor, dtype=np.float128)
    shape = _expand_arg("shape", shape, size=anchor.size)
    spacing = _expand_arg("spacing", spacing, size=anchor.size)
    num_spacings = shape - 1
    size = num_spacings * spacing
    if anchor.size != shape.size:
        raise ValueError(
            f"Parameter `anchor` must have the same size as `shape`, "
            f"but input argument had size {anchor.size} and shape {shape.size}. "
            f"Input was: {anchor}"
        )
    if anchor_type == "center":
        lower = anchor - size / 2
        upper = anchor + size / 2
    elif anchor_type == "lower":
        lower = anchor
        upper = anchor + size
    elif anchor_type == "upper":
        lower = anchor - size
        upper = anchor
    else:
        anchor_type = np.asarray(anchor_type, dtype=int)
        lower = anchor - anchor_type * spacing
        upper = lower + size
    return from_bounds_shape(lower=lower, upper=upper, shape=shape)


def from_anchor_size_spacing(
    anchor: Sequence[float],
    size: Sequence[float] | float,
    spacing: Sequence[float] | float,
    *,
    anchor_type: Literal["lower", "center", "upper"] | Sequence[int] = "lower",
    shrink_to_fit: bool = False,
):
    """Create a `Grid` from an anchor point, size, and nominal spacings.

    Parameters
    ----------
    anchor
        Coordinates of the anchor point of the grid.
        The type of anchor point is determined by `anchor_type`.
    size
        Physical length of the grid in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `anchor`.
    spacing
        Spacing between grid points in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `anchor`.
    anchor_type
        Type of anchor point; either the index of a grid point (e.g. [0, 1, 2])
        or one of the following keywords:
        - "lower": the anchor point is the lower bound of the grid.
        - "center": the anchor point is the center of the grid.
        - "upper": the anchor point is the upper bound of the grid.
    shrink_to_fit
        - `True`: ensure the grid fits inside the bounds
           by flooring the number of intervals.
        - `False`: ensure the grid covers the full range
           by ceiling the number of intervals.
    """
    anchor = np.asarray(anchor, dtype=np.float128)
    size = _expand_arg("size", size, size=anchor.size)
    spacing = _expand_arg("spacing", spacing, size=anchor.size)
    num_spacings = size / spacing
    fit_func = np.floor if shrink_to_fit else np.ceil
    return from_anchor_shape_spacing(
        shape=fit_func(num_spacings).astype(int) + 1,
        spacing=spacing,
        anchor=anchor,
        anchor_type=anchor_type,
    )


def from_bounds_shape(
    lower: Sequence[float],
    upper: Sequence[float],
    shape: Sequence[int] | int,
) -> Grid:
    """Create a `Grid` from its bounds and shape.

    Note that `lower` and `upper` must be 1D arrays of the same size,
    where the size is the number of dimensions of the grid.
    For example for 3D grids, arguments must be 1D arrays of length 3,
    corresponding to the x, y, and z axes, respectively.

    Parameters
    ----------
    lower
        Coordinates of the point with minimum index values in all dimensions.
    upper
        Coordinates of the point with maximum index values in all dimensions.
    shape
        Shape of the grid, i.e. number of points in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `lower` and `upper`.
    """
    lower = np.asarray(lower, dtype=np.float128)
    upper = np.asarray(upper, dtype=np.float128)
    shape = _expand_arg("shape", shape, size=lower.size)
    size = upper - lower
    return Grid(
        shape=shape,
        size=size,
        lower=lower,
        upper=upper,
        spacing=size / (shape - 1),
    )


def from_bounds_spacing(
    lower: Sequence[float],
    upper: Sequence[float],
    spacing: Sequence[float] | float,
    *,
    wiggle: Literal["bounds", "lower", "upper", "spacing"] = "bounds",
    shrink_to_fit: bool = False,
) -> Grid:
    """Create a `Grid` from its bounds and nominal spacings.

    Note that `lower` and `upper` must be 1D arrays of the same size,
    where the size is the number of dimensions of the grid.
    For example for 3D grids, arguments must be 1D arrays of length 3,
    corresponding to the x, y, and z axes, respectively.

    Parameters
    ----------
    lower
        Coordinates of the point with minimum values in all dimensions.
    upper
        Coordinates of the point with maximum values in all dimensions.
    spacing
        Spacing between grid points in each dimension.
        If a scalar is provided, it is expanded to a 1D array
        of the same size as the grid dimension.
        Otherwise, it must be a 1D array of the same size as `lower` and `upper`.
    wiggle
        Which quantities to adjust to satisfy the grid shape:
        - "bounds": fix `spacing`, stretch/contract both bounds equally.
        - "lower": fix `spacing` and `upper`, adjust `lower`.
        - "upper": fix `spacing` and `lower`, adjust `upper`.
        - "spacing": fix both bounds, adjust `spacing`.
    shrink_to_fit
        - `True`: ensure the grid fits inside the bounds
           by flooring the number of intervals.
        - `False`: ensure the grid covers the full range
           by ceiling the number of intervals.
    """
    lower = np.asarray(lower, dtype=np.float128)
    upper = np.asarray(upper, dtype=np.float128)
    spacing = _expand_arg("spacing", spacing, size=lower.size)
    size = upper - lower
    num_spacings = size / spacing
    fit_func = np.floor if shrink_to_fit else np.ceil
    shape = fit_func(num_spacings).astype(int) + 1
    if wiggle == "spacings":
        pass
    elif wiggle == "lower_bounds":
        lower = upper - spacing * (shape - 1)
    elif wiggle == "upper_bounds":
        upper = lower + spacing * (shape - 1)
    elif wiggle == "bounds":
        target_size = spacing * (shape - 1)
        delta = (target_size - size) / 2
        lower -= delta
        upper += delta
    else:
        raise ValueError(
            f"Invalid wiggle option '{wiggle}'. Must be one of: "
            f"'bounds', 'lower_bounds', 'upper_bounds', 'spacings'."
        )
    return from_bounds_shape(
        lower=lower,
        upper=upper,
        shape=shape,
    )


def from_data(
    *,
    shape: Sequence[int],
    size: Sequence[float],
    spacing: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> Grid:
    """Create a `Grid` from pre-computed data.

    Note that all arguments must be 1D arrays of the same size,
    where the size is the number of dimensions of the grid.
    For example for 3D grids, arguments must be 1D arrays of length 3,
    corresponding to the x, y, and z axes, respectively.

    Parameters
    ----------
    shape
        Shape of the grid, i.e. number of points in each dimension.
    size
        Physical length of the grid in each dimension.
    spacing
        Spacing between grid points in each dimension.
    lower
        Coordinates of the point with minimum values in all dimensions.
    upper
        Coordinates of the point with maximum values in all dimensions.
    """
    shape = np.asarray(shape, dtype=int)
    size = np.asarray(size, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    for bound, arg_name in zip(
        (lower, upper, shape),
        ("lower", "upper", "shape"),
        strict=True,
    ):
        if bound.ndim != 1:
            raise ValueError(
                f"Parameter `{arg_name}` expects a 1D array, "
                f"but input argument had {bound.ndim} dimensions. Input was: {bound}"
            )
    for bound, arg_name in zip(
        (lower, upper),
        ("lower", "upper"),
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
    for arg, arg_name in zip((upper, shape), ("upper", "shape"), strict=True):
        if lower.size != arg.size:
            raise ValueError(
                f"Parameters `lower` and `{arg_name}` expect 1D arrays of same size, "
                f"but input argument `lower` had size {lower.size}, "
                f"while `{arg_name}` had size {arg.size}. "
                f"Inputs were: `lower` = {lower}\n`{arg_name}` = {arg}."
            )
    size_is_invalid = size <= 0
    if np.any(size_is_invalid):
        raise ValueError(
            "All values in `lower` must be strictly smaller "
            f"than corresponding values in `upper`, but at indices {np.where(size_is_invalid)[0]} "
            f"`lower` had values {lower[size_is_invalid]} and `upper` had "
            f"values {upper[size_is_invalid]}."
        )
    shape_is_invalid = shape <= 0
    if np.any(shape_is_invalid):
        raise ValueError(
            "All values in `shape` must be positive integers, "
            f"but at indices {np.where(shape_is_invalid)[0]} "
            f"`shape` had values {shape[shape_is_invalid]}."
        )
    return Grid(
        shape=shape,
        size=size,
        spacing=spacing,
        lower=lower,
        upper=upper,
    )


def _expand_arg(
    arg_type: Literal["shape", "size", "spacing"],
    arg_val: Sequence[float | int] | float | int,
    size: int
) -> np.ndarray:
    """Expand a scalar or 1D array to a 1D array of the same size as the grid dimension.

    Parameters
    ----------
    arg_type
        Type of the argument, which determines how to interpret the input.
    arg_val
        A scalar or a sequence representing the value of the argument.
    size
        Number of dimensions of the grid, which determines the size of the output array.
    """
    dtype = np.int64 if arg_type == "shape" else np.float128
    arg_val = (
        np.array([arg_val] * size, dtype=dtype)
        if np.isscalar(arg_val) else
        np.asarray(arg_val, dtype=dtype)
    )
    if arg_val.ndim != 1:
        raise ValueError(
            f"Parameter `{arg_type}` must be a scalar or a 1D array, "
            f"but input argument had {arg_val.ndim} dimensions. Input was: {arg_val}"
        )
    if arg_val.size != size:
        raise ValueError(
            f"Parameter `{arg_type}` must have the same size as the grid dimension ({size}), "
            f"but input argument had size {arg_val.size}. Input was: {arg_val}"
        )
    if not np.issubdtype(arg_val.dtype, np.floating) and not np.issubdtype(arg_val.dtype, np.integer):
        raise ValueError(
            f"Parameter `{arg_type}` must be an array of real numbers, "
            f"but input argument had elements of type {arg_val.dtype}. Input was: {arg_val}"
        )
    if arg_type == "shape" and not jnp.issubdtype(arg_val.dtype, jnp.integer):
        raise ValueError(
            f"Parameter `shape` must be an integer or a 1D array of integers, "
            f"but input argument was a scalar of type {type(arg_val)}. Input was: {arg_val}"
        )
    return arg_val
