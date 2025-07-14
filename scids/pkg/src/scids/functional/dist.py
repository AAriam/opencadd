from typing import Sequence, Callable
import numpy as np
import jax
from scipy.spatial import KDTree
from numpy.typing import ArrayLike
import arrayer

from numba import njit
from numba.typed import List


def exclude_overlapping_spheres(
    centers: ArrayLike,
    radii: float | Sequence[float],
    *,
    p_norm: float = 2,
    exclude_tangents: bool = True,
    max_out: int | None = None,
    kdtree_leafsize_single: int = 40,
    kdtree_leafsize_batch: int = 70,
    batch_threshold: int = 20_000,
    batch_size_min: int = 50,
    batch_size_max: int = 2_000,
    batch_size_grow_factor: float = 2.0,
) -> list[int]:
    """Select a subset of spheres where no two spheres overlap.

    This function selects a subset of the input spheres
    defined by their `centers` and `radii`,
    such that the distance (under a given Minkowski `p_norm`)
    between the centers of any two spheres
    is larger than (or at least equal to; see `exclude_tangents`)
    the sum of their radii.

    This function can also be used to select a subset of points
    where no two points are within a certain distance.
    For this, you can simply provide the point coordinates as `centers`
    and half the minimum distance as a scalar `radii`.

    Note that the first sphere in the input
    always wins over later overlapping neighbors.
    If you want to pick the spatially "best" spheres,
    you need a sorting step before this greedy sweep,
    so that spheres are ordered by decreasing "importance".

    Parameters
    ----------
    centers
        2D array of shape `(n_spheres, n_dimensions)`,
        containing the center coordinates of
        `n_spheres` spheres in `n_dimensions`-dimensional space.
    radii
        Radius of the spheres.
        If a scalar is provided, it applies to all spheres.
        If a sequence is provided, it must have length `n_spheres`
        and specifies the radius of each sphere.
    p_norm
        Minkowski p-norm (in the range `[1, inf]`)
        to use for distance calculations
        (e.g., 2 for Euclidean, np.inf for Chebyshev).
        Note that a finite large `p` may cause a `ValueError`
        if overflow can occur.
    exclude_tangents
        Whether tanget spheres
        (i.e., spheres whose center distance
        is exactly equal to the sum of their radii)
        are considered overlapping.
    max_out
        Maximum number of spheres to return.
        If provided, the function will stop once this limit is reached.
        If `None` (default), all non-overlapping spheres are returned.
    kdtree_leafsize
        Leaf size for the KDTree used to find neighbors.
        A smaller value may lead to more memory usage,
        while a larger value may lead to slower queries.
        The default is 40, which is a good compromise
        between speed and memory usage for most cases.
    batch_threshold
        Minimum number of spheres at which the algorithm
        switches to batch processing to save memory and avoid large allocations.
        Benchmarking has shown that for more than 20,000 spheres,
        batching is faster than processing all spheres at once.
    batch_size_min
        Minimum batch size.
    batch_size_max
        Maximum batch size; batches will not exceed this many points.
        The default is 2000, which was determined by profiling
        with a large number of points.
        Too small values result in too many Python loops,
        which can slow down the process,
        while too large values can lead to large memory allocations,
        which may also slow down or crash the process
        (see issue [#6010](https://github.com/scikit-image/scikit-image/issues/6010)
        and pull request [#6035](https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691)
        for more details and benchmark results).
    batch_size_grow_factor
        Factor by which the batch size grows.
        The first batch will be of size `batch_size_min`,
        and subsequent batches will grow by this factor
        until they reach `batch_size_max`.

    Returns
    -------
    List of integers containing the indices
    of the selected spheres in the input `centers` array.

    Raises
    ------
    ValueError
    - If `centers` is not a 2D array.
    - If `radii` is not a scalar or a 1D array of length equal to the number of spheres.
    - If any radius in `radii` is non-positive, NaN, or infinite.

    Notes
    -----
    This implementation can process points in batches to bound memory usage,
    by first filtering each batch against the growing set of accepted points,
    and then applying a greedy within-batch distance filter.
    """
    centers = np.asarray(centers)
    n_spheres = centers.shape[0]
    if centers.ndim != 2:
        raise ValueError(
            "`centers` must be a 2D array of shape (n_spheres, n_dims), "
            f"but got shape {centers.shape}."
        )
    if np.isscalar(radii) or (
        isinstance(radii, np.ndarray | jax.Array) and radii.ndim == 0
    ):
        radii = float(radii)
        max_distance = radii * 2
        distance_comparison_function = None
        if not exclude_tangents:
            # If tangents should be included,
            # calculate largest possible float (within machine precision)
            # smaller than max_distance so that the KDTree.query_ball_point
            # doesn't return points at exactly max_distance.
            # This way we can avoid the subsequent distance check altogether.
            max_distance = np.nextafter(max_distance, 0.0)
    else:
        radii = np.asarray(radii)
        if radii.shape != (n_spheres,):
            raise ValueError(
                "`radii` array must have shape (n_spheres,) to match centers, "
                f"but got shape {radii.shape} for {n_spheres} spheres."
            )
        if not np.issubdtype(radii.dtype, np.floating) and not np.issubdtype(radii.dtype, np.integer):
            raise TypeError(
                "`radii` must be a floating-point or integer array, "
                f"but got dtype {radii.dtype}."
            )
        if arrayer.tensor.pairwise_allclose(radii):
            # All radii are the same, so we can treat it as a scalar case
            radii = float(radii[0])
            max_distance = radii * 2
            distance_comparison_function = None
            if not exclude_tangents:
                max_distance = np.nextafter(max_distance, 0.0)
        else:
            max_distance = None
            distance_comparison_function = np.less_equal if exclude_tangents else np.less
    if np.any(radii <= 0):
        raise ValueError(
            f"`radii` must be positive, but got {radii}."
        )
    if np.any(np.isnan(radii)):
        raise ValueError("`radii` cannot contain NaN values.")
    if np.any(np.isinf(radii)):
        raise ValueError("`radii` cannot contain infinite values.")

    max_out = np.inf if max_out is None else int(max_out)
    if n_spheres == 0 or max_out == 0:
        return []

    if n_spheres <= batch_threshold:
        return _exclude_overlapping_spheres_single(
            centers=centers,
            radii=radii,
            max_distance=max_distance,
            distance_comparison_function=distance_comparison_function,
            p_norm=p_norm,
            max_out=max_out,
            kdtree_leafsize=kdtree_leafsize_single,
        )
    return _exclude_overlapping_spheres_batched(
        centers=centers,
        radii=radii,
        max_distance=max_distance,
        distance_comparison_function=distance_comparison_function,
        p_norm=p_norm,
        max_out=max_out,
        batch_size_min=batch_size_min,
        batch_size_max=batch_size_max,
        batch_size_grow_factor=batch_size_grow_factor,
        kdtree_leafsize=kdtree_leafsize_batch,
    )


def _exclude_overlapping_spheres_single(
    centers: np.ndarray,
    radii: np.ndarray | float,
    max_distance: float | None,
    distance_comparison_function: Callable | None,
    p_norm: float,
    max_out: int,
    kdtree_leafsize: int = 40,
) -> list[int]:
    """Unbatched implementation of `exclude_overlapping_spheres`."""
    if not max_distance:
        max_radius = radii.max()
    tree = KDTree(centers, leafsize=kdtree_leafsize)
    n_spheres = centers.shape[0]
    removed = np.zeros(n_spheres, dtype=bool)
    selected_indices = []
    for i in range(n_spheres):
        # Skip if this sphere has already been removed
        if removed[i]:
            continue
        # Keep this sphere
        selected_indices.append(i)
        # Stop if we have reached the maximum number of spheres
        if len(selected_indices) == max_out:
            break
        # Maximum possible exclusion radius for sphere i against any later sphere
        max_excl_radius = max_distance or (radii[i] + max_radius)
        # Find all neighbors within that maximum radius
        neighbors = tree.query_ball_point(
            centers[i],
            r=max_excl_radius,
            p=p_norm,
            workers=-1,
        )
        # Filter neighbors strictly less important (higher index)
        neighbors = [j for j in neighbors if j > i]
        if not neighbors:
            continue
        if max_distance:
            # If we are in the scalar case,
            # we can directly add the neighbors to the removed set.
            removed[neighbors] = True
            continue
        # Compute actual distances and mark overlaps
        neigh_idx = np.array(neighbors, dtype=int)
        diffs = centers[neigh_idx] - centers[i]  # shape (k, n_dims)
        dists = np.linalg.norm(diffs, ord=p_norm, axis=1)
        overlap_mask = distance_comparison_function(
            dists,
            max_distance or (radii[i] + radii[neigh_idx])
        )
        removed[neigh_idx[overlap_mask]] = True
    return selected_indices


def _exclude_overlapping_spheres_batched(
    centers: np.ndarray,
    radii: np.ndarray | float,
    max_distance: float | None,
    distance_comparison_function: Callable | None,
    p_norm: float,
    max_out: int,
    batch_size_min: int,
    batch_size_max: int,
    batch_size_grow_factor: float,
    kdtree_leafsize: int = 70,
) -> list[int]:
    """Batched implementation of `exclude_overlapping_spheres`."""
    accepted_points: list[np.ndarray] = []
    accepted_indices: list[int] = []
    accepted_count = 0

    point_batches = arrayer.tensor.make_batches(
        centers,
        axis=0,
        min_size=batch_size_min,
        max_size=batch_size_max,
        grow_factor=batch_size_grow_factor,
    )
    index_batches = arrayer.tensor.make_batches(
        np.arange(centers.shape[0]),
        axis=0,
        min_size=batch_size_min,
        max_size=batch_size_max,
        grow_factor=batch_size_grow_factor,
    )

    for point_batch, index_batch in zip(point_batches, index_batches):
        # Filter batch against already accepted points
        if accepted_points:
            if max_distance:
                # Query first nearest neighbor within max_distance
                dists, _ = KDTree(
                    np.vstack(accepted_points),
                    leafsize=kdtree_leafsize
                ).query(
                    point_batch,
                    k=1,
                    p=p_norm,
                    distance_upper_bound=max_distance,
                    workers=-1,
                )
                mask = np.isinf(dists)
            else:
                # KDTree.query expects a single scalar radius
                # and doesn’t support arrays for distance_upper_bound.
                # We compute worst-case search radius = max_i,j (r_i + r_j)
                curr_max_distance = (
                    radii[accepted_indices][:,None] + radii[index_batch][None,:]
                ).max()
                pairs = KDTree(
                    np.vstack(accepted_points),
                    leafsize=kdtree_leafsize
                ).sparse_distance_matrix(
                    other=KDTree(point_batch, leafsize=kdtree_leafsize),
                    max_distance=curr_max_distance,
                    p=p_norm
                )
                mask = np.ones(len(point_batch), bool)
                for (i, j), d in pairs.items():
                    thresh = radii[accepted_indices[i]] + radii[index_batch[j]]
                    if distance_comparison_function(d, thresh):
                        mask[j] = False
            candidates = point_batch[mask]
            candidate_indices = index_batch[mask]
        else:
            candidates = point_batch
            candidate_indices = index_batch
        if candidates.size == 0:
            continue

        # Greedy within-batch filtering.
        # First, rough filter up to the maximum possible sum-of-radii.
        neighbor_lists = KDTree(
            candidates,
            leafsize=kdtree_leafsize
        ).query_ball_point(
            candidates,
            r=max_distance if max_distance else (
                radii[candidate_indices][:,None] + radii[candidate_indices][None,:]
            ).max(),
            p=p_norm,
            workers=-1,
            return_sorted=True,
        )
        batch_rejected: set[int] = set()
        for idx, neighbors in enumerate(neighbor_lists):
            if idx in batch_rejected:
                # Already rejected this point
                continue
            # Accept this point
            accepted_points.append(candidates[idx])
            accepted_indices.append(int(candidate_indices[idx]))
            # If we have reached the maximum number of points, stop
            accepted_count += 1
            if accepted_count == max_out:
                break
            # Reject any neighbor j where dist <= r_i + r_j (with tangent logic)
            if max_distance:
                # Remove self from neighbors
                neighbors = [i for i in neighbors if i != idx]
                # Reject all remaining neighbors
                batch_rejected.update(neighbors)
            else:
                for j in neighbors:
                    if j == idx:
                        continue
                    d = np.linalg.norm(candidates[idx] - candidates[j], ord=p_norm)
                    thresh = radii[candidate_indices[idx]] + radii[candidate_indices[j]]
                    if distance_comparison_function(d, thresh):
                        batch_rejected.add(j)
        if accepted_count == max_out:
            break
    return accepted_indices


def ensure_pointcloud_spacing(
    points: ArrayLike,
    point_types: Sequence[int] | None,
    min_spacing: float | ArrayLike,
    include_exact: bool = True,
    max_count: int | Sequence[int] | None = None,
    p_norm: float = 2,
) -> list[int]:
    """Ensure spacing between points of different types using a greedy selection.

    This function selects the largest possible subset
    of points from the input `points` array,
    such that the Minkowski distance of order `p_norm`
    between any two points `i` and `j` of types `t_i` and `t_j`
    is at least `spacing[t_i, t_j]`.
    The points are selected in the order they appear in the input,
    so that earlier points are preferred over later points
    when it comes to satisfying the spacing constraints.
    At any time, if the number of points of a certain type
    reaches the maximum allowed count `max_count[t_i]`,
    no more points of that type will be selected.

    Parameters
    ----------
    points
        2D array of shape `(n_points, n_dimensions)`,
        containing the coordinates of
        `n_points` points in `n_dimensions`-dimensional space.
    point_types
        Type of each point.
        If `None`, all points are considered to be of the same type.
        Otherwise, this must be a 1D integer array of shape `(n_points,)`,
        containing the types of each point.
        Types should be integers in the range `[0, n_types - 1]`,
        where each unique integer represents a different point type.
    spacing
        Minimum spacing between points of different types.
        If `point_types` is `None`, this must be a scalar value
        specifying the minimum distance between any two points.
        Otherwise, this must be a square (optionally upper triangular)
        matrix of shape `(n_types, n_types)`,
        where `spacing[i, j]` specifies the minimum distance
        between points of type `i` and `j`.
        The matrix must be symmetric, i.e., `spacing[i, j] == spacing[j, i]`.

        2D array (or upper triangular matrix) of shape `(n_types, n_types)`,
        containing the minimum spacing between points of different types.
        The element at `(i, j)` (i <= j) specifies the minimum distance
        between points of type `i` and type `j`.

        Array of shape `(n_types, n_types)` with upper-triangular values
        specifying minimum Minkowski distances between types.

    max_count
        1D integer array of shape `(n_types,)`,
        containing the maximum number of points allowed for each type.

        Sequence of length `n_types` giving maximum number of points per type.
    p_norm
        Order of the Minkowski distance (p >= 1).

    Returns
    -------
    List of indices of selected points that satisfy the spacing constraints.
    The returned indices correspond to the input `points` array.
    """
    points = np.asarray(points)
    if not np.issubdtype(points.dtype, np.floating) and not np.issubdtype(points.dtype, np.integer):
        raise TypeError(
            "`points` must be a floating-point or integer array, "
            f"but got dtype {points.dtype}."
        )
    if points.ndim != 2:
        raise ValueError(
            "`points` must be a 2D array of shape (n_points, n_dims), "
            f"but got shape {points.shape}."
        )
    n_points = points.shape[0]
    if point_types is None:
        point_types = np.zeros(n_points, dtype=int)
        min_required_types = 1
        single_type = True
    else:
        point_types = np.asarray(point_types)
        if not np.issubdtype(point_types.dtype, np.integer):
            raise TypeError(
                "`point_types` must be an integer array, "
                f"but got dtype {point_types.dtype}."
            )
        if point_types.ndim != 1 or point_types.shape[0] != n_points:
            raise ValueError(
                "`point_types` must be a 1D array of shape (n_points,), "
                f"but got shape {point_types.shape}."
            )
        if np.any(point_types < 0):
            raise ValueError(
                "`point_types` must contain non-negative integers, "
                f"but got {point_types}."
            )
        min_required_types = point_types.max() + 1
        single_type = min_required_types == 1

    if np.isscalar(min_spacing) or (
        isinstance(min_spacing, np.ndarray | jax.Array) and min_spacing.ndim == 0
    ):
        if min_required_types > 1:
            raise ValueError(
                "`min_spacing` must be a 2D array of shape (n_types, n_types) "
                "when `point_types` is not None, "
                f"but got a scalar {min_spacing} for {min_required_types} types."
            )
        min_spacing = np.full((1, 1), float(min_spacing))
    else:
        min_spacing = np.asarray(min_spacing)
        if min_spacing.ndim != 2:
            raise ValueError(
                "`min_spacing` must be a 2D array of shape (n_types, n_types), "
                f"but got shape {min_spacing.shape}."
            )
        if min_spacing.shape[0] != min_spacing.shape[1]:
            raise ValueError(
                "`min_spacing` must be a square matrix, "
                f"but got shape {min_spacing.shape}."
            )
        if min_spacing.shape[0] < min_required_types:
            raise ValueError(
                "`min_spacing` must have at least `n_types` rows and columns, "
                f"but got shape {min_spacing.shape} for {min_required_types} types."
            )
        if single_type and min_spacing.shape[0] > 1:
            raise ValueError(
                "`min_spacing` must have at most `n_types` rows and columns, "
                f"but got shape {min_spacing.shape} for a single type."
            )
    if not np.all(np.isfinite(min_spacing)):
        raise ValueError(
            "`min_spacing` must contain finite values, "
            f"but got {min_spacing}."
        )
    if not np.all(np.isreal(min_spacing)):
        raise ValueError(
            "`min_spacing` must contain real values, "
            f"but got {min_spacing}."
        )
    if np.any(min_spacing < 0):
        raise ValueError(
            "`min_spacing` must contain non-negative values, "
            f"but got {min_spacing}."
        )
    if max_count is not None:
        max_count = np.asarray(max_count)
        if not np.issubdtype(max_count.dtype, np.integer):
            raise TypeError(
                "`max_count` must be an integer array, "
                f"but got dtype {max_count.dtype}."
            )
        if max_count.ndim != 1:
            raise ValueError(
                "`max_count` must be a 1D array, "
                f"but got shape {max_count.shape}."
            )
        if max_count.shape[0] < min_required_types:
            raise ValueError(
                "`max_count` must have at least `n_types` elements, "
                f"but got shape {max_count.shape} for {min_required_types} types."
            )
    # print("points:", points)
    # print("point_types:", point_types)
    # print("min_spacing:", min_spacing)
    # print("max_count:", max_count)
    # print("p_norm:", p_norm)

    return _ensure_pointcloud_spacing(
        points=points,
        point_types=point_types,
        spacing=min_spacing,
        max_count=max_count,
        include_exact=include_exact,
        p_norm=float(p_norm)
    )


@njit
def _ensure_pointcloud_spacing(
    points: np.ndarray,
    point_types: np.ndarray,
    spacing: np.ndarray,
    include_exact: bool,
    max_count: np.ndarray | None,
    p_norm: float,
) -> List:
    # Dimensions
    n_points, n_dim = points.shape
    n_types = spacing.shape[0]

    # Precompute spacing^p_norm for threshold comparisons
    spacing_p = np.empty((n_types, n_types), dtype=np.float64)
    for i in range(n_types):
        for j in range(i, n_types):
            spacing_p[i, j] = spacing_p[j, i] = spacing[i, j] ** p_norm
    # print("spacing_p:", spacing_p)

    # Initialize counts and selected list
    counts = np.zeros(n_types, dtype=np.int64)
    selected_indices = List()

    # Greedy selection
    for point_idx in range(n_points):
        point_type = point_types[point_idx]
        # Skip if this type is disallowed or reaches max
        if max_count is not None and counts[point_type] >= max_count[point_type]:
            continue
        point_coords = points[point_idx]
        ok = True
        # Check against all previously selected
        for selected_idx in selected_indices:
            selected_type = point_types[selected_idx]
            # threshold^p for this pair of types
            thresh_p = spacing_p[point_type, selected_type]
            # compute Minkowski distance^p
            d_p = 0.0
            for d in range(n_dim):
                d_p += abs(point_coords[d] - points[selected_idx, d]) ** p_norm
            if d_p < thresh_p:
                ok = False
                break
        if ok:
            counts[point_type] += 1
            selected_indices.append(point_idx)
    return selected_indices
