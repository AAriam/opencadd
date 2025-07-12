from typing import Sequence
import numpy as np
from scipy.spatial import KDTree
from numpy.typing import ArrayLike
import arrayer


def points_with_min_dist(
    points: ArrayLike,
    min_distance: float | Sequence[float],
    p_norm: float = 2,
    max_points: int | None = None,
    batch_size_min: int | None = 50,
    batch_size_max: int = 2000,
    batch_size_grow_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a subset of points where a minimum distance is guaranteed.

    This function selects a subset of the input `points`
    such that no two points are closer than a given `min_distance`
    under a given Minkowski `p_norm`.

    Note that the first point in the input always wins over later neighbors
    that are closer than `min_distance`.
    If you want to pick the spatially "best" points,
    you need a sorting step before this greedy sweep,
    so that points are ordered by decreasing "importance".

    Parameters
    ----------
    points
        2D array of shape `(n_points, n_dimensions)`,
        containing the coordinates of `n_points` points in `n_dimensions` space.
    min_distance
        The "blocking radius" for each point.
        For each selected point, all other remaining points
        within this distance are rejected.
        If a scalar is provided, it applies to all points.
        If a sequence is provided, it must have the same length as `points`
        and specifies the minimum distance for each point.
    p_norm
        Minkowski p-norm (in the range `[1, inf]`) to use for distances
        (e.g., 2 for Euclidean, np.inf for Chebyshev).
        Note that a finite large `p` may cause a `ValueError`
        if overflow can occur.
    max_points
        Maximum number of points to return.
        The function will stop adding points once this limit is reached.
        If `None` (default), all points that satisfy the distance condition are returned.
    batch_size_min
        Minimum batch size.
        If set to `None`, no batching is applied,
        otherwise the input `points` are processed in batches
        to save memory and avoid large allocations.
        This is useful for large datasets where memory usage needs to be controlled.
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
    selected_points
        2D array of shape `(n_selected_points, n_dimensions)`
        containing the coordinates of the selected points from `points`
        that are at least `min_distance` apart.
    selected_indices
        1D array of shape `(n_selected_points,)` containing the indices
        of the selected points in the original `points` array.

    Notes
    -----
    This implementation can process points in batches to bound memory usage,
    by first filtering each batch against the growing set of accepted points,
    and then applying a greedy within-batch distance filter.
    """
    points = np.asarray(points)
    n_points = points.shape[0]
    if points.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {points.shape}")
    if np.isscalar(min_distance):
        scalar_case = True
        if min_distance <= 0:
            raise ValueError("min_distance must be positive")
        # Calculate largest possible float (within machine precision) smaller than min_distance
        # to keep points at exactly min_distance (due to how KDTree works)
        r_eff = np.nextafter(float(min_distance), 0.0)
    else:
        radii = np.asarray(min_distance, float)
        if radii.ndim != 1 or radii.shape[0] != n_points:
            raise ValueError(
                f"min_distance array must have shape ({n_points},), got {radii.shape}"
            )
        if np.any(radii <= 0):
            raise ValueError("all min_distance values must be positive")
        if np.any(np.isnan(radii)):
            raise ValueError("min_distance cannot contain NaN values")
        if np.any(np.isinf(radii)):
            raise ValueError("min_distance cannot contain infinite values")
        if arrayer.tensor.pairwise_allclose(radii):
            # All radii are the same, so we can treat it as a scalar case
            scalar_case = True
            r_eff = np.nextafter(radii[0], 0.0)
        else:
            # We have a different radius for each point
            # Use the next float smaller than each radius
            # to ensure we can still select points at exactly min_distance
            scalar_case = False
            r_eff = np.nextafter(radii, 0.0)
    if n_points == 0:
        return points, np.empty((0,), dtype=int)

    accepted_points: list[np.ndarray] = []
    accepted_indices: list[int] = []
    accepted_count = 0

    point_batches = arrayer.tensor.make_batches(
        points,
        axis=0,
        min_size=batch_size_min,
        max_size=batch_size_max,
        grow_factor=batch_size_grow_factor,
    )
    index_batches = arrayer.tensor.make_batches(
        np.arange(n_points),
        axis=0,
        min_size=batch_size_min,
        max_size=batch_size_max,
        grow_factor=batch_size_grow_factor,
    )

    for point_batch, index_batch in zip(point_batches, index_batches):
        # Filter batch against already accepted points
        if accepted_points:
            if scalar_case:
                # Query first nearest neighbor within min_distance
                dists, _ = KDTree(np.vstack(accepted_points)).query(
                    point_batch,
                    k=1,
                    p=p_norm,
                    distance_upper_bound=r_eff,
                    workers=-1,
                )
                mask = np.isinf(dists)
            else:
                # KDTree.query expects a single scalar radius
                # and doesn’t support arrays for distance_upper_bound
                pairs = KDTree(np.vstack(accepted_points)).sparse_distance_matrix(
                    other=KDTree(point_batch),
                    max_distance=r_eff[np.array(accepted_indices)].max(),
                    p=p_norm
                )
                mask = np.ones(len(point_batch), bool)
                for (i, j), d in pairs.items():
                    if d < r_eff[accepted_indices[i]]:
                        mask[j] = False
            candidates = point_batch[mask]
            candidate_indices = index_batch[mask]
        else:
            candidates = point_batch
            candidate_indices = index_batch

        if candidates.size == 0:
            continue

        # Greedy within-batch filtering
        neighbor_lists = KDTree(candidates).query_ball_point(
            candidates,
            r=r_eff if scalar_case else r_eff[candidate_indices],
            p=p_norm,
            workers=-1,
            return_sorted=True,
        )
        batch_rejected: set[int] = set()
        batch_accepted: list[np.ndarray] = []
        batch_accepted_indices: list[int] = []

        for idx, neighbors in enumerate(neighbor_lists):
            if idx in batch_rejected:
                # Already rejected this point
                continue
            # Accept this point
            batch_accepted.append(candidates[idx])
            batch_accepted_indices.append(int(candidate_indices[idx]))
            # Remove self from neighbors
            neighbors = [i for i in neighbors if i != idx]
            # Reject all remaining neighbors
            batch_rejected.update(neighbors)
            # If we have reached the maximum number of points, stop
            accepted_count += 1
            if max_points is not None and accepted_count == max_points:
                break

        accepted_points.extend(batch_accepted)
        accepted_indices.extend(batch_accepted_indices)
        if max_points is not None and accepted_count == max_points:
            break

    if not accepted_points:
        return np.empty((0, points.shape[1]), dtype=points.dtype), np.empty((0,), dtype=int)

    final_points = np.vstack(accepted_points)
    final_indices = np.array(accepted_indices, dtype=int)
    if max_points is not None:
        final_points = final_points[:max_points]
        final_indices = final_indices[:max_points]
    return final_points, final_indices
