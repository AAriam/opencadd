import numpy as np
from scipy.spatial import KDTree, distance
from numpy.typing import ArrayLike


import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
import arrayer


def points_with_min_dist(
    points: ArrayLike,
    min_distance: float,
    p_norm: float = 2,
    max_points: int | None = None,
    batch_size_min: int | None = 50,
    batch_size_max: int = 2000,
    batch_size_grow_factor: float = 2.0,
) -> np.ndarray:
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
        Minimum required distance between any two returned points.
        A later point with a distance less than this to an earlier point will be rejected.
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
    2D array of shape `(n_selected_points, n_dimensions)`
    containing the coordinates of the selected points from `points`
    that are at least `min_distance` apart.

    Notes
    -----
    This implementation can process points in batches to bound memory usage,
    by first filtering each batch against the growing set of accepted points,
    and then applying a greedy within-batch distance filter.
    """
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {points.shape}")
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    if points.size == 0:
        return points

    # Calculate largest possible float (within machine precision) smaller than min_distance
    # to keep points at exactly min_distance (due to how KDTree works)
    r_eff = np.nextafter(min_distance, 0.0)

    accepted_points: list[np.ndarray] = []
    accepted_count = 0

    for batch in arrayer.tensor.make_batches(
        points,
        axis=0,
        min_size=batch_size_min,
        max_size=batch_size_max,
        grow_factor=batch_size_grow_factor,
    ):
        # Filter batch against already accepted points
        if accepted_points:
            # Query first nearest neighbor within min_distance
            dists, _ = KDTree(np.vstack(accepted_points)).query(
                batch,
                k=1,
                p=p_norm,
                distance_upper_bound=r_eff,
                workers=-1,
            )
            candidates = batch[np.isinf(dists)]
        else:
            candidates = batch

        if candidates.size == 0:
            continue

        # Greedy within-batch filtering
        neighbor_lists = KDTree(candidates).query_ball_point(
            candidates,
            r=r_eff,
            p=p_norm,
            workers=-1,
            return_sorted=True,
        )
        batch_rejected: set[int] = set()
        batch_accepted: list[np.ndarray] = []

        for idx, neighbors in enumerate(neighbor_lists):
            if idx in batch_rejected:
                # Already rejected this point
                continue
            # Accept this point
            batch_accepted.append(candidates[idx])
            # Remove self from neighbors
            neighbors = [i for i in neighbors if i != idx]
            # Reject all remaining neighbors
            batch_rejected.update(neighbors)
            # If we have reached the maximum number of points, stop
            accepted_count += 1
            if max_points is not None and accepted_count == max_points:
                break

        accepted_points.extend(batch_accepted)
        if max_points is not None and accepted_count == max_points:
            break

    if not accepted_points:
        return np.empty((0, points.shape[1]), dtype=points.dtype)

    result = np.vstack(accepted_points)
    if max_points is not None:
        result = result[:max_points]
    return result
