from typing import Sequence
import numpy as np
import jax
from scipy.spatial import KDTree
from numpy.typing import ArrayLike
import arrayer


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
    # Broadcast scalar radius to all spheres and validate
    radii = (
        np.full(n_spheres, float(radii)) if np.isscalar(radii) or (
            isinstance(radii, np.ndarray | jax.Array) and radii.ndim == 0
        ) else np.asarray(radii, dtype=float)
    )
    if radii.shape != (n_spheres,):
        raise ValueError(
            "`radii` array must have shape (n_spheres,) to match centers, "
            f"but got shape {radii.shape} for {n_spheres} spheres."
        )
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
        return _exclude_overlapping_spheres(
            centers=centers,
            radii=radii,
            p_norm=p_norm,
            exclude_tangents=exclude_tangents,
            max_out=max_out,
            kdtree_leafsize=kdtree_leafsize_single,
        )
    return _exclude_overlapping_spheres_batched(
        centers=centers,
        radii=radii,
        p_norm=p_norm,
        exclude_tangents=exclude_tangents,
        max_out=max_out,
        batch_size_min=batch_size_min,
        batch_size_max=batch_size_max,
        batch_size_grow_factor=batch_size_grow_factor,
        kdtree_leafsize=kdtree_leafsize_batch,
    )


def _exclude_overlapping_spheres(
    centers: np.ndarray,
    radii: np.ndarray,
    p_norm: float,
    exclude_tangents: bool,
    max_out: int,
    kdtree_leafsize: int = 40,
) -> list[int]:
    """Unbatched implementation of `exclude_overlapping_spheres`."""
    tree = KDTree(centers, leafsize=kdtree_leafsize)
    dist_comparison_function = np.less_equal if exclude_tangents else np.less
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
        max_excl_radius = radii[i] + radii.max()
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
        # Compute actual distances and mark overlaps
        neigh_idx = np.array(neighbors, dtype=int)
        diffs = centers[neigh_idx] - centers[i]  # shape (k, n_dims)
        dists = np.linalg.norm(diffs, ord=p_norm, axis=1)
        overlap_mask = dist_comparison_function(dists, (radii[i] + radii[neigh_idx]))
        removed[neigh_idx[overlap_mask]] = True
    return selected_indices


def _exclude_overlapping_spheres_batched(
    centers: np.ndarray,
    radii: np.ndarray,
    p_norm: float,
    exclude_tangents: bool,
    max_out: int,
    batch_size_min: int,
    batch_size_max: int,
    batch_size_grow_factor: float,
    kdtree_leafsize: int = 70,
) -> list[int]:
    """Batched implementation of `exclude_overlapping_spheres`."""
    if arrayer.tensor.pairwise_allclose(radii):
        # All radii are the same, so we can treat it as a scalar case
        scalar_case = True
        radii = radii[0]
        r_sum = radii * 2
        # If tangents should be included,
        # calculate largest possible float (within machine precision)
        # smaller than r_sum to keep points at exactly r_sum (due to how KDTree works)
        r_eff = r_sum if exclude_tangents else np.nextafter(r_sum, 0.0)
    else:
        scalar_case = False
        distance_comparison_function = np.less_equal if exclude_tangents else np.less

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
            if scalar_case:
                # Query first nearest neighbor within r_sum
                dists, _ = KDTree(np.vstack(accepted_points), leafsize=kdtree_leafsize).query(
                    point_batch,
                    k=1,
                    p=p_norm,
                    distance_upper_bound=r_eff,
                    workers=-1,
                )
                mask = np.isinf(dists)
            else:
                # KDTree.query expects a single scalar radius
                # and doesn’t support arrays for distance_upper_bound.
                # We compute worst-case search radius = max_i,j (r_i + r_j)
                max_distance = (
                    radii[accepted_indices][:,None] + radii[index_batch][None,:]
                ).max()
                pairs = KDTree(np.vstack(accepted_points), leafsize=kdtree_leafsize).sparse_distance_matrix(
                    other=KDTree(point_batch, leafsize=kdtree_leafsize),
                    max_distance=max_distance,
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
        neighbor_lists = KDTree(candidates, leafsize=kdtree_leafsize).query_ball_point(
            candidates,
            r=r_eff if scalar_case else (
                radii[candidate_indices][:,None] + radii[candidate_indices][None,:]
            ).max(),
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
            # If we have reached the maximum number of points, stop
            accepted_count += 1
            if accepted_count == max_out:
                break
            # Reject any neighbor j where dist <= r_i + r_j (with tangent logic)
            if scalar_case:
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

        accepted_points.extend(batch_accepted)
        accepted_indices.extend(batch_accepted_indices)
        if accepted_count == max_out:
            break
    return accepted_indices
