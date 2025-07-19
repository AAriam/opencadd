from typing import Sequence, Any
import math

import numpy as np


def create_job_inputs(
    feature_radius: dict[str, float],
    filter: dict[str, Any] | None = None,
    largest_peaks: dict[str, Any] | None = None,
    cnn: dict[str, Any] | None = None,
) -> list[dict[str, str | dict[str, Any]]]:
    if cnn is None and largest_peaks is None:
        raise ValueError("At least one of `cnn` or `largest_peaks` must be provided.")
    filters = _filter(
        radius_base=feature_radius,
        **(filter or {})
    )
    out = []
    if largest_peaks is not None:
        out.extend(_largest_peaks(filters=filters, **largest_peaks))
    if cnn is not None:
        out.extend(_cnn(filters=filters, feature_radius=feature_radius, **cnn))
    return out


def _cnn(
    filters: Sequence[dict[str, dict[str, Any]]],
    feature_radius: dict[str, float],
    grid_spacing: float,
    grid_unique_distances: np.ndarray,
    threshold_percentiles: Sequence[float],
    max_members_factors: Sequence[float],
    min_members_fraction: float,
    min_members_count: int,
) -> list[dict[str, str | dict[str, Any]]]:
    max_members = _cnn_max_members(feature_radius, grid_spacing)
    out = []
    max_distance, min_neighbors = _cnn_clustering_params(
        feature_radius=feature_radius,
        grid_spacing=grid_spacing,
        grid_unique_distances=grid_unique_distances,
    )
    for filter_ in filters:
        for threshold_percentile in threshold_percentiles:
            for max_member_factor in max_members_factors:
                max_members_scaled = {
                    feature_type: int(np.ceil(max_member_factor * max_member))
                    for feature_type, max_member in max_members.items()
                }
                min_members_all = {
                    feature_type: np.floor(
                        np.linspace(start=1, stop=max_members * min_members_fraction, num=min_members_count)
                    ).astype(int).tolist()
                    for feature_type, max_members in max_members_scaled.items()
                }
                for min_members_idx in range(min_members_count):
                    min_members = {
                        feature_type: min_members_all[feature_type][min_members_idx]
                        for feature_type in min_members_all
                    }
                    for best_per_point in (True, False):
                        out.append(
                            {
                                "method": "cnn",
                                "identifier": filter_["identifier"] | {
                                    "min_members_idx": min_members_idx,
                                    "max_members_factor": max_member_factor,
                                    "best_per_point": best_per_point,
                                    "threshold_percentile": threshold_percentile,
                                },
                                "kwargs": filter_["kwargs"] | {
                                    "max_distance": max_distance,
                                    "min_neighbors": min_neighbors,
                                    "min_members": min_members,
                                    "max_members": max_members_scaled,
                                    "center_type": "midpoint",
                                    "radius_type": "max",
                                    "peak_type": "min",
                                    "best_per_point": best_per_point,
                                    "threshold_value": 0,
                                    "threshold_percentile": threshold_percentile,
                                    "threshold_include_equal": False,
                                }
                            }
                        )
    return out


def _largest_peaks(
    filters: Sequence[dict[str, dict[str, Any]]],
    min_distance: dict[tuple[str, str], float],
    min_distance_factors: Sequence[float],
    max_features: dict[str, int],
    max_features_factors: Sequence[int],
    priority_factors: Sequence[dict[str, float]],
) -> list[dict[str, str | dict[str, Any]]]:
    out = []
    for filter_ in filters:
        for min_distance_factor in min_distance_factors:
            min_distance_ = {k: v * min_distance_factor for k, v in min_distance.items()}
            for max_feature_factor in max_features_factors:
                max_features_ = {k: v * max_feature_factor for k, v in max_features.items()}
                for priority_factor_idx, priority_factor in enumerate(priority_factors):
                    out.append(
                        {
                            "method": "largest_peaks",
                            "identifier": filter_["identifier"] | {
                                "min_distance_factor": min_distance_factor,
                                "max_feature_factor": max_feature_factor,
                                "priority_factor_idx": priority_factor_idx,
                            },
                            "kwargs": filter_["kwargs"] | {
                                "min_distance": min_distance_,
                                "max_features": max_features_,
                                "priority_factor": priority_factor,
                            }
                        }
                    )
    return out


def _filter(
    radius_base: dict[str, float],
    radius_factors: Sequence[float] | None = None,
    percentiles: Sequence[float] | None = None,
    sigma_factors: Sequence[float] | None = None,
) -> list[dict[str, dict[str, Any]]]:
    """Generate a list of inputs for different filter configurations.

    The created filter configurations include:
    - One "none" filter (no filtering).
    - One mean filter for each radius factor.
    - One percentile filter for each radius factor/percentile combination.
    - One Gaussian filter for each radius factor/sigma factor combination.

    Parameters
    ----------
    radius_base
        Base radii for different feature types.
    radius_factors
        Factors to scale the base radii to create different filter radii.
        If not provided, only the "none" filter will be created.
    percentiles
        Percentiles to use for percentile filters.
        If not provided, no percentile filters will be created.
    sigma_factors
        Factors to scale the base radii to create different Gaussian sigma values.
        If not provided, no Gaussian filters will be created.

    Returns
    -------
    List of dictionaries with identifiers
    and keyword arguments for each filter configuration.
    """
    out = [
        {
            "identifier": {"filter_function": "none"},
            "kwargs": {"filter_function": None},
        }
    ]
    for filter_radius_factor in (radius_factors or []):
        filter_radius = {k: v * filter_radius_factor for k, v in radius_base.items()}
        out.append(
            {
                "identifier": {
                    "filter_function": "mean",
                    "filter_radius_factor": filter_radius_factor
                },
                "kwargs": {
                    "filter_function": "mean",
                    "filter_radius": filter_radius
                }
            }
        )
        for percentile in (percentiles or []):
            out.append(
                {
                    "identifier": {
                        "filter_function": "percentile",
                        "filter_radius_factor": filter_radius_factor,
                        "filter_percentile": percentile
                    },
                    "kwargs": {
                        "filter_function": "percentile",
                        "filter_radius": filter_radius,
                        "filter_percentile": percentile
                    }
                }
            )
        for sigma_factor in (sigma_factors or []):
            sigma = {k: v * sigma_factor for k, v in filter_radius.items()}
            out.append(
                {
                    "identifier": {
                        "filter_function": "gaussian",
                        "filter_radius_factor": filter_radius_factor,
                        "filter_sigma_factor": sigma_factor
                    },
                    "kwargs": {
                        "filter_function": "gaussian",
                        "filter_radius": filter_radius,
                        "filter_gaussian_sigma": sigma
                    }
                }
            )
    return out


def _cnn_clustering_params(
    feature_radius: dict[str, float],
    grid_spacing: float,
    grid_unique_distances: np.ndarray,
    min_neighbors_list_length: int = 5,
) -> dict[str, list[float]]:
    """Calculate `max_distance` and `min_neighbors` parameters for CNN clustering.

    Parameters
    ----------
    feature_radius
        Dictionary mapping feature types to their radii.
    grid_spacing
        Spacing of the grid.
    grid_unique_distances
        Unique distances between grid points in the grid.
    min_neighbors_list_length
        Length of the list of minimum neighbors to generate for each `max_distance`.

    Returns
    -------
    max_distances
        Dictionary mapping feature types to lists of maximum distances.
    min_neighbors
        Dictionary mapping feature types to lists of minimum neighbors.
    """
    max_distances = {}
    min_neighbors = {}
    for feature_type, radius in feature_radius.items():
        feature_max_distances = np.sort(grid_unique_distances[grid_unique_distances <= radius])[::-1].tolist()
        for feature_max_distance in feature_max_distances:
            max_common_neighbors = _cnn_max_common_neighbors(
                distance=feature_max_distance,
                spacing=grid_spacing,
                offset=(0, 0, 1),
            )
            common_neighbors_series = _cnn_min_neighbors(
                start=1,
                end=max_common_neighbors,
                count=min_neighbors_list_length,
            )
            max_distances.setdefault(feature_type, []).extend(
                [feature_max_distance] * len(common_neighbors_series)
            )
            min_neighbors.setdefault(feature_type, []).extend(common_neighbors_series)
    return max_distances, min_neighbors


def _cnn_max_members(
    feature_radius: dict[str, float],
    grid_spacing: float,
) -> dict[str, int]:
    voxel_volume = grid_spacing ** 3
    max_members = {}
    for feature_type, radius in feature_radius.items():
        max_volume = (4/3) * np.pi * (radius ** 3)
        max_voxels = int(np.ceil(max_volume / voxel_volume))
        max_members[feature_type] = max_voxels
    return max_members


def _cnn_max_common_neighbors(
    distance: float,
    spacing: float,
    offset: tuple[int, int, int]
) -> int:
    """Calculate the maximum number of common neighbors for two grid points.

    This function counts lattice points within `distance`
    of two grid points separated by `offset * spacing`.
    That is, given a regular 3D grid of spacing `spacing`,
    and two “seed” points at (0,0,0) and (dx,dy,dz)·spacing (where `offset = (dx,dy,dz)` are integers),
    it returns the number of grid points whose Euclidean distance to **both** seeds is ≤ `distance`.

    Parameters
    ----------
    distance
        Maximum radius for a point to be considered a neighbor.
    spacing
        Grid spacing (must be > 0).
    offset
        Integer per-axis offsets (dx, dy, dz) from the first seed to the second.
        Must not be (0, 0, 0).

    Returns
    -------
    Number of lattice points (excluding the seed points) satisfying the distance criterion.
    """
    dx, dy, dz = offset
    if distance < 0:
        raise ValueError("`distance` must be non-negative")
    if spacing <= 0:
        raise ValueError("`spacing` must be positive")
    if dx == dy == dz == 0:
        raise ValueError("`offset` cannot be (0, 0, 0)")

    # Normalize to grid units
    r2 = (distance / spacing) ** 2

    # Precompute integer radius bound
    r_int = math.isqrt(max(0, math.floor(r2)))

    # Determine loop bounds so we cover both spheres
    i_min = min(-r_int, dx - r_int)
    i_max = max( r_int, dx + r_int)
    j_min = min(-r_int, dy - r_int)
    j_max = max( r_int, dy + r_int)

    total = 0
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            # remaining budget for k from sphere at (0,0,0)
            m1 = r2 - (i*i + j*j)
            if m1 < 0:
                continue
            k1_max = math.isqrt(math.floor(m1))

            # remaining budget for k from sphere at (dx,dy,dz)
            m2 = r2 - ((i - dx)**2 + (j - dy)**2)
            if m2 < 0:
                continue
            k2_max = math.isqrt(math.floor(m2))

            # intersection of k ∈ [-k1_max..k1_max] and k ∈ [dz-k2_max..dz+k2_max]
            low  = max(-k1_max, dz - k2_max)
            high = min( k1_max, dz + k2_max)
            if high >= low:
                total += (high - low + 1)

    return max(0, total - 2)


def _cnn_min_neighbors(start: int, end: int, count: int) -> list[int]:
    """Generate a geometric sequence of given length within a closed interval.

    This function generates at most `count` integers in range [start, end] so that:
    - all but the final jump form a geometric progression
      with integer ratio ≥2,
    - the last element is exactly `end` (the last ratio may differ),
    - if no ratio ≥2 works, falls back to [start, end].

    Parameters
    ----------
    start
        Lower bound of the closed interval. Must be positive.
    end
        Upper bound of the closed interval.
    count
        Maximum number of integers to return.

    Returns
    -------
    list[int]
        A list of length ≤ count satisfying the above.
        If start > end, returns []; if count ≤ 1, returns [end];
        if start == end, returns [start].
    """
    if start > end:
        return [start]
    if count <= 1:
        return [end]
    if start == end:
        return [start]

    # Try lengths from n down to 2
    for m in range(count, 1, -1):
        if m == 2:
            return [start, end]

        # number of constant-ratio steps
        exp = m - 2

        # initial float estimate of the integer ratio
        c = int((end / start) ** (1 / exp))

        # adjust c so that a*c^exp <= b < a*(c+1)^exp
        while c > 0 and start * (c ** exp) > end:
            c -= 1
        while start * ((c + 1) ** exp) <= end:
            c += 1

        if c < 2:
            continue  # no valid integer ratio for this m

        # build the geometric part
        seq = [start * (c ** i) for i in range(exp + 1)]

        # if it already hits b exactly, done
        if seq[-1] == end:
            return seq

        # otherwise append b as the "free" final jump
        return seq + [end]

    # fallback (should never really hit this)
    return [start, end]
