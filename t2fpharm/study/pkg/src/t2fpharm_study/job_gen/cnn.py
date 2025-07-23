from typing import Sequence, Any

import numpy as np

from t2fpharm.grid import Grid
from ._filter import generate as generate_filter


def generate(
    grid: Grid,

    max_distance_base: dict[str, float],
    max_distance_multipliers: Sequence[float],
    min_neighbors_start_percents: Sequence[float],
    min_neighbors_list_lengths: Sequence[int],

    min_members_percents: Sequence[float],

    max_members_radius: dict[str, float],
    max_members_multipliers: Sequence[float],

    best_per_points: Sequence[bool],
    threshold_values: Sequence[float | dict[str, float] | None],
    threshold_percentiles: Sequence[float | dict[str, float] | None],

    center_types: Sequence[str] = ("average", "mean", "midpoint"),

    filter_none: bool = True,
    filter_radius: dict[str, float] | None = None,
    filter_radius_factors: Sequence[float] | None = None,
    filter_mean: bool = False,
    filter_percentiles: Sequence[float] | None = None,
    filter_gaussian_sigma_factors: Sequence[float] | None = None,

) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate inputs for the `t2fpharm.Modeler.cnn` method."""

    max_members_data = _max_members(
        max_members_radius=max_members_radius,
        max_members_multipliers=max_members_multipliers,
        grid_spacing=grid.spacings[0],
    )
    clustering_inputs = _clustering_params(
        max_distance_base=max_distance_base,
        max_distance_multipliers=max_distance_multipliers,
        min_neighbors_start_percents=min_neighbors_start_percents,
        min_neighbors_list_lengths=min_neighbors_list_lengths,
        grid=grid,
    )
    filters = generate_filter(
        radius_base=filter_radius,
        radius_multipliers=filter_radius_factors,
        none=filter_none,
        mean=filter_mean,
        percentiles=filter_percentiles,
        sigma_multipliers=filter_gaussian_sigma_factors,
    )

    single_jobs = []
    grouped_jobs = []
    job_idx = 0
    for max_members_mult_idx, (max_members_mult, max_members) in enumerate(max_members_data):
        min_members_data = _min_members(
            max_members=max_members,
            min_members_percents=min_members_percents,
        )
        min_members_dicts = [min_members for _, min_members in min_members_data]
        for filter_ in filters:
            for clustering_input in clustering_inputs:
                for threshold_value_idx, threshold_value in enumerate(threshold_values):
                    for threshold_percentile_idx, threshold_percentile in enumerate(threshold_percentiles):
                        for best_per_point in best_per_points:
                            base_job = filter_ | clustering_input | {
                                "max_members": max_members,
                                "max_members_factor": max_members_mult,
                                "max_members_factor_idx": max_members_mult_idx,

                                "best_per_point": best_per_point,

                                "threshold_value": threshold_value,
                                "threshold_value_idx": threshold_value_idx,

                                "threshold_percentile": threshold_percentile,
                                "threshold_percentile_idx": threshold_percentile_idx,
                            }
                            grouped_jobs.append(
                                base_job | {
                                    "job_idx": job_idx,
                                    "min_members_dicts": min_members_dicts,
                                    "center_types": center_types,
                                }
                            )
                            for min_members_percent_idx, (min_members_percent, min_members) in enumerate(min_members_data):
                                for center_type in center_types:
                                    single_jobs.append(
                                        base_job | {
                                            "job_idx": job_idx,
                                            "min_members": min_members,
                                            "min_members_percent": min_members_percent,
                                            "min_members_percent_idx": min_members_percent_idx,
                                            "center_type": center_type,
                                        }
                                    )
                                    job_idx += 1
    return single_jobs, grouped_jobs


def _clustering_params(
    max_distance_base: dict[str, float],
    max_distance_multipliers: Sequence[float],
    min_neighbors_start_percents: Sequence[float],
    min_neighbors_list_lengths: Sequence[int],
    grid: Grid,
):
    max_dists_scaled = _max_dist(
        max_distance_base=max_distance_base,
        max_distance_multipliers=max_distance_multipliers
    )
    out = []
    for max_dist_mult_idx, (max_dist_mult, max_dist_scaled) in enumerate(max_dists_scaled):
        for min_neigh_start_percent_idx, min_neigh_start_percent in enumerate(min_neighbors_start_percents):
            for min_neigh_list_len_idx, min_neigh_list_len in enumerate(min_neighbors_list_lengths):
                max_distance, min_neighbors = _clustering_params_single(
                    grid=grid,
                    max_distance=max_dist_scaled,
                    min_neighbors_start_percent=min_neigh_start_percent,
                    min_neighbors_list_length=min_neigh_list_len,
                )
                out.append(
                    {
                        "max_distance": max_distance,
                        "max_distance_mult": max_dist_mult,
                        "max_distance_mult_idx": max_dist_mult_idx,

                        "min_neighbors": min_neighbors,
                        "min_neighbors_start_percent": min_neigh_start_percent,
                        "min_neighbors_start_percent_idx": min_neigh_start_percent_idx,
                        "min_neighbors_list_length": min_neigh_list_len,
                        "min_neighbors_list_length_idx": min_neigh_list_len_idx,
                    }
                )
    return out


def _clustering_params_single(
    grid: Grid,
    max_distance: dict[str, float],
    min_neighbors_start_percent: float,
    min_neighbors_list_length: int,
) -> dict[str, list[float]]:
    """Calculate a single set of `max_distance` and `min_neighbors` arguments for CNN clustering.

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
    grid_unique_distances = grid.unique_distances
    for feature_type, radius in max_distance.items():
        feature_max_distances = np.sort(
            grid_unique_distances[grid_unique_distances <= radius]
        )[::-1].tolist()
        for feature_max_distance in feature_max_distances:

            # Calculate the maximum number of common neighbors
            # between two adjacent grid points for the current max distance.
            max_common_neighbors = grid.common_neighbors_count(
                neighborhood_radius=feature_max_distance,
                point1=(0, 0, 1),
            )

            # Generate a geometric sequence (evenly spaced on log scale)
            # within the closed interval `[start, stop]`
            # with at most `num` unique integers.
            min_neighbors_series = np.unique(
                np.round(
                    np.geomspace(
                        start=int(max(1, np.rint((min_neighbors_start_percent / 100) * max_common_neighbors))),
                        stop=max_common_neighbors,
                        num=min_neighbors_list_length,
                        endpoint=True
                    )
                ).astype(int)
            ).tolist()

            min_neighbors.setdefault(feature_type, []).extend(min_neighbors_series)
            max_distances.setdefault(feature_type, []).extend(
                [feature_max_distance] * len(min_neighbors_series)
            )

    return max_distances, min_neighbors


def _max_dist(
    max_distance_base: dict[str, float],
    max_distance_multipliers: Sequence[float],
) -> list[tuple[float, dict[str, float]]]:
    """Generate different maximum distance values for each feature type.

    This is done by scaling the base maximum distance values by the multipliers.

    Parameters
    ----------
    max_distance_base
        Dictionary mapping feature types to their base maximum distance values.
    max_distance_multipliers
        Sequence of multipliers to scale the base maximum distances.

    Returns
    -------
    List of 2-tuples, each containing a multiplier
    and a dictionary mapping feature types to the maximum distance values
    calculated by scaling the base maximum distances by the multiplier.
    """
    out = []
    for max_dist_mult in max_distance_multipliers:
        max_dist_scaled = {
            feature_type: max_dist_mult * max_distance_value
            for feature_type, max_distance_value in max_distance_base.items()
        }
        out.append((max_dist_mult, max_dist_scaled))
    return out


def _max_members(
    max_members_radius: dict[str, float],
    max_members_multipliers: Sequence[float],
    grid_spacing: float
) -> list[tuple[float, dict[str, int]]]:
    """Calculate maximum number of cluster members for each feature.

    This is done by calculating the maximum number of grid points that can fit
    within a sphere defined by the feature's radius, given the grid spacing.
    These values are then scaled by the multipliers provided
    to generate a range of `max_members` arguments for the CNN clustering method.

    Parameters
    ----------
    max_members_radius
        Dictionary mapping feature types to their radii.
    max_members_multipliers
        Sequence of multipliers to scale the base maximum members.
    grid_spacing
        Spacing between grid points.

    Returns
    -------
    List of 2-tuples, each containing a multiplier
    and a dictionary mapping feature types
    to the maximum number of cluster members
    calculated by scaling the base maximum members by the multiplier.
    """
    max_members_base = _max_members_base(
        feature_radius=max_members_radius,
        grid_spacing=grid_spacing
    )
    out = []
    for max_members_mult in max_members_multipliers:
        max_members = {
            feature_type: int(np.ceil(max_members_mult * max_member))
            for feature_type, max_member in max_members_base.items()
        }
        out.append((max_members_mult, max_members))
    return out


def _max_members_base(
    feature_radius: dict[str, float],
    grid_spacing: float,
) -> dict[str, int]:
    """Calculate the maximum number of cluster members for each feature type based on its radius.

    This is done by calculating the volume of a sphere with the given radius
    and dividing it by the volume of a voxel defined by the grid spacing
    to determine how many voxels can fit within that sphere.

    Parameters
    ----------
    feature_radius
        Dictionary mapping feature types to their radii.
    grid_spacing
        Spacing between grid points.

    Returns
    -------
    Dictionary mapping feature types to the maximum number of grid points
    that can fit within a sphere defined by the feature's radius.
    """
    voxel_volume = grid_spacing ** 3
    max_members = {}
    for feature_type, radius in feature_radius.items():
        max_volume = (4/3) * np.pi * (radius ** 3)
        max_voxels = int(np.ceil(max_volume / voxel_volume))
        max_members[feature_type] = max_voxels
    return max_members


def _min_members(
    max_members: dict[str, int],
    min_members_percents: Sequence[float],
) -> list[tuple[float, dict[str, int]]]:
    """Calculate minimum number of cluster members for each feature type based on their maximum members.

    This is done by applying the provided percentages to the provided maximum members counts
    to determine the minimum required members for each feature type.

    Parameters
    ----------
    max_members
        Dictionary mapping feature types to their maximum number of members.
    min_members_percents
        Sequence of percentages to apply to the maximum members
        to calculate the minimum required members.

    Returns
    -------
    List of 2-tuples, each containing a percentage
    and a dictionary mapping feature types to the minimum number of members
    calculated by applying the percentage to the maximum members.
    """
    out = []
    for percent in min_members_percents:
        min_members = {
            feature_type: int(np.ceil((percent / 100) * max_member))
            for feature_type, max_member in max_members.items()
        }
        out.append((percent, min_members))
    return out
