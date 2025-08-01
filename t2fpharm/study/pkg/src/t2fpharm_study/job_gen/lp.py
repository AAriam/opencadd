from typing import Any, Sequence

from ._filter import generate as generate_filter


def generate(
    min_distance_base: dict[tuple[str, str], float],
    min_distance_multipliers: Sequence[float],
    max_features_base: dict[str, int],
    max_features_multipliers: Sequence[int],
    priority_factors: Sequence[dict[str, float] | None] = (None,),
    threshold_values: Sequence[float | dict[str, float] | None] = (None,),
    filter_none: bool = True,
    filter_radius_base: dict[str, float] | None = None,
    filter_radius_multipliers: Sequence[float] | None = None,
    filter_mean: bool = False,
    filter_percentiles: Sequence[float] | None = None,
    filter_gaussian_sigma_multipliers: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate inputs for the `t2fpharm.Modeler.largest_peaks` method."""
    min_distance_data = _min_distance(
        base=min_distance_base,
        multipliers=min_distance_multipliers,
    )
    max_features_data = _max_features(
        base=max_features_base,
        multipliers=max_features_multipliers,
    )
    filters = generate_filter(
        radius_base=filter_radius_base,
        radius_multipliers=filter_radius_multipliers,
        none=filter_none,
        mean=filter_mean,
        percentiles=filter_percentiles,
        sigma_multipliers=filter_gaussian_sigma_multipliers,
    )

    out = []
    job_idx = 0
    for filter_ in filters:
        for min_distance_mult_idx, (min_distance_mult, min_distance) in enumerate(min_distance_data):
            for max_features_mult_idx, (max_features_mult, max_features) in enumerate(max_features_data):
                for priority_factor_idx, priority_factor in enumerate(priority_factors):
                    for threshold_value_idx, threshold_value in enumerate(threshold_values):
                        out.append(
                            filter_ | {
                                "job_idx": job_idx,
                                "min_distance": min_distance,
                                "min_distance_mult": min_distance_mult,
                                "min_distance_mult_idx": min_distance_mult_idx,
                                "priority_factor": priority_factor,
                                "priority_factor_idx": priority_factor_idx,
                                "max_features": max_features,
                                "max_features_mult": max_features_mult,
                                "max_features_mult_idx": max_features_mult_idx,
                                "threshold_value": threshold_value,
                                "threshold_value_idx": threshold_value_idx,
                            }
                        )
                        job_idx += 1
    return out, out


def _min_distance(base: dict[tuple[str, str], float], multipliers: Sequence[float]):
    return [
        (mult, {k: v * mult for k, v in base.items()})
        for mult in multipliers
    ]


def _max_features(base: dict[str, int], multipliers: Sequence[int]):
    return [
        (mult, {k: v * mult for k, v in base.items()})
        for mult in multipliers
    ]
