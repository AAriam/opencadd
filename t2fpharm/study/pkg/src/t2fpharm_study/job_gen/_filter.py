from typing import Any, Sequence


def generate(
    radius_base: dict[str, float] | None,
    radius_multipliers: Sequence[float] | None,
    none: bool,
    mean: bool,
    percentiles: Sequence[float] | None,
    sigma_multipliers: Sequence[float] | None,
) -> list[dict[str, Any]]:
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
    out = []
    if none:
        out.append({"filter_function": None})
    if (mean or percentiles or sigma_multipliers) and radius_base is None:
        raise ValueError(
            "`radius_base` must be provided if `mean`, `percentiles`, or `sigma_factors` are used."
        )
    for radius_mult_idx, radius_mult in enumerate(radius_multipliers or []):
        radius = {k: v * radius_mult for k, v in radius_base.items()}
        if mean:
            out.append(
                {
                    "filter_function": "mean",
                    "filter_radius": radius,
                    "filter_radius_mult": radius_mult,
                    "filter_radius_mult_idx": radius_mult_idx
                }
            )
        for percentile_idx, percentile in enumerate(percentiles or []):
            out.append(
                {
                    "filter_function": "percentile",
                    "filter_radius": radius,
                    "filter_radius_mult": radius_mult,
                    "filter_radius_mult_idx": radius_mult_idx,
                    "filter_percentile": percentile,
                    "filter_percentile_idx": percentile_idx
                }
            )
        for sigma_mult_idx, sigma_mult in enumerate(sigma_multipliers or []):
            sigma = {k: v * sigma_mult for k, v in radius.items()}
            out.append(
                {
                    "filter_function": "gaussian",
                    "filter_radius": radius,
                    "filter_radius_mult": radius_mult,
                    "filter_radius_mult_idx": radius_mult_idx,
                    "filter_gaussian_sigma": sigma,
                    "filter_gaussian_sigma_mult": sigma_mult,
                    "filter_gaussian_sigma_mult_idx": sigma_mult_idx
                }
            )
    return out
