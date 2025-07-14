from typing import Any, Callable, Sequence
from functools import partial

import numpy as np
from scipy import ndimage

from t2fpharm.input import _validator
from t2fpharm.grid import Grid


def validate(
    filter_function: Any,
    filter_radius: Any,
    filter_extension_mode: Any,
    filter_extension_constant_value: Any,
    filter_gaussian_sigma: Any,
    filter_percentile: Any,
    feature_types: Sequence[str],
    grid: Grid,
):
    args = {}
    for name, value, fill_value, validator, none_allowed in (
        ("filter_function", filter_function, None, _validator_function, True),
        ("filter_radius", filter_radius, None, _validator.is_positive_number, True),
        ("filter_extension_mode", filter_extension_mode, "constant", _validator_extension_mode, False),
        ("filter_extension_constant_value", filter_extension_constant_value, 0, _validator.is_real_number, False),
        ("filter_gaussian_sigma", filter_gaussian_sigma, None, _validator.is_positive_number, True),
        ("filter_percentile", filter_percentile, 50, _validator_percentile, False),
    ):
        args[name] = _validator.validate_input_dict(
            name=name,
            value=value,
            fill_value=fill_value,
            value_validator=validator,
            feature_types=feature_types,
            none_allowed=none_allowed,
        )
    functions = {}
    for feature_type in feature_types:
        func_input = args["filter_function"][feature_type]
        radius = args["filter_radius"][feature_type]
        if func_input is None:
            if radius is not None:
                raise ValueError(
                    f"Filter radius for feature type '{feature_type}' is not None, "
                    "but no filter function is provided. "
                    "Please provide a valid filter function or unset radius."
                )
            continue
        if radius is None:
            raise ValueError(
                f"Filter radius for feature type '{feature_type}' is None, "
                "but a filter function is provided. "
                "Please provide a valid radius."
            )
        func_args = {
            "mode": args["filter_extension_mode"][feature_type],
            "cval": args["filter_extension_constant_value"][feature_type]
        }
        if isinstance(func_input, Callable) or func_input == "mean":
            func = ndimage.generic_filter
            func_args |= {
                "function": np.mean if func_input == "mean" else func_input,
                "footprint": grid.footprint_spherical(radius=radius),
            }
        elif func_input == "percentile":
            func = ndimage.percentile_filter
            func_args |= {
                "percentile": args["filter_percentile"][feature_type],
                "footprint": grid.footprint_spherical(radius=radius),
            }
        elif func_input == "gaussian":
            func = ndimage.gaussian_filter
            func_args |= {
                "sigma": _calculate_gaussian_sigma(
                    input_sigma=args["filter_gaussian_sigma"][feature_type],
                    radius=radius,
                    grid=grid,
                ),
                "radius": int(np.rint(radius / grid.spacings[0])),
            }
        else:
            raise ValueError(
                f"Invalid filter function '{func_input}' for feature type '{feature_type}'. "
                "Expected a callable or one of ['gaussian', 'mean', 'percentile']."
            )

        functions[feature_type] = partial(func, **func_args)
    return functions


def _validator_function(value: Any) -> bool:
    if isinstance(value, Callable):
        return True
    if isinstance(value, str):
        return value in ["gaussian", "mean", "percentile"]
    return False


def _validator_extension_mode(value: Any) -> bool:
    return value in ["constant", "nearest", "mirror", "reflect", "wrap"]


def _validator_percentile(value: Any) -> bool:
    return _validator.is_real_number(value) and 0 <= value <= 100


def _calculate_gaussian_sigma(
    input_sigma: float | None,
    radius: float,
    grid: Grid,
) -> float:
    if input_sigma is not None:
        return input_sigma / grid.spacings[0]
    gaussian_full_width_at_half_max = 2 * np.sqrt(2 * np.log(2))
    sigma_real_units = radius / gaussian_full_width_at_half_max
    return sigma_real_units / grid.spacings[0]
