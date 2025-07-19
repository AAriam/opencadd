from typing import Literal, Callable, Any
from functools import partial

import numpy as np
from scipy import ndimage
from pydantic import BaseModel, model_validator

from t2fpharm.input import validator
from t2fpharm.typing import PositiveFloat
from t2fpharm.grid import Grid


class ModelerSimpleInput(BaseModel):
    method: Literal["simple"] = "simple"

    peak_type: dict[str, Literal["min", "max"]]
    best_per_point: dict[str, bool]
    threshold_value: dict[str, float | None]
    threshold_percentile: dict[str, float | None]
    threshold_include_equal: dict[str, bool]
    filter_function: dict[str, Callable | None]
    filter_radius: dict[str, PositiveFloat | None]
    filter_extension_mode: dict[str, Literal["constant", "nearest", "wrap", "reflect"]]
    filter_extension_constant_value: dict[str, float]
    filter_gaussian_sigma: dict[str, PositiveFloat | None]
    filter_percentile: dict[str, PositiveFloat | None]

    @model_validator(mode="before")
    def _validate_modeler_simple_input(cls, values: dict[str, object]) -> dict[str, object]:
        """Preprocess and validate the input values."""
        feature_types = values["feature_types"]
        for argname, fill_value, value_validator, none_allowed in (
            ("filter_function", None, _validator_function, True),
            ("filter_radius", None, validator.is_positive_number, True),
            ("filter_extension_mode", None, _validator_extension_mode, False),
            ("filter_extension_constant_value", None, validator.is_real_number, False),
            ("filter_gaussian_sigma", None, validator.is_positive_number, True),
            ("filter_percentile", None, _validator_percentile, False),
            ("peak_type", "min", None, False),
            ("best_per_point", False, None, False),
            ("threshold_value", None, None, True),
            ("threshold_percentile", None, _validator_percentile, True),
            ("threshold_include_equal", True, None, False),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                value_validator=value_validator,
                fill_value=fill_value,
                feature_types=feature_types,
                none_allowed=none_allowed,
            )

        grid: Grid = values["grid"]
        for feature_type in feature_types:
            func_input = values["filter_function"][feature_type]
            radius = values["filter_radius"][feature_type]
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
                "mode": values["filter_extension_mode"][feature_type],
                "cval": values["filter_extension_constant_value"][feature_type]
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
                    "percentile": values["filter_percentile"][feature_type],
                    "footprint": grid.footprint_spherical(radius=radius),
                }
            elif func_input == "gaussian":
                func = ndimage.gaussian_filter
                func_args |= {
                    "sigma": _calculate_gaussian_sigma(
                        input_sigma=values["filter_gaussian_sigma"][feature_type],
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

            values["filter_function"][feature_type] = partial(func, **func_args)
        return values



def _validator_function(value: Any) -> bool:
    if isinstance(value, Callable):
        return True
    if isinstance(value, str):
        return value in ["gaussian", "mean", "percentile"]
    return False


def _validator_extension_mode(value: Any) -> bool:
    return value in ["constant", "nearest", "mirror", "reflect", "wrap"]


def _validator_percentile(value: Any) -> bool:
    return validator.is_real_number(value) and 0 <= value <= 100


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
