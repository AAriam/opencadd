from typing import Literal, Callable

from pydantic import BaseModel, model_validator

from t2fpharm.input import validator
from t2fpharm.input.modeler import _filter
from t2fpharm.typing import PositiveFloat


class SimpleInput(BaseModel):
    method: Literal["simple"] = "simple"

    peak_type: dict[str, Literal["min", "max"]]
    best_per_point: dict[str, bool]
    threshold_value: dict[str, float | None]
    threshold_include_equal: dict[str, bool]
    filter_function: dict[str, Callable | None]
    filter_radius: dict[str, PositiveFloat | None]
    filter_extension_mode: dict[str, Literal["constant", "nearest", "wrap", "reflect"]]
    filter_extension_constant_value: dict[str, float]
    filter_gaussian_sigma: dict[str, PositiveFloat | None]
    filter_percentile: dict[str, PositiveFloat | None]

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, object]) -> dict[str, object]:
        """Preprocess and validate the input values."""
        feature_types = values["feature_types"]
        values |= _filter.validate(
            filter_function=values["filter_function"],
            filter_radius=values["filter_radius"],
            filter_extension_mode=values["filter_extension_mode"],
            filter_extension_constant_value=values["filter_extension_constant_value"],
            filter_gaussian_sigma=values["filter_gaussian_sigma"],
            filter_percentile=values["filter_percentile"],
            feature_types=feature_types,
            grid=values["grid"],
        )
        for argname, fill_value, none_allowed in (
            ("peak_type", "min", False),
            ("best_per_point", False, False),
            ("threshold_value", None, True),
            ("threshold_include_equal", True, False),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                fill_value=fill_value,
                feature_types=feature_types,
                none_allowed=none_allowed,
            )
        return values
