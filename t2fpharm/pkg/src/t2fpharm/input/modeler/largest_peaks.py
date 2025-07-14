from typing import Sequence, Literal, Callable, Any
from functools import partial

from pydantic import BaseModel, model_validator

from t2fpharm.input import _validator
from t2fpharm.input.modeler import _filter
from t2fpharm.typing import PositiveInt, PositiveFloat


class LargestPeaksInput(BaseModel):
    """Validator for arguments of the `Modeler.largest_peaks` method.

    In addition to the arguments defined as fields below,
    this class requires the following parameters;
    these are not included in the created instance,
    but are used to validate/create the other parameters:

    Parameters
    ----------
    field_count
        Number of feature types in the field tensor.
        This is used to validate the length of other parameters.
    filter_radius
    filter_extension_mode
    filter_extension_constant_value
    filter_gaussian_sigma
    filter_percentile
        Parameters for the filter function.
        These are used to create a partial function
        that is returned as `filter_function`.
    """
    method: Literal["largest_peaks"] = "largest_peaks"

    peak_type: dict[str, Literal["min", "max"]]
    best_per_point: dict[str, bool]
    threshold_value: dict[str, float]
    threshold_include_equal: dict[str, bool]
    max_features: dict[str, PositiveInt]
    min_distance: dict[tuple[str, str], PositiveFloat]
    priority_factor: dict[str, float]
    filter_function: dict[str, partial]

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, object]) -> dict[str, object]:
        """Preprocess and validate the input values."""
        feature_types = values["feature_types"]
        values["filter_function"] = _filter.validate(
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
            ("max_features", None, True),
            ("min_distance", None, True),
            ("priority_factor", None, True),
        ):
            values[argname] = _validator.validate_input_dict(
                name=argname,
                value=values[argname],
                fill_value=fill_value,
                feature_types=feature_types,
                none_allowed=none_allowed,
            )
        return values

