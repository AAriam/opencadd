from typing import Literal

from pydantic import model_validator

from t2fpharm.input import validator
from t2fpharm.input.modeler.simple import SimpleInput
from t2fpharm.typing import PositiveInt, PositiveFloat


class LargestPeaksInput(SimpleInput):
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

    max_features: dict[str, PositiveInt | None]
    min_distance: dict[tuple[str, str], PositiveFloat] | None
    priority_factor: dict[str, float | None]

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, object]) -> dict[str, object]:
        """Preprocess and validate the input values."""
        feature_types = values["feature_types"]
        for argname, fill_value, none_allowed in (
            ("max_features", None, True),
            ("priority_factor", None, True),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                fill_value=fill_value,
                feature_types=feature_types,
                none_allowed=none_allowed,
            )
        min_distance = values["min_distance"]
        if min_distance is None:
            if any(factor is not None for factor in values["priority_factor"].values()):
                raise ValueError(
                    "`priority_factor` is provided, "
                    "but `min_distance` is None. "
                    "Please provide a valid `min_distance` value."
                )
        return values
