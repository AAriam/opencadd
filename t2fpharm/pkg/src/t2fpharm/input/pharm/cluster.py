from typing import Callable, Literal

from pydantic import model_validator
import numpy as np
from numpy.typing import NDArray

from t2fpharm.input import validator
from t2fpharm.typing import ClusteringFunction


class ClusterInput:
    function: dict[str, ClusteringFunction]
    weights: NDArray[np.float64]
    center_type: dict[str, Literal["function", "midpoint", "mean", "average"]]
    radius_type: dict[str, Literal["average", "mean", "max", "min"]]
    per_instance: bool = True

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, object]) -> dict[str, object]:
        for argname, value_validator in (
            ("function", callable),
            ("center_type", None)
            ("radius_type", None),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                feature_types=values["feature_types"],
                value_validator=value_validator,
                none_allowed=False,
                require_all_types=True,
            )
        input_weights = values["weights"]
        n_features = values["n_features"]
        if input_weights is None:
            weights = np.ones(n_features, dtype=np.float64)
        else:
            try:
                weights = np.asarray(input_weights)
            except Exception as e:
                raise ValueError(f"Invalid weights: {e}")
            if weights.ndim != 1:
                raise ValueError("Weights must be a 1D array.")
            if len(weights) != n_features:
                raise ValueError(
                    "Weights must match the number of features, "
                    f"but got {len(weights)} weights for {values['n_features']} features."
                )
        values["weights"] = weights
        return values
