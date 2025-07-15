from typing import Self, Any, Sequence, TypeAlias, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from t2fpharm.input import validator
from t2fpharm.typing import PositiveFloat, PositiveInt


PriorityType: TypeAlias = Literal["lowest", "highest"]


class RemoveOverlapsInput(BaseModel):
    min_distance: dict[tuple[str, str], PositiveFloat]
    priority: np.ndarray
    highest_priority: PriorityType
    max_features: dict[str, PositiveInt]

    # Allow arbitrary types like pandas DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def _remove_overlaps_input(cls, values: dict[str, object]) -> dict[str, object]:
        """Preprocess and validate the input values."""
        feature_types = values["feature_types"]
        for argname, fill_value, none_allowed in (
            ("max_features", None, True),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                feature_types=feature_types,
                fill_value=fill_value,
                value_validator=None,
                none_allowed=none_allowed,
                require_all_types=False,
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