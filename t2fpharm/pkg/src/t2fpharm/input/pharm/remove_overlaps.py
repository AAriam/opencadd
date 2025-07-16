from typing import Self, Any, Sequence, TypeAlias, Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from t2fpharm.input import validator
from t2fpharm.typing import PositiveFloat, PositiveInt


PriorityType: TypeAlias = Literal["lowest", "highest"]


class RemoveOverlapsInput(BaseModel):
    min_distance: dict[tuple[str, str], PositiveFloat]
    priority: NDArray[np.float64]
    highest_priority: PriorityType
    max_features: dict[str, PositiveInt | None]

    min_distance_asarray: NDArray[np.float64]
    max_features_asarray: NDArray[np.int64] | None = None

    # Allow arbitrary types like pandas DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def _remove_overlaps_input(cls, values: dict[str, object]) -> dict[str, object]:
        """Preprocess and validate the input values."""
        def validate_min_distance():
            min_distance = values["min_distance"]
            if validator.is_real_number(min_distance):
                if not validator.is_positive_number(min_distance):
                    raise ValueError(f"Minimum distance must be positive, got {min_distance}")
                return np.full((n_types, n_types), min_distance, dtype=np.float64)
            if not isinstance(min_distance, dict):
                raise ValueError(
                    f"Invalid type for `min_distance`: {type(min_distance)}. "
                    "Expected a positive number or a dictionary."
                )
            min_distance_asarray = np.zeros((n_types, n_types), dtype=np.float64)
            seen = set()
            for type_pair, distance in min_distance.items():
                if not isinstance(type_pair, tuple | list) or len(type_pair) != 2:
                    raise ValueError(
                        f"Invalid type pair for `min_distance` key: {type(type_pair)}. "
                        f"Expected a tuple or list of two feature types, got {type_pair}."
                    )
                for typ in type_pair:
                    if typ not in feature_types:
                        raise ValueError(f"Invalid feature type: {typ}. Allowed: {feature_types}")
                unique_pair = tuple(sorted(type_pair))
                if unique_pair in seen:
                    raise ValueError(
                        f"Duplicate minimum distance for feature types {type_pair}: {distance}"
                    )
                seen.add(unique_pair)
                if distance < 0:
                    raise ValueError(
                        f"Minimum distance must be non-negative, got {distance} for types {unique_pair}"
                    )
                i, j = feature_types.index(unique_pair[0]), feature_types.index(unique_pair[1])
                min_distance_asarray[i, j] = min_distance_asarray[j, i] = distance
            return min_distance_asarray

        feature_types = list(values["feature_types"])
        n_types = len(feature_types)
        n_features = values["n_features"]

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

        values["min_distance_asarray"] = validate_min_distance()

        # Convert priority to numpy array and validate length
        priority = np.asarray(values["priority"])
        if priority.ndim != 1 or priority.shape[0] != n_features:
            raise ValueError(
                f"Priority must be 1D with length {n_features}, but got shape {priority.shape}"
            )
        values["priority"] = priority

        if any(max_feat is not None for max_feat in values["max_features"].values()):
            max_features_asarray = np.full(n_types, np.iinfo(np.int64).max, dtype=np.int64)
            for feature_type, max_count in values["max_features"].items():
                max_features_asarray[feature_types.index(feature_type)] = max_count
            values["max_features_asarray"] = max_features_asarray
        return values