from typing import Sequence, NamedTuple
from functools import partial

import numpy as np
from pydantic import model_validator

import scids

from t2fpharm.input import validator
from t2fpharm.typing import (
    PositiveInt, PositiveFloat, PositiveIntTuple, PositiveFloatTuple, ClusteringFunction, ClusteringResult
)


class CNNInput:
    clustering_function: dict[str, ClusteringFunction]
    max_distance: dict[str, PositiveFloatTuple]
    min_neighbors: dict[str, PositiveIntTuple]
    min_members: dict[str, PositiveInt]
    max_members: dict[str, PositiveInt | None]
    per_instance: bool = True

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, object]) -> dict[str, object]:
        feature_types = values["feature_types"]
        for argname in (
            "max_distance",
            "min_neighbors",
            "min_members",
            "max_members",
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                feature_types=feature_types,
                none_allowed=False,
                require_all_types=True,
            )

        functions = {}
        for feature_type in feature_types:
            function = partial(
                cnn_function,
                max_distance=values["max_distance"][feature_type],
                min_neighbors=values["min_neighbors"][feature_type],
                min_members=values["min_members"][feature_type],
                max_members=values["max_members"][feature_type],
            )
            functions[feature_type] = function
        values["clustering_function"] = functions
        return values


class CNNResult(NamedTuple):
    """Result of the CNN clustering."""
    labels: np.ndarray
    centers: None = None


def cnn_function(
    centers: np.ndarray,
    weights: np.ndarray,
    max_distance: PositiveFloat | Sequence[PositiveFloat],
    min_neighbors: PositiveInt | Sequence[PositiveInt],
    min_members: PositiveInt,
    max_members: PositiveInt | None,
) -> ClusteringResult:
    labels = scids.pointcloud.from_array(centers).cluster_cnn(
        max_distance=max_distance,
        min_neighbors=min_neighbors,
        min_members=min_members,
        max_members=max_members,
    )
    return CNNResult(labels=labels)
