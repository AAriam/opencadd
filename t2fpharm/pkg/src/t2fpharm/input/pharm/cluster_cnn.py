from typing import Sequence, NamedTuple, Literal
from functools import partial

import numpy as np
from pydantic import model_validator

import scids

from t2fpharm.input import validator
from t2fpharm.input.pharm.cluster import PharmClusterInput, ClusteringFunction, ClusteringResult, CenterTypeNoFunction
from t2fpharm.typing import (
    PositiveInt, PositiveFloat, PositiveIntTuple, PositiveFloatTuple
)


class PharmClusterCNNInput(PharmClusterInput):
    method: Literal["Pharmacophore.cluster_cnn"] = "Pharmacophore.cluster_cnn"

    max_distance: dict[str, PositiveFloat | PositiveFloatTuple]
    min_neighbors: dict[str, PositiveFloat | PositiveIntTuple]
    min_members: dict[str, PositiveInt]
    max_members: dict[str, PositiveInt | None]
    function: dict[str, ClusteringFunction]
    center_type: dict[str, CenterTypeNoFunction]

    @model_validator(mode="before")
    def _cluster_cnn_input(cls, values: dict[str, object]) -> dict[str, object]:
        feature_types = values["feature_types"]
        for argname, none_allowed in (
            ("max_distance", False),
            ("min_neighbors", False),
            ("min_members", False),
            ("max_members", True),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                feature_types=feature_types,
                none_allowed=none_allowed,
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
        values["function"] = functions
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
    if np.any(labels < 0):
        raise RuntimeError("CNN clustering returned negative labels.")
    # Adjust labels to start from 0, with -1 indicating noise
    labels -= 1
    return CNNResult(labels=labels)
