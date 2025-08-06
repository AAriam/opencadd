from typing import NamedTuple, Literal, Callable, Any, TypeAlias
from functools import partial

import numpy as np
from pydantic import model_validator

from sklearn.cluster import AgglomerativeClustering

from t2fpharm.input import validator
from t2fpharm.input.pharm.cluster import PharmClusterInput, ClusteringFunction, ClusteringResult, CenterTypeNoFunction
from t2fpharm.typing import PositiveInt, PositiveFloat


AggLinkageType: TypeAlias = Literal["average", "complete", "single", "ward"]
AggLinkageMetricType: TypeAlias = Literal["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"] | Callable


class PharmClusterAggInput(PharmClusterInput):
    method: Literal["Pharmacophore.cluster_agg"] = "Pharmacophore.cluster_agg"

    distance_threshold: dict[str, PositiveFloat | None]
    n_clusters: dict[str, PositiveInt | None]
    linkage: dict[str, AggLinkageType]
    metric: dict[str, AggLinkageMetricType]
    memory: Any | None
    function: dict[str, ClusteringFunction]
    center_type: dict[str, CenterTypeNoFunction]

    @model_validator(mode="before")
    def _cluster_agg_input(cls, values: dict[str, object]) -> dict[str, object]:
        feature_types = values["feature_types"]
        for argname, none_allowed, require_all_types in (
            ("n_clusters", True, False),
            ("distance_threshold", True, False),
            ("linkage", False, True),
            ("metric", False, True),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                feature_types=feature_types,
                none_allowed=none_allowed,
                require_all_types=require_all_types,
            )

        functions = {}
        for feature_type in feature_types:
            function = partial(
                agg_function,
                n_clusters=values["n_clusters"][feature_type],
                distance_threshold=values["distance_threshold"][feature_type],
                linkage=values["linkage"][feature_type],
                metric=values["metric"][feature_type],
                memory=values["memory"],
            )
            functions[feature_type] = function
        values["function"] = functions
        return values


class AggResult(NamedTuple):
    """Result of the Agg clustering."""
    labels: np.ndarray
    centers: None = None


def agg_function(
    centers: np.ndarray,
    weights: np.ndarray,
    n_clusters: PositiveInt | None,
    distance_threshold: PositiveFloat | None,
    linkage: Literal["average", "complete", "single", "ward"],
    metric: Literal["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"] | Callable,
    memory: Any | None = None,
) -> ClusteringResult:
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        memory=memory,
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(centers)
    return AggResult(labels=labels)
