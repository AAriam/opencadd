from typing import Callable, Literal, Sequence, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator
import numpy as np
from numpy.typing import NDArray

from t2fpharm.input import validator
from t2fpharm.typing import PositiveInt


@runtime_checkable
class ClusteringResult(Protocol):
    """Protocol for clustering results.

    Any given clustering function must
    return an instance of this protocol.

    Attributes
    ----------
    labels
        1D integer array/sequence of cluster labels
        for each feature center in the input array.
        Labels that are 0 or negative are considered background/noise.
    centers
        Optional 2D array/sequence of coordinates of cluster centers.
        If available, a cluster with label `i` must have its center
        at `centers[i]`.
    """
    labels: np.ndarray | Sequence[int]
    centers: np.ndarray | Sequence[tuple[float, float, float]] | None


ClusteringFunction: TypeAlias = Callable[[np.ndarray, np.ndarray], ClusteringResult]
CenterType: TypeAlias = Literal["average", "mean", "midpoint", "function"]
CenterTypeNoFunction: TypeAlias = Literal["average", "mean", "midpoint"]
RadiusType: TypeAlias = Literal["average", "mean", "max", "min"]


class PharmClusterInput(BaseModel):
    method: Literal["Pharmacophore.cluster"] = "Pharmacophore.cluster"

    function: dict[str, ClusteringFunction]
    min_members: dict[str, PositiveInt]
    noise_as_singleton: dict[str, bool]
    weights: NDArray[np.float64]
    center_type: dict[str, CenterType]
    radius_type: dict[str, RadiusType]
    per_instance: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def _cluster_input(cls, values: dict[str, object]) -> dict[str, object]:
        for argname, value_validator, none_allowed in (
            ("function", callable, False),
            ("min_members", None, False),
            ("noise_as_singleton", None, False),
            ("center_type", None, False),
            ("radius_type", None, False),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                feature_types=values["feature_types"],
                value_validator=value_validator,
                none_allowed=none_allowed,
                require_all_types=True,
            )
        values["noise_as_singleton"] = {
            feature_type: values["min_members"]["feature_type"] == 1 and value is True
            for feature_type, value in values["noise_as_singleton"].items()
        }
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
