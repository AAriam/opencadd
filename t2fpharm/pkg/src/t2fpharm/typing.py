from typing import Annotated, Callable, Sequence, Protocol, runtime_checkable

import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import pandas as pd
from pydantic import Field


PositiveInt = Annotated[
    int,
    Field(gt=0, description="A positive integer value."),
]
PositiveFloat = Annotated[
    float,
    Field(gt=0, description="A positive float value."),
]

PositiveIntTuple = Annotated[
    tuple[PositiveInt, ...],
    Field(
        min_length=1,
        description="A tuple of positive integer values."
    )
]
PositiveFloatTuple = Annotated[
    tuple[PositiveFloat, ...],
    Field(
        min_length=1,
        description="A tuple of positive float values."
    )
]

DataFrameLike = pd.DataFrame | list[dict[str, ArrayLike]]


@runtime_checkable
class ClusteringResult(Protocol):
    """Protocol for clustering results.

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


ClusteringFunction = Callable[[np.ndarray, np.ndarray], ClusteringResult]
