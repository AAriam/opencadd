from typing import Self, Sequence

import jax.numpy as jnp
from pydantic import BaseModel, Field, field_validator, model_validator


class CNNClusteringConfig(BaseModel):
    """Configuration parameters for Common Nearest Neighbor (CNN) clustering.

    Parameters
    ----------
    max_distance
        Maximum distance between two points to consider them as neighbors.
    min_neighbors
        Minimum number of common neighbors between two points
        that belong to the same cluster.
    min_members
        Minimum number of members in a cluster.
        Cluster with fewer members than this are discarded.
    max_members
        Maximum number of members in a cluster.
        If specified, clusters with more members than this
        are reclustered into smaller clusters.
        For this, either one or both of `max_distance` and `min_neighbors`
        must be a sequence of values, where the i-th value
        corresponds to the i-th clustering step.
        In each step, clusters from the last step
        with more members than `max_members`
        are reclustered until either all clusters
        have maximum `max_members` members,
        or the end of the sequence is reached.
        If only one of `max_distance` or `min_neighbors`
        is a sequence, the other one is assumed to be constant
        for all clustering steps.
        If both are sequences,
        they must have the same length,
        and the i-th value of `max_distance` and `min_neighbors`
        is used for the i-th clustering step.
    """

    max_distance: float | Sequence[float] = Field(...)
    min_neighbors: int | Sequence[int] = Field(...)
    min_members: int = Field(..., gt=0)
    max_members: int | None = Field(None)

    model_config = {"frozen": True}

    @field_validator("max_distance", mode="after")
    def _validate_max_distance(cls, max_distance: float | Sequence[float]) -> tuple[float, ...]:
        max_dist_is_single = _is_real_number(max_distance)
        if max_dist_is_single:
            max_distance = (max_distance,)
        else:
            max_distance = tuple(max_distance)
            for idx, max_dist in enumerate(max_distance):
                if not _is_real_number(max_dist):
                    raise TypeError(
                        f"Invalid type for `max_distance` at index {idx}: {type(max_dist)}"
                    )
        # ensure all values are > 0
        for idx, dist in enumerate(max_distance):
            if dist <= 0:
                raise ValueError(
                    f"All `max_distance` values must be > 0, but got {dist} at index {idx}."
                )
        return max_distance

    @field_validator("min_neighbors", mode="after")
    def _validate_min_neighbors(cls, min_neighbors: int | Sequence[int]) -> tuple[int, ...]:
        min_neighbors_is_single = _is_integer(min_neighbors)
        if min_neighbors_is_single:
            min_neighbors = (min_neighbors,)
        else:
            min_neighbors = tuple(min_neighbors)
            for idx, min_neighbor in enumerate(min_neighbors):
                if not _is_integer(min_neighbor):
                    raise TypeError(
                        f"Invalid type for `min_neighbors` at index {idx}: {type(min_neighbor)}"
                    )
        # ensure all values are > 0
        for idx, neighbor in enumerate(min_neighbors):
            if neighbor <= 0:
                raise ValueError(
                    f"All `min_neighbors` values must be > 0, but got {neighbor} at index {idx}."
                )
        return min_neighbors

    @model_validator(mode="after")
    def _validate_max_members(self) -> Self:
        name_value_pairs = (
            ("max_distance", self.max_distance),
            ("min_neighbors", self.min_neighbors),
        )
        if self.max_members is None:
            for name, value in name_value_pairs:
                if len(value) > 1:
                    raise ValueError(
                        f"`{name}` must be a single value if `max_members` is not set, "
                        f"but got {value}."
                    )
        else:
            if self.max_members < self.min_members:
                raise ValueError(
                    f"`max_members` ({self.max_members}) must be greater than or equal to "
                    f"`min_members` ({self.min_members})."
                )
            if all(len(value) == 1 for _, value in name_value_pairs):
                raise ValueError(
                    "If `max_members` is set, at least one of `max_distance` or `min_neighbors` "
                    "must be a sequence of values, but got "
                    f"{', '.join(f'{name}={value}' for name, value in name_value_pairs)}."
                )
            elif len(self.max_distance) == 1:
                self.max_distance = (self.max_distance[0],) * len(self.min_neighbors)
            elif len(self.min_neighbors) == 1:
                self.min_neighbors = (self.min_neighbors[0],) * len(self.max_distance)
            elif len(self.max_distance) != len(self.min_neighbors):
                raise ValueError(
                    "If both `max_distance` and `min_neighbors` are sequences, "
                    "they must have the same length, but got "
                    f"{len(self.max_distance)} and {len(self.min_neighbors)}."
                )
        return self


def _is_real_number(value) -> bool:
    """Check if the value is a real number (int or float).

    This covers both native Python types, as well as JAX/NumPy types.
    """
    return _is_integer(value) or _is_float(value)


def _is_integer(value) -> bool:
    """Check if the value is an integer (int or np.integer).

    This covers both native Python types, as well as JAX/NumPy types.
    """
    return jnp.issubdtype(type(value), jnp.integer)


def _is_float(value) -> bool:
    """Check if the value is a float (float or np.floating).

    This covers both native Python types, as well as JAX/NumPy types.
    """
    return jnp.issubdtype(type(value), jnp.floating)