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

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, object]) -> dict[str, object]:
        max_distant_raw = values.get("max_distance")
        min_neighbors_raw = values.get("min_neighbors")
        max_members = values.get("max_members")
        min_members = values.get("min_members")

        if _is_real_number(max_distant_raw):
            max_distance = (float(max_distant_raw),)
        elif not isinstance(max_distant_raw, (str, bytes)):
            max_distance = tuple(max_distant_raw)
        else:
            raise TypeError(
                f"Invalid type for `max_distance`; "
                f"got {max_distant_raw} with type {type(max_distant_raw)}"
            )

        if _is_integer(min_neighbors_raw):
            min_neighbors = (int(min_neighbors_raw),)
        elif not isinstance(min_neighbors_raw, (str, bytes)):
            min_neighbors = tuple(min_neighbors_raw)
        else:
            raise TypeError(
                f"Invalid type for `min_neighbors`; "
                f"got {min_neighbors_raw} with type {type(min_neighbors_raw)}"
            )

        name_value_pairs = (
            ("max_distance", max_distance),
            ("min_neighbors", min_neighbors),
        )
        if max_members is None:
            for name, value in name_value_pairs:
                if len(value) > 1:
                    raise ValueError(
                        f"`{name}` must be a single value if `max_members` is not set, "
                        f"but got {value}."
                    )
        else:
            if max_members < min_members:
                raise ValueError(
                    f"`max_members` ({max_members}) must be greater than or equal to "
                    f"`min_members` ({min_members})."
                )
            if all(len(value) == 1 for _, value in name_value_pairs):
                raise ValueError(
                    "If `max_members` is set, at least one of `max_distance` or `min_neighbors` "
                    "must be a sequence of values, but got "
                    f"{', '.join(f'{name}={value}' for name, value in name_value_pairs)}."
                )
            if len(max_distance) == 1:
                max_distance = max_distance * len(min_neighbors)
            elif len(min_neighbors) == 1:
                min_neighbors = min_neighbors * len(max_distance)
            elif len(max_distance) != len(min_neighbors):
                raise ValueError(
                    "When both `max_distance` and `min_neighbors` are sequences, "
                    "they must have equal length, but got "
                    f"{len(max_distance)} vs {len(min_neighbors)}."
                )

        values["max_distance"]  = max_distance
        values["min_neighbors"] = min_neighbors
        return values

    @field_validator("max_distance", mode="after")
    def _validate_max_distance(cls, max_distance: float | Sequence[float]) -> tuple[float, ...]:
        for idx, max_dist in enumerate(max_distance):
            if not _is_real_number(max_dist):
                raise TypeError(
                    f"Invalid type for `max_distance` at index {idx}: {type(max_dist)}"
                )
            if max_dist <= 0:
                raise ValueError(
                    f"All `max_distance` values must be > 0, but got {max_dist} at index {idx}."
                )
        return max_distance

    @field_validator("min_neighbors", mode="after")
    def _validate_min_neighbors(cls, min_neighbors: int | Sequence[int]) -> tuple[int, ...]:
        for idx, min_neighbor in enumerate(min_neighbors):
            if not _is_integer(min_neighbor):
                raise TypeError(
                    f"Invalid type for `min_neighbors` at index {idx}: {type(min_neighbor)}"
                )
            if min_neighbor <= 0:
                raise ValueError(
                    f"All `min_neighbors` values must be > 0, but got {min_neighbor} at index {idx}."
                )
        return min_neighbors


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
