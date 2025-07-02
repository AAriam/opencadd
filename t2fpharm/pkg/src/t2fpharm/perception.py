
from typing import Sequence, Self

import pandas as pd
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

import scids

from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.pharmacophore import Pharmacophore
from t2fpharm.typing import PositiveInt, PositiveIntTuple, PositiveFloatTuple, is_real_number, is_integer, is_float


class Perception:
    def __init__(
        self,
        pocket: Pocket,
        field: Field,
    ):
        if not isinstance(pocket, Pocket):
            raise TypeError(f"Expected Pocket object, got {type(pocket).__name__}.")
        if not isinstance(field, Field):
            raise TypeError(f"Expected Field object, got {type(field).__name__}.")
        if pocket.grid != field.grid:
            raise ValueError(
                "Pocket and field must have the same grid, "
                f"but got pocket grid {pocket.grid} and field grid {field.grid}."
            )
        if pocket.tensor.shape != field.tensor.shape[1:]:
            raise ValueError(
                "Pocket and field tensors must have the same shape along their last dimensions, "
                f"but got pocket tensor shape {pocket.tensor.shape} "
                f"and field tensor shape {field.tensor.shape}."
            )
        self._pocket = pocket
        self._field = field
        return

    @property
    def pocket(self) -> Pocket:
        return self._pocket

    @property
    def field(self) -> Field:
        return self._field

    def original(
        self,
        max_value: float | Sequence[float],
        max_distance: float | Sequence[float] | Sequence[Sequence[float]] = 1.21,
        min_neighbors: int | Sequence[int] | Sequence[Sequence[int]] = (6, 12, 16, 20),
        min_members: int | Sequence[int] = 15,
        max_members: int | None | Sequence[int | None] = 80,
    ):
        args = _OriginalArgs(
            field_count=len(self.field.ids),
            max_value=max_value,
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            min_members=min_members,
            max_members=max_members,
        )
        field_masks = np.less_equal(
            self.field.tensor,
            np.array(args.max_value).reshape(-1, *(1,) * (self.field.tensor.ndim - 1)),
        )
        final_masks = np.logical_and(
            field_masks,
            self.pocket.voxels
        )

        features = []
        for field_idx, field_id in enumerate(self.field.ids):
            for instance_idx in np.ndindex(self.field.batch_shape):
                mask = final_masks[field_idx, *instance_idx]
                points = self.pocket.coordinates[mask]
                labels = scids.pointcloud.from_array(points).cluster_cnn(
                    max_distance=args.max_distance[field_idx],
                    min_neighbors=args.min_neighbors[field_idx],
                    min_members=args.min_members[field_idx],
                    max_members=args.max_members[field_idx],
                )
                cluster_sizes = np.bincount(labels)[1:]  # Exclude background label (0)
                cluster_count = cluster_sizes.size
                for cluster_label in range(1, cluster_count + 1):
                    n_points = cluster_sizes[cluster_label - 1]
                    volume = n_points * self.pocket.voxel_volume
                    point_coordinates = points[labels == cluster_label]
                    features.append(
                        {
                            "instance": instance_idx,
                            "type": field_id,
                            "n_points": n_points,
                            "volume": volume,
                            "radius": (volume / ((4/3) * np.pi)) ** (1/3),
                            "center": point_coordinates.mean(axis=0),
                            "label": cluster_label,
                            "points": point_coordinates,
                        }
                    )
        return Pharmacophore(
            features=pd.DataFrame(features),
            pocket=self.pocket,
            field=self.field,
            args=args,
        )


class _OriginalArgs(BaseModel):
    field_count: int
    max_value: tuple[float, ...]
    max_distance: tuple[PositiveFloatTuple, ...]
    min_neighbors: tuple[PositiveIntTuple, ...]
    min_members: tuple[PositiveInt, ...]
    max_members: tuple[PositiveInt | None, ...]

    @model_validator(mode="before")
    def _preprocess(cls, values: dict[str, object]) -> dict[str, object]:
        field_count = values["field_count"]
        max_value_raw = values["max_value"]
        max_distant_raw = values["max_distance"]
        min_neighbors_raw = values["min_neighbors"]
        min_members_raw = values["min_members"]
        max_members_raw = values["max_members"]

        # Process `max_value`
        if is_real_number(max_value_raw):
            max_value = (max_value_raw,) * field_count
        else:
            max_value = tuple(max_value_raw)
            if len(max_value) != field_count:
                raise ValueError(
                    f"`max_value` must have length {field_count}, "
                    f"but got {len(max_value)}."
                )

        # Process `max_distance`
        if is_real_number(max_distant_raw):
            max_distance = [(max_distant_raw,) for _ in range(field_count)]
        elif isinstance(max_distant_raw, Sequence) and not isinstance(max_distant_raw, str | bytes):
            if all(is_real_number(x) for x in max_distant_raw):
                max_dist = tuple(max_distant_raw)
                max_distance = [max_dist for _ in range(field_count)]
            elif all(isinstance(x, Sequence) and not isinstance(x, str | bytes) for x in max_distant_raw):
                max_distance = [tuple(x) for x in max_distant_raw]
                if len(max_distance) != field_count:
                    raise ValueError(
                        f"`max_distance` must have length {field_count}, "
                        f"but got {len(max_distance)}."
                    )
            else:
                raise TypeError(
                    f"Invalid type for `max_distance`; "
                    f"got {max_distant_raw} with type {type(max_distant_raw)}"
                )
        else:
            raise TypeError(
                f"Invalid type for `max_distance`; "
                f"got {max_distant_raw} with type {type(max_distant_raw)}"
            )

        # Process `min_neighbors`
        if is_integer(min_neighbors_raw):
            min_neighbors = [(min_neighbors_raw,) for _ in range(field_count)]
        elif isinstance(min_neighbors_raw, Sequence) and not isinstance(min_neighbors_raw, str | bytes):
            if all(is_integer(x) for x in min_neighbors_raw):
                min_neigh = tuple(min_neighbors_raw)
                min_neighbors = [min_neigh for _ in range(field_count)]
            elif all(isinstance(x, Sequence) and not isinstance(x, str | bytes) for x in min_neighbors_raw):
                min_neighbors = [tuple(x) for x in min_neighbors_raw]
                if len(min_neighbors) != field_count:
                    raise ValueError(
                        f"`min_neighbors` must have length {field_count}, "
                        f"but got {len(min_neighbors)}."
                    )
            else:
                raise TypeError(
                    f"Invalid type for `min_neighbors`; "
                    f"got {min_neighbors_raw} with type {type(min_neighbors_raw)}"
                )
        else:
            raise TypeError(
                f"Invalid type for `min_neighbors`; "
                f"got {min_neighbors_raw} with type {type(min_neighbors_raw)}"
            )

        # Process `min_members`
        if is_integer(min_members_raw):
            min_members = (min_members_raw,) * field_count
        else:
            min_members = tuple(min_members_raw)
            if len(min_members) != field_count:
                raise ValueError(
                    f"`min_members` must have length {field_count}, "
                    f"but got {len(min_members)}."
                )

        # Process `max_members`
        if is_integer(max_members_raw) or max_members_raw is None:
            max_members = (max_members_raw,) * field_count
        else:
            max_members = tuple(max_members_raw)
            if len(max_members) != field_count:
                raise ValueError(
                    f"`max_members` must have length {field_count}, "
                    f"but got {len(max_members)}."
                )

        # Validate `max_distance` and `min_neighbors` against `max_members`
        name_value_pairs = (
            ("max_distance", max_distance),
            ("min_neighbors", min_neighbors),
        )
        for idx, max_mem in enumerate(max_members):
            if max_mem is None:
                for name, value in name_value_pairs:
                    if len(value[idx]) > 1:
                        raise ValueError(
                            f"`{name}` must be a single value if `max_members` is not set, "
                            f"but got {value} for field index {idx}."
                        )
            else:
                if max_mem < min_members[idx]:
                    raise ValueError(
                        f"`max_members` ({max_mem}) must be greater than or equal to "
                        f"`min_members` ({min_members[idx]}) for field index {idx}."
                    )
                if all(len(value[idx]) == 1 for _, value in name_value_pairs):
                    raise ValueError(
                        "If `max_members` is set, at least one of `max_distance` or `min_neighbors` "
                        "must be a sequence of values, but got "
                        f"{', '.join(f'{name}={value[idx]}' for name, value in name_value_pairs)} for field index {idx}."
                    )
                elif len(max_distance[idx]) == 1:
                    max_distance[idx] = max_distance[idx] * len(min_neighbors[idx])
                elif len(min_neighbors[idx]) == 1:
                    min_neighbors[idx] = min_neighbors[idx] * len(max_distance[idx])
                elif len(max_distance[idx]) != len(min_neighbors[idx]):
                    raise ValueError(
                        "When both `max_distance` and `min_neighbors` are sequences, "
                        "they must have equal length, but got "
                        f"{len(max_distance)} vs {len(min_neighbors)} for field index {idx}."
                    )

        values["max_value"] = max_value
        values["max_distance"] = tuple(max_distance)
        values["min_neighbors"] = tuple(min_neighbors)
        values["min_members"] = min_members
        values["max_members"] = max_members
        return values
