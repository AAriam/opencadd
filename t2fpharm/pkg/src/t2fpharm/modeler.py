
from typing import Sequence

import pandas as pd
import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel, Field, model_validator

import scids

from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.pharmacophore_receptor import ReceptorPharmacophore
from t2fpharm.typing import PositiveInt, PositiveIntTuple, PositiveFloatTuple, is_real_number, is_integer


class Modeler:
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

    def cnn(
        self,
        max_value: float | Sequence[float] = (-0.35, -0.4, -0.6, -1, -1),
        max_distance: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
        min_neighbors: int | Sequence[int] | Sequence[Sequence[int]] = tuple(range(6, 100, 4)),
        min_members: int | Sequence[int] | None = None,
        max_members: int | Sequence[int | None] | None = None,
    ):
        """Perceive pharmacophore features using the CNN algorithm.

        This method first selects points from the field tensor
        that are within the pocket and below a specified `max_value`,
        then clusters these points using the Common Nearest Neighbor (CNN) algorithm
        with the specified parameters.

        Parameters
        ----------
        max_value
            Maximum value for feature types in the field tensor.
            - If a single number is provided, it applies to all feature types.
            - If a sequence is provided, it must match
              the order and number of feature types in the field.
        max_distance
            Maximum distance between two points to consider them as neighbors during clustering.
            - If a single number is provided, it applies to all feature types and all (re)clustering runs.
            - If a sequence of numbers is provided, the sequence is applied to all feature types,
              where the i-th number in the sequence corresponds to the input for the i-th clustering run
              (see the `max_members` parameter below for more details).
            - If a sequence of sequences is provided, the outer sequence must match
              the order and number of feature types in the field.
            - If `None`, defaults to 2.1 times the grid spacing of the field
              for all feature types and clustering runs.
              This ensures that for each grid point, all 26 first neighbors
              plus 6 orthogonal second neighbors are included in the clustering.
        min_neighbors
            Minimum number of common neighbors between two points
            that belong to the same cluster.
            Similar to `max_distance`, this can be a single integer,
            a sequence of integers, or a sequence of sequences.
        min_members
            Minimum number of members in a cluster.
            Cluster with fewer members than this are discarded.
            - If a single integer is provided, it applies to all feature types.
            - If a sequence of integers is provided, it must match
              the order and number of feature types in the field.
            - If `None`, defaults to the number of grid points with the same
              volume as half the van der Waals volume of a hydrogen atom.
            - To disable this filtering, set `min_members` to 1
              for all or selected feature types.
        max_members
            Maximum number of members in a cluster.
            If specified, clusters with more members than this
            are reclustered into smaller clusters.
            For this, either one or both of `max_distance` and `min_neighbors`
            must be a sequence (or sequence of sequences) of values,
            where the i-th value corresponds to the i-th clustering step.
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
            - If a single integer is provided,
              it applies to all feature types and clustering runs.
            - If a sequence of integers is provided,
              it must match the order and number of feature types in the field.
            - If `None`, defaults to 5 times the `min_members` value
              for each feature type.
        """
        if max_distance is None:
            # As default, include all 26 neighbors in a 3D grid
            # plus orthogonal second neighbors (i.e., 26 + 6 = 32 neighbors)
            max_distance = self.field.grid.spacings[0] * 2.1
        if min_members is None:
            hydrogen_radius = 1.2
            hydrogen_volume = (4/3) * np.pi * hydrogen_radius**3
            half_hydrogen_volume = hydrogen_volume / 2
            voxel_volume = self.field.grid.point_volume
            min_members = int(np.ceil(half_hydrogen_volume / voxel_volume))
        if max_members is None:
            max_members = min_members * 5 if isinstance(min_members, int) else [
                min_member * 5 for min_member in min_members
            ]

        args = _CNNArgs(
            field_count=self.field.tensor.shape[0],
            max_value=max_value,
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            min_members=min_members,
            max_members=max_members,
        )
        field_masks = jnp.less_equal(
            self.field.tensor,
            jnp.array(args.max_value).reshape(-1, *(1,) * (self.field.tensor.ndim - 1)),
        )
        final_masks = jnp.logical_and(
            field_masks,
            self.pocket.tensor
        )
        features = []
        for idx in np.ndindex(tuple(self.field.batch_shape)):
            field_idx = idx[0]
            field_id = self.field.batch_instance_labels["feature"][field_idx]
            mask = final_masks[idx]
            points = self.field.grid.coordinates[mask]
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
                volume = n_points * self.field.grid.point_volume
                point_coordinates = points[labels == cluster_label]
                features.append(
                    {
                        "instance": idx[1:] or 0,
                        "type": field_id,
                        "n_points": n_points,
                        "volume": volume,
                        "radius": (volume / ((4/3) * np.pi)) ** (1/3),
                        "center": point_coordinates.mean(axis=0),
                        "label": cluster_label,
                        "points": point_coordinates,
                    }
                )
        return ReceptorPharmacophore(
            features=pd.DataFrame(features),
            pocket=self.pocket,
            field=self.field,
            args=args,
        )


class _CNNArgs(BaseModel):
    method: str = "cnn"
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
