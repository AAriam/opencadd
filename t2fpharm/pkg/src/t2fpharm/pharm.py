"""Pharmacophore."""

from typing import Sequence, Any, Self, Literal, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import scids.functional.dist
import scishow
import caddpy
import scids

from t2fpharm.system import System
from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.input import validator
from t2fpharm.input.pharm.cnn import CNNInput
from t2fpharm.input.pharm.cluster import ClusterInput
from t2fpharm.typing import DataFrameLike, PositiveFloat, PositiveInt, ClusteringFunction

import scids.functional


class Pharmacophore:
    """Pharmacophore.

    Parameters
    ----------
    features
        DataFrame-like object containing pharmacophore feature data.
        It can be a `pandas.DataFrame`, or any object that can be
        converted to a DataFrame using the `pandas.DataFrame()` constructor.
        Each row in the resulting DataFrame must represent
        a pharmacophore feature with the following columns:
        - `instance`: An identifier for the feature instance,
          e.g., for when the pharmacophore is derived
          from multiple receptors or ligands.
          If not present, a default value of 0 is added to all features.
        - `type`: A string representing the feature type,
           e.g., "hbond_donor", "hbond_acceptor", "hydrophobic", etc.
        - `label`: An identifier for different features of the same type
           within the same instance. That is, for each unique (`instance`, `type`) pair,
           each feature must have a unique label.
           Each feature in the whole pharmacophore can thus be uniquely identified
           by its (`instance`, `type`, `label`) triplet.
           If not present, it will be added with sequential integers starting from 1.
        - `center`: A sequence of three real numbers representing
           the 3D coordinates of the feature in some reference frame.
        - `radius`: A non-negative real number representing the feature radius.
           If not present, it will be added with a default value of 0.
    feature_types
        Set of all feature types that were considered when creating the pharmacophore.
        That is, all `type` values in the `features` DataFrame must be a subset of this set.
        If provided, it is used to:
        - Validate `type` values in the `features` DataFrame.
        - Ensure that query pharmacophores passed to the `match()` method
          only contain features of these types.
    inputs
        Optional dictionary containing input arguments used to create the pharmacophore.
        This is not used by this class, but can be useful for downstream processing
        and tracking how the pharmacophore was generated.
    name
        Optional name for the pharmacophore.
        This can be used to identify the pharmacophore
        in visualizations or batch analyses.
    system
        Optional chemical system associated with the pharmacophore.
        If provided, it is used by the `display()` method
        to visualize the pharmacophore in the context of the chemical structure.
    pocket
        Optional binding pocket associated with the pharmacophore.
        If provided, it is used by the `display()` method
        to visualize the pharmacophore in the context of a binding pocket.
    field
        Optional field associated with the pharmacophore.
        If provided, it is used by the `display()` method
        to visualize the pharmacophore in the context of a field.
    extra
        Optional dictionary to bundle additional information
        related to the pharmacophore, such as metadata or processing results.
        This is not used by this class, but can be useful for downstream processing.
    """

    def __init__(
        self,
        features: DataFrameLike,
        feature_types: set[str] | None = None,
        inputs: dict[str, Any] | None = None,
        name: str = "Pharmacophore",
        system: System | None = None,
        pocket: Pocket | None = None,
        field: Field | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self._features = _FeaturesInput(features=features, feature_types=feature_types).features
        self._feature_types = feature_types or set(self._features['type'].unique())
        self._inputs = inputs or {}
        self._name = name
        self._system = system
        self._pocket = pocket
        self._field = field
        self._extra = extra or {}

        # DataFrame for joining query features with target features (see `match()` method)
        self._features_for_match = self._features[
            ['instance', 'type', 'label', 'center', 'radius']
        ].rename(
            columns={
                "instance": "target_instance",
                "label": "target_label",
                "center": "target_center",
                "radius": "target_radius"
            }
        )

        # DataFrame for cross-joining query features with target instances (see `match()` method)
        self._feature_instances_for_match = pd.DataFrame(
            {'target_instance': self._features_for_match['target_instance'].unique(), "_key": 1}
        )

        self._feature_colors = {
            "HD": (0, 0.6, 0),
            "OA": (0.6, 0, 0),
            "C": (1.0, 1.0, 0),
            "e+": (0, 0, 1.0),
            "e-": (1.0, 0, 0),
        }
        return

    @property
    def features(self) -> pd.DataFrame:
        """Pharmacophore features."""
        return self._features

    @property
    def feature_types(self) -> set[str] | None:
        """Set of all considered feature types."""
        return self._feature_types

    @property
    def inputs(self) -> dict[str, Any]:
        """Inputs used to create the pharmacophore."""
        return self._inputs

    @property
    def name(self) -> str:
        """Name of the pharmacophore."""
        return self._name

    @property
    def system(self) -> System | None:
        """Chemical system associated with the pharmacophore."""
        return self._system

    @property
    def pocket(self) -> Pocket | None:
        """Binding pocket associated with the pharmacophore."""
        return self._pocket

    @property
    def field(self) -> Field | None:
        """Field associated with the pharmacophore."""
        return self._field

    @property
    def extra(self) -> dict[str, Any]:
        """Additional information related to the pharmacophore."""
        return self._extra

    def remove_overlaps(
        self,
        min_distance: PositiveFloat | dict[tuple[str, str], PositiveFloat],
        priority: pd.Series | Sequence,
        highest_priority: Literal["lowest", "highest"] = "lowest",
        max_features: dict[str, PositiveInt] | None = None,
    ) -> Self:
        """Remove overlapping features in each pharmacophore instance."""
        def make_min_spacing():
            if validator.is_real_number(min_distance):
                if not validator.is_positive_number(min_distance):
                    raise ValueError(f"Minimum distance must be positive, got {min_distance}")
                return np.full((n_types, n_types), min_distance, dtype=np.float64)
            min_spacing = np.zeros((n_types, n_types), dtype=np.float64)
            seen = set()
            for type_pair, distance in min_distance.items():
                unique_pair = tuple(sorted(type_pair))
                if unique_pair in seen:
                    raise ValueError(
                        f"Duplicate minimum distance for feature types {type_pair}: {distance}"
                    )
                seen.add(unique_pair)
                for typ in unique_pair:
                    if typ not in feature_types:
                        raise ValueError(f"Invalid feature type: {typ}. Allowed: {feature_types}")
                if distance < 0:
                    raise ValueError(
                        f"Minimum distance must be non-negative, got {distance} for types {unique_pair}"
                    )
                if len(unique_pair) != 2:
                    raise ValueError(
                        f"Minimum distance must be specified for pairs of feature types, got {unique_pair}"
                    )
                i, j = feature_types.index(unique_pair[0]), feature_types.index(unique_pair[1])
                min_spacing[i, j] = min_spacing[j, i] = distance
            return min_spacing

        def make_max_count():
            if max_features is None:
                return None
            max_counts = np.full(n_types, np.iinfo(np.int64).max, dtype=np.int64)
            for feature_type, max_count in max_features.items():
                if feature_type not in feature_types:
                    raise ValueError(f"Invalid feature type: {feature_type}. Allowed: {feature_types}")
                max_counts[feature_types.index(feature_type)] = max_count
            return max_counts

        feature_types = list(self.feature_types)
        n_types = len(feature_types)

        # Convert priority to numpy array and validate length
        priority = np.asarray(priority)
        if priority.ndim != 1 or priority.shape[0] != len(self._features):
            raise ValueError(
                f"Priority must be 1D with length {len(self._features)}, but got shape {priority.shape}"
            )
        # Validate highest_priority argument
        if highest_priority not in ("lowest", "highest"):
            raise ValueError("`highest_priority` must be either 'lowest' or 'highest'")

        # Validate feature_types covers all present types
        unique_types = set(self._features["type"])
        missing = unique_types.difference(feature_types)
        if missing:
            raise ValueError(f"Found types not in feature_types: {missing}")

        # Build mapping from type value to index
        type_to_idx = {t: i for i, t in enumerate(feature_types)}

        # Work on a copy to avoid modifying original
        df = self._features.copy()
        df["_priority"] = priority

        selected_groups: list[pd.DataFrame] = []
        ascending = highest_priority == "lowest"

        min_spacing = make_min_spacing()
        max_count = make_max_count()

        # Process each instance separately
        for _, group in df.groupby("instance", sort=False):
            # sort by priority
            sorted_grp = group.sort_values("_priority", ascending=ascending)
            # prepare arrays for ensure_spacing
            centers = np.stack(sorted_grp["center"].to_numpy())
            types_idx = sorted_grp["type"].map(type_to_idx).to_numpy(dtype=int)
            # call external spacing function
            chosen_indices = scids.functional.dist.ensure_pointcloud_spacing(
                points=centers,
                point_types=types_idx,
                min_spacing=min_spacing,
                max_count=max_count,
            )
            # select corresponding rows
            filtered = sorted_grp.iloc[chosen_indices]
            selected_groups.append(filtered)
        # concatenate results and drop helper column
        if selected_groups:
            result = pd.concat(selected_groups).drop(columns="_priority")
        else:
            result = self._features.iloc[0:0].copy()
        return Pharmacophore(
            features=result,
            feature_types=self.feature_types,
            inputs=self.inputs,
            name=self.name,
            system=self.system,
            pocket=self.pocket,
            field=self.field,
            extra=self.extra,
        )

    def cluster(
        self,
        function: ClusteringFunction | dict[str, ClusteringFunction],
        weights: pd.Series | np.ndarray | Sequence[float] | None = None,
        center_type: Literal["function", "midpoint", "mean", "average"] | dict[str, Literal["function", "midpoint", "mean", "average"]] = "average",
        radius_type: Literal["average", "mean", "max", "min"] = "average",
        per_instance: bool = True,
    ) -> Self:
        """Cluster pharmacophore features using provided clustering functions.

        Parameters
        ----------
        function
            Either a single clustering function for all feature types,
            or a dictionary mapping each feature type to a clustering function.
            Each function must accept a 2D numpy array of shape `(n_features, 3)`
            containing the coordinates of feature centers,
            and return a 1D array/sequence of cluster labels as integers
            for each feature center in the input array.
            Labels that are 0 or negative are considered background/noise
            and will not be included in the output pharmacophore.
        per_instance
            If `True`, clusters features separately for each instance.
            If `False`, clusters all features together regardless of instance.
        """
        args = ClusterInput(
            function=function,
            weights=weights,
            center_type=center_type,
            per_instance=per_instance,
            n_features=len(self.features),
            feature_types=self.feature_types
        )
        function = args.function
        center_type = args.center_type
        radius_type = args.radius_type
        per_instance = args.per_instance

        df = self._features.copy()
        df["_weights"] = args.weights

        selected: list[dict] = []
        for group_idx, group in df.groupby(
            ["instance", "type"] if per_instance else ["type"],
            sort=False
        ):
            feature_type = group_idx[-1]
            centers = np.stack(group["center"].to_numpy())
            weights = group["_weights"].to_numpy(dtype=np.float64)
            clustering_result = function[feature_type](centers, weights)
            labels = np.asarray(clustering_result.labels)
            if labels.ndim != 1:
                raise ValueError(
                    f"Clustering function for feature type '{feature_type}' must return a 1D array of labels, "
                    f"but got shape {labels.shape}."
                )
            if labels.size != centers.shape[0]:
                raise ValueError(
                    f"Clustering function for feature type '{feature_type}' must return labels "
                    f"for all feature centers, but got {labels.size} labels for {centers.shape[0]} centers."
                )
            if labels.dtype.kind != 'i':
                raise ValueError(
                    f"Clustering function for feature type '{feature_type}' must return integer labels, "
                    f"but got dtype {labels.dtype}."
                )
            unique_labels = np.unique(labels)
            mask_accepted = unique_labels > 0
            unique_positive_labels = unique_labels[mask_accepted]
            instance = group_idx[0] if per_instance else 0
            feature_center_type = center_type[feature_type]
            feature_radius_type = radius_type[feature_type]
            for label in unique_positive_labels:
                label_mask = labels == label
                point_coordinates = centers[label_mask]

                if feature_center_type == "function":
                    cluster_center = clustering_result.centers[label]
                elif feature_center_type == "midpoint":
                    cluster_center = (point_coordinates.min(axis=0) + point_coordinates.max(axis=0)) / 2
                elif feature_center_type == "mean":
                    cluster_center = np.mean(point_coordinates, axis=0)
                elif feature_center_type == "average":
                    cluster_center = np.average(point_coordinates, weights=weights[label_mask], axis=0)
                else:
                    raise ValueError(
                        f"Invalid center_type '{feature_center_type}' for feature type '{feature_type}'. "
                        "Must be one of 'function', 'midpoint', or 'average'."
                    )
                dist_to_center = np.linalg.norm(point_coordinates - cluster_center, axis=1) + group["radius"][label_mask].to_numpy(dtype=np.float64)
                if feature_radius_type == "average":
                    radius = np.average(dist_to_center, weights=weights[label_mask])
                elif feature_radius_type == "mean":
                    radius = np.mean(dist_to_center)
                elif feature_radius_type == "max":
                    radius = np.max(dist_to_center)
                elif feature_radius_type == "min":
                    radius = np.min(dist_to_center)
                else:
                    raise ValueError(
                        f"Invalid radius_type '{feature_radius_type}' for feature type '{feature_type}'. "
                        "Must be one of 'average', 'mean', 'max', or 'min'."
                    )
                selected.append(
                    {
                        "instance": instance,
                        "type": feature_type,
                        "label": label,
                        "center": cluster_center,
                        "radius": radius,
                        "members": group.index[label_mask].to_numpy()
                    }
                )
        return Pharmacophore(
            features=pd.DataFrame(selected) if selected else self.features.iloc[0:0].copy(),
            feature_types=self.feature_types,
            inputs=self.inputs,
            name=self.name,
            system=self.system,
            pocket=self.pocket,
            field=self.field,
            extra=self.extra,
        )

    def cluster_cnn(
        self,
        max_distance: PositiveFloat | Sequence[PositiveFloat] | dict[str, PositiveFloat | Sequence[PositiveFloat]],
        min_neighbors: PositiveInt | Sequence[PositiveInt] | dict[str, PositiveInt | Sequence[PositiveInt]],
        min_members: PositiveInt | dict[str, PositiveInt] = 1,
        max_members: PositiveInt | dict[str, PositiveInt] | None = None,
        weights: pd.Series | np.ndarray | Sequence[float] | None = None,
        center_type: Literal["function", "midpoint", "mean", "average"] | dict[str, Literal["function", "midpoint", "mean", "average"]] = "average",
        radius_type: Literal["average", "mean", "max", "min"] = "average",
        per_instance: bool = True,
    ) -> Self:
        """Cluster pharmacophore features using the Common Nearest Neighbors (CNN) algorithm.

        Parameters
        ----------
        max_distance
            Maximum distance between two feature centers
            to consider them as neighbors during clustering.
            - If a single number is provided,
              it applies to all feature types and all (re)clustering runs.
            - If a sequence of numbers is provided,
              the sequence is applied to all feature types,
              where the i-th number in the sequence corresponds to
              the input for the i-th clustering run
              (see the `max_members` parameter below for more details).
            - If a dictionary is provided,
              it must map each feature type in the pharmacophore
              to a single number or a sequence of numbers.
        min_neighbors
            Minimum number of common neighbors
            between two feature centers that belong to the same cluster.
            Similar to `max_distance`, this can be a single integer,
            a sequence of integers, or a dictionary.
        min_members
            Minimum number of members in a cluster.
            Cluster with fewer members than this are discarded.
            This can either be a single integer applied to all feature types,
            or a dictionary mapping each feature type in the pharmacophore
            to an integer.
        max_members
            Optional cap for the maximum number of members in a cluster.
            This can either be a single integer applied to all feature types,
            or a dictionary mapping each feature type in the pharmacophore
            to an integer.
            If specified, clusters with more members than this
            are reclustered into smaller clusters.
            For this, either one or both of `max_distance` and `min_neighbors`
            must be a sequence of values,
            where the i-th value corresponds to the i-th clustering step.
            In each step, clusters from the last step
            with more members than `max_members`
            are reclustered until all clusters
            have maximum `max_members` members.
            If all `max_distance` and `min_neighbors`
            values are exhausted without reaching the desired number of members,
            an error is raised.
            If only one of `max_distance` or `min_neighbors`
            is a sequence, the other one is assumed to be constant
            for all clustering steps.
            If both are sequences,
            they must have the same length,
            and the i-th value of `max_distance` and `min_neighbors`
            is used for the i-th clustering step.
        """
        args = CNNInput(
            feature_types=self.feature_types,
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            min_members=min_members,
            max_members=max_members,
            per_instance=per_instance
        )
        return self._cluster(clustering_function=args.clustering_function, per_instance=args.per_instance)

    def match(
        self,
        query: Self | DataFrameLike,
        max_distance: float | Literal["radius_sum"] | None = "radius_sum",
        raise_missing_types: bool = True,
    ) -> pd.DataFrame:
        """Match query pharmacophore features against the target pharmacophore.

        For each feature (i.e. unique combination of `instance`, `type`, and `label`)
        in the input query pharmacophore, this method finds the closest feature
        of the same type in each instance of the target pharmacophore (i.e. `self`), if any.
        Subsequently, for each pair, the distance between their centers
        and the sum of their radii are computed.
        If `max_distance` is specified, a `match` column is added
        indicating whether the distance is less than `max_distance`,
        which can be a fixed value or the sum of the radii.

        Parameters
        ----------
        query
            Query pharmacophore features to match against the target pharmacophore.
            This can be a `Pharmacophore` instance or any object convertible to a DataFrame
            with the same structure as the `features` DataFrame.
        max_distance
            Maximum distance between query and target features to consider a match.
            If set to "radius_sum", the maximum distance is the sum of the radii
            of the query and target features.
            If set to `None`, no `match` column is added.
        raise_missing_types
            If `True`, raises an error if the query features contain types
            not present in the target pharmacophore's feature types,
            otherwise treats them as missing.

        Returns
        -------
        DataFrame with the same rows as the query input,
        containing the following columns:
        - `instance`: Query feature instance identifier.
        - `type`: Feature type.
        - `label`: Query feature label.
        - `target_instance`: Feature instance in self.
        - `target_label`: Label of the matching feature in self.
           If no match is found, this will be `NaN`.
        - `distance`: Distance between the matched query and target feature centers.
           If no match is found, this will be `NaN`.
        - `radius_sum`: Sum of the radii of the matched query and target features.
           If no match is found, this will be `NaN`.
        - `match`: Boolean indicating if distance is less than `max_distance`.
           Only present if `max_distance` is not `None`.
        """
        query_unverified = query.features if isinstance(query, Pharmacophore) else query
        query = _FeaturesInput(
            features=query_unverified,
            feature_types=self.feature_types if raise_missing_types else None
        ).features

        # Put query row indices in a column for later reference
        query = query.reset_index().rename(columns={'index': 'query_idx'})

        # Cross-join ligand × instance
        # This creates a DataFrame with all combinations of query features and target instances
        query['_key'] = 1
        cross = query.merge(self._feature_instances_for_match, on='_key').drop(columns=['_key'])

        # Merge with target features on instance & type
        # This will add target feature data to each query × instance pair
        merged = cross.merge(
            self._features_for_match,
            on=['target_instance', 'type'],
            how='left'
        ).convert_dtypes()

        # Ensure these cols always exist, even if merged is empty
        merged['distance'] = np.nan
        merged['radius_sum'] = np.nan
        if max_distance is not None:
            merged['match'] = False

        # Compute distances where feature exists
        mask = merged['target_label'].notna()
        if mask.any():
            query_centers = np.stack(merged.loc[mask, 'center'].values)
            target_centers = np.stack(merged.loc[mask, 'target_center'].values)
            merged.loc[mask, 'distance'] = np.linalg.norm(query_centers - target_centers, axis=1)
            merged.loc[mask, "radius_sum"] = merged.loc[mask, 'radius'] + merged.loc[mask, 'target_radius']
            if max_distance is not None:
                merged.loc[mask, 'match'] = merged.loc[mask, 'distance'] < (
                    merged.loc[mask, 'radius_sum']
                    if max_distance == "radius_sum" else
                    max_distance
                )

        # Defaults for missing-feature cases
        merged['distance'] = merged['distance'].astype(float)

        # Pick minimum-distance feature per query_idx×instance
        # Treat NaN distances as +inf so real distances sort first
        merged['dist_sort'] = merged['distance'].fillna(np.inf)
        best = (
            merged
            .sort_values(['query_idx', 'target_instance', 'dist_sort'])
            .drop_duplicates(['query_idx', 'target_instance'], keep='first')
        )

        # Reorder & return
        final_cols = [
            'instance',
            'type',
            'label',
            'target_instance',
            'target_label',
            'distance',
            'radius_sum'
        ]
        if max_distance is not None:
            final_cols.append('match')
        return best[final_cols].reset_index(drop=True)

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        system: Any | None = None,
        default_radius: float = 1.5,
        show_box: bool = True,
        show_pocket: bool = True,
        show_fields: bool = False,
        show_feature_centers: bool = True,
        show_feature_points: bool = False,
        feature_colors: dict[str, tuple[float, float, float] | tuple[int, int, int]] | None = None,
    ):
        def feature_color(feature_id: str) -> tuple[float, float, float] | tuple[int, int, int]:
            """Get color for a feature type, defaulting to gray if not set."""
            return feature_colors.get(
                feature_id,
                self._feature_colors.get(feature_id, (0.5, 0.5, 0.5))
            )

        nv = nglwidget or scishow.nglview.NGLWidget()
        feature_colors = feature_colors or {}

        # System
        if system is not None:
            nv.add_trajectory(system)
        elif self.system is not None:
            nv.add_trajectory(self.system)

        # Pocket
        if self.pocket is not None:
            self.pocket.display(
                nglwidget=nv,
                show_box=show_box,
                visible=show_pocket,
                receptor=False,
            )

        # Field
        if self.field is not None:
            for feature_type in self.field.batch_instance_labels["feature"]:
                nv.add_volume(
                    data=self.field(feature=feature_type),
                    basis=self.field.grid.unit_vectors,
                    origin=self.field.grid.lower_bounds,
                    name=f"{feature_type} Field",
                    representation_params=scishow.nglview.SurfaceRepresentationParameters(
                        isolevel=0,
                        isolevel_type="value",
                        contour=False,
                        wireframe=True,
                        color=feature_color(feature_type),
                        visible=show_fields,
                    )
                )

        # Features
        for _, feature in self.features.iterrows():
            name = f"{feature['instance']}_{feature['type']}_{feature['label']}"
            nv.add_spheres(
                coords=feature["center"],
                radii=feature["radius"] or default_radius,
                name=f"{name} Center",
                colors=feature_color(feature["type"]),
                representation_params=scishow.nglview.RepresentationParameters(
                    opacity=0.8,
                    visible=show_feature_centers,
                    lazy=True,
                )
            )
            if "points" in feature:
                nv.add_spheres(
                    coords=feature["points"],
                    radii=self.field.grid.spacings[0] / 2,
                    name=f"{name} Points",
                    colors=feature_color(feature["type"]),
                    representation_params=scishow.nglview.RepresentationParameters(
                        visible=show_feature_points,
                    )
                )
        return nv.display(gui=True)

    def set_feature_color(self, **kwargs: tuple[int, int, int] | tuple[float, float, float]) -> None:
        """Set custom colors for pharmacophore features.

        Parameters
        ----------
        **kwargs
            Feature types as keys and RGB color tuples as values.
            Each color can be a tuple of three integers (0-255) or floats (0.0-1.0).
            Example: `set_feature_color(HD=(0, 255, 0), OA=(255, 0, 0))`
        """
        for feature, color in kwargs.items():
            if self.feature_types is not None and feature not in self.feature_types:
                raise ValueError(f"Invalid feature type: {feature}. Allowed: {self.feature_types}")
            if isinstance(color, Sequence) and len(color) == 3:
                if all(isinstance(c, (int, float)) for c in color):
                    self._feature_colors[feature] = tuple(color)
                else:
                    raise ValueError(f"Invalid color format for feature '{feature}': {color}")
            else:
                raise ValueError(f"Color must be a tuple of three values for feature '{feature}'")
        return


def from_complex(
    pdb_files: str | bytes | Path | Sequence,
    ligands: Sequence[tuple[str, int | str, int]] | None = None,
    type_hbond_donor: str = "HD",
    type_hbond_acceptor: str = "OA",
    type_anion: str = "e-",
    type_cation: str = "e+",
    type_hydrophobic: str = "C",
    pocket: Pocket | None = None,
    receptor: System | None = None,
):
    plip = caddpy.interaction.from_pdb(pdb_files, ligands=ligands)
    out = []
    for _, row in plip.all.iterrows():
        position_col = "l_position"
        match row["type"]:
            case "hbond":
                if row["r_is_d"]:
                    feature_type = type_hbond_acceptor
                else:
                    feature_type = type_hbond_donor
                    position_col = "h_position"
            case "water_bridge":
                position_col = "w_position"
                feature_type = type_hbond_acceptor if row["r_is_d"] else type_hbond_donor
            case "salt_bridge":
                feature_type = type_anion if row["r_is_cation"] else type_cation
            case "hydrophobic":
                feature_type = type_hydrophobic
            case _:
                continue
        position = row[position_col]
        # Sometimes a single atom can be involved
        # in multiple interactions of the same type, e.g., hydrophobic interactions with different residues.
        # Therefore, we only add a new feature if not already present
        for entry in out:
            if entry["type"] == feature_type and np.allclose(entry["center"], position):
                break
        else:
            out.append({"type": feature_type, "center": position})
    if pocket is not None:
        positions = np.stack([feature["center"] for feature in out])
        coverages = pocket.point_coverage(positions)
        out = [feature for feature, coverage in zip(out, coverages) if coverage]
    return Pharmacophore(features=out, extra={"plip": plip}, system=receptor)


class _FeaturesInput(BaseModel):
    """Model to validate and normalize the `features` input argument.

    This model accepts any input convertible to a pandas DataFrame and ensures:
    - Columns 'type' and 'center' are present.
    - 'type' values are strings.
    - 'center' entries are 1D numpy arrays of three floats.
    - A non-negative 'radius' column is present (added with zeros if missing).
    - 'instance' column is added with default value 0 if missing.
    - 'label' column is added with sequential integers starting from 1 if missing.

    Attributes
    ----------
    features
        Normalized DataFrame with columns
        'instance', 'type', 'label', 'center', 'radius',
        and any additional columns from the input.
    """
    features: pd.DataFrame
    feature_types: set[str] | None = None

    # Allow arbitrary types like pandas DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('features', mode='before')
    def ensure_dataframe(cls, v: Any) -> pd.DataFrame:
        """Convert input to a pandas DataFrame if it isn't already."""
        if isinstance(v, pd.DataFrame):
            return v.copy().convert_dtypes()
        try:
            return pd.DataFrame(v).convert_dtypes()
        except Exception as e:
            raise ValueError(f"Cannot convert input to DataFrame.") from e

    @model_validator(mode='after')
    def validate_and_normalize(self) -> Self:
        """Validate and normalize the features DataFrame."""
        def to_array(val: Any) -> np.ndarray:
            """Convert position to a 1D numpy array of three floats."""
            arr = np.asarray(val, dtype=float)
            if arr.shape != (3,):
                raise ValueError(
                    f"Position must be sequence of 3 numbers, got shape {arr.shape}"
                )
            return arr

        df = self.features

        # Check required columns
        required_cols = {'type', 'center'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate 'instance'
        if "instance" not in df.columns:
            df["instance"] = 0

        # Validate 'type'
        if not pd.api.types.is_string_dtype(df['type']):
            raise ValueError("Feature column 'type' must be strings")
        if self.feature_types is not None:
            invalid_types = set(df['type']) - self.feature_types
            if invalid_types:
                raise ValueError(
                    f"Invalid feature types found: {sorted(invalid_types)}. "
                    f"Allowed: {sorted(self.feature_types)}"
                )

        # Validate 'label'
        if "label" in df.columns:
            # Compute group sizes vs. unique label counts
            grp = df.groupby(["instance", "type"])["label"]
            sizes = grp.size()
            unique_counts = grp.nunique()
            bad = sizes[unique_counts != sizes]
            if not bad.empty:
                bad_groups = ", ".join(f"{inst},{typ}" for inst, typ in bad.index)
                raise ValueError(f"Duplicate labels found in groups: {bad_groups}")
        else:
            # Add labels sequentially within each (instance, type) group
            df["label"] = df.groupby(["instance", "type"]).cumcount() + 1

        # Validate and normalize 'center'
        df['center'] = df['center'].apply(to_array)

        # Handle 'radius' column
        if 'radius' in df.columns:
            try:
                df['radius'] = df['radius'].astype(float)
            except Exception:
                raise ValueError("Radius column must be real numbers")
            neg_idx = df.index[df['radius'] < 0].tolist()
            if neg_idx:
                raise ValueError(f"Negative radius at rows: {neg_idx}")
        else:
            df['radius'] = 0.0

        main_cols = ['instance', 'type', 'label', 'center', 'radius']
        extra_cols = [col for col in df.columns if col not in main_cols]
        all_cols = main_cols + extra_cols
        self.features = df[all_cols].convert_dtypes()
        return self


def _center_midpoint(points: np.ndarray) -> np.ndarray:
    """Compute the midpoint of a set of points."""
    return np.mean(points, axis=0)