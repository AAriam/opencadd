"""Pharmacophore."""

from typing import Sequence, Any, Self, Literal
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.spatial

import scids.functional.dist
import scishow
import scids

from t2fpharm.system import System
from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.input.pharm.cluster import PharmClusterInput, ClusteringFunction, CenterType, CenterTypeNoFunction, RadiusType
from t2fpharm.input.pharm.cluster_agg import PharmClusterAggInput, AggLinkageType, AggLinkageMetricType
from t2fpharm.input.pharm.cluster_cnn import PharmClusterCNNInput
from t2fpharm.input.pharm.features import PharmFeaturesInput
from t2fpharm.input.pharm.remove_overlaps import RemoveOverlapsInput
from t2fpharm.typing import DataFrameLike, PositiveFloat, PositiveInt

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
           The radius defines the uncertainty of the feature center.
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
        If provided, it is only used by the `display()` method
        to visualize the pharmacophore in the context of the chemical structure.
        This can be any object that can be visualized by NGLView
        using its `add_trajectory()` method.
    pocket
        Optional binding pocket associated with the pharmacophore.
        If provided, it is only used by the `display()` method
        to visualize the pharmacophore in the context of its binding pocket.
    field
        Optional field associated with the pharmacophore.
        If provided, it is only used by the `display()` method
        to visualize the pharmacophore in the context of its fields.
    extra
        Optional dictionary to bundle additional information
        related to the pharmacophore, such as metadata or processing results.
        This is not used by this class, but can be useful for downstream processing.
    """

    def __init__(
        self,
        features: DataFrameLike,
        feature_types: set[str] | None = None,
        inputs: Sequence[dict[str, Any]] | None = None,
        name: str = "Pharmacophore",
        system: System | None = None,
        pocket: Pocket | None = None,
        field: Field | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self._features = PharmFeaturesInput(features=features, feature_types=feature_types).features
        if self._features.empty or (self._features["instance"] == 0).all():
            self._batch_shape = np.array([], dtype=np.int64)
        elif self._features["instance"].dtype != "object":
            self._batch_shape = np.array([self._features["instance"].max() + 1], dtype=np.int64)
        else:
            self._batch_shape = np.vstack(self._features["instance"].to_numpy()).max(axis=0) + 1

        if pocket is not None:
            self._check_instance_consistency_with_pocket(pocket)

        if feature_types is None:
            self._feature_types = set(self._features['type'].unique())
        else:
            feature_types = set(feature_types)
            # Validate feature_types covers all present types
            unique_types = set(self._features["type"])
            missing = unique_types.difference(feature_types)
            if missing:
                raise ValueError(f"Found types not in feature_types: {missing}")
            self._feature_types = feature_types

        self._inputs = list(inputs) if inputs is not None else []
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

        self._features_per_instance = self._features.groupby('instance')

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

        if self._field is None:
            self._field_in_pocket = None
        else:
            self._field_in_pocket = self._field.new(
                tensor=np.where(self._pocket.tensor, self._field.tensor, 0),
            ) if self._pocket is not None else self._field
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
    def inputs(self) -> list[dict[str, Any]]:
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

    def filter(
        self,
        mask: pd.Series | np.ndarray | Sequence[bool],
        name: str | None = None,
    ) -> Self:
        """Filter pharmacophore features using a boolean mask.

        Parameters
        ----------
        mask
            Boolean mask to filter features.
            It can be a `pandas.Series`, a numpy array,
            or any sequence of boolean values.
            It must have the same length and order as `self.features`.

        Returns
        -------
        A new `Pharmacophore` instance containing only the features
        where the corresponding value in `mask` is `True`.
        """
        return self.new(
            features=self._features[mask],
            inputs=self.inputs + [{"action": "filter", "params": {"mask": mask}}],
            name=name,
        )

    def refine_centers(
        self,
        by_field: bool = False,
        by_pocket: bool = False,
        field_extrema_type: Literal["min", "max"] = "min",
        field_search_radius: float = 1.5,
        max_pocket_distance: float = 0.5,
    ):
        if by_field and self._field is None:
            raise ValueError("Cannot refine by field: No field associated with the pharmacophore.")
        if by_pocket and self._pocket is None:
            raise ValueError("Cannot refine by pocket: No pocket associated with the pharmacophore.")

        feats = self._features.copy()
        feats["original_center"] = feats["center"]

        if by_field:
            feats = self._refine_by_field(
                feats=feats,
                field_search_radius=field_search_radius,
                field_extrema_type=field_extrema_type,
            )
        if by_pocket:
            feats = self._refine_by_pocket(
                feats=feats,
                max_pocket_distance=max_pocket_distance,
            )
        return self.new(features=feats.drop(columns=["original_center"]))

    def _refine_by_field(
        self,
        feats: pd.DataFrame,
        field_search_radius: float = 1.5,
        field_extrema_type: Literal["min", "max"] = "min",
    ) -> pd.DataFrame:
        field = self._field
        old_centers = np.stack(feats["center"])
        grid_indices, _, is_inside = field.grid.nearest_point(old_centers)
        feats = feats.loc[is_inside]
        grid_indices = grid_indices[is_inside]
        field_prefix_indices = feats["type"].map(
            {val: idx for idx, val in enumerate(field.batch_instance_labels["feature"])}
        ).to_numpy().reshape(-1, 1)
        if self._batch_shape.size > 0:
            # Merge instance indices with grid indices to get full field indices
            instances = feats["instance"]
            N = len(feats)
            if instances.dtype != "object":
                field_prefix_indices = np.concatenate(
                    [field_prefix_indices, instances.to_numpy().reshape(-1, 1)],
                    axis=1
                )
            else:
                vals = instances.tolist()
                K = len(vals[0]) + 1
                # Stack rows into a 2D array of shape (N, K)
                # Using a single allocation via np.empty + fill for speed/robustness
                prefix = np.empty((N, K), dtype=np.int64)
                for i, (feature_prefix, instance_prefix) in enumerate(zip(field_prefix_indices, vals)):
                    prefix[i, 0] = feature_prefix
                    prefix[i, 1:] = instance_prefix
                field_prefix_indices = prefix
        field_indices = np.concatenate([field_prefix_indices, grid_indices], axis=1)
        footprint = field.grid.footprint_spherical(field_search_radius)
        extrema_indices = _extrema_under_footprint(
            field=field.tensor,
            field_indices=field_indices,
            footprint=footprint,
            maximize=(field_extrema_type == "max"),
        )
        extrema_coords = field.grid.index_coordinates(extrema_indices[..., -3:])
        extrema_values = field.tensor[tuple(extrema_indices.T)]
        feats["radius"] = np.linalg.norm(extrema_coords - old_centers[is_inside], axis=-1)
        feats["center"] = list(extrema_coords)
        feats["value"] = extrema_values
        return feats

    def _refine_by_pocket(
        self,
        feats: pd.DataFrame,
        max_pocket_distance: float
    ) -> pd.DataFrame:
        pocket = self._pocket
        indices, distances = pocket.nearest_point(np.stack(feats["center"]))
        if self._batch_shape.size > 0:
            instances = feats["instance"]
            N = len(feats)
            if instances.dtype != "object":
                prefix = np.empty((N, 2), dtype=np.int64)
                prefix[:, 0] = instances.to_numpy()
                prefix[:, 1] = np.arange(N, dtype=np.int64)
            else:
                vals = instances.tolist()
                batch_n_dim = len(vals[0])
                K = batch_n_dim + 1
                # Stack rows into a 2D array of shape (N, K)
                # Using a single allocation via np.empty + fill for speed/robustness
                prefix = np.empty((N, K), dtype=np.int64)
                for i, (instance_prefix) in enumerate(vals):
                    prefix[i, 0:batch_n_dim] = instance_prefix
                    prefix[i, batch_n_dim] = i
            prefix_unpacked = tuple(prefix.T)
            indices = indices[prefix_unpacked]
            distances = distances[prefix_unpacked]
        dist_mask = distances <= max_pocket_distance
        indices = indices[dist_mask]
        feats = feats.loc[dist_mask]
        in_pocket_coords = pocket.grid.index_coordinates(indices)
        dists_to_orig_center = np.linalg.norm(in_pocket_coords - np.stack(feats["original_center"]), axis=-1)
        feats["radius"] = np.maximum(feats["radius"], dists_to_orig_center)
        return feats

    def remove_overlaps(
        self,
        min_distance: PositiveFloat | dict[tuple[str, str], PositiveFloat],
        priority: pd.Series | Sequence | str = "value",
        highest_priority: Literal["lowest", "highest"] = "lowest",
        max_features: PositiveInt | dict[str, PositiveInt] | None = None,
    ) -> Self:
        """Remove overlapping features in each pharmacophore instance.

        Parameters
        ----------
        min_distance
            Minimum required distance between feature centers.
            This can be a single value for all feature types,
            or a dictionary mapping feature type pairs
            to their corresponding minimum distance.
            That is, for n feature types, the dictionary can contain
            at most n(n + 1) / 2 entries. For example, if there are
            three feature types "A", "B", and "C",
            the dictionary can contain the following keys:
            ("A", "A"), ("A", "B"), ("A", "C"),
            ("B", "B"), ("B", "C"),
            ("C", "C").
            This allows for specifying different minimum distances
            for different pairs of feature types.
            Pairs that are not specified are assumed
            to have no minimum distance constraint.
        priority
            Priority of each feature in the pharmacophore.
            This can be either
            - A 1D array-like object with the same length
               and order as features in `self.features`.
            - A column name in the `self.features` DataFrame.
        highest_priority
            How to interpret the `priority` values:
            - "lowest": The lowest value is the highest priority.
            - "highest": The highest value is the highest priority.
        max_features
            Maximum number of features to keep.
            This can be a single value for all feature types
            (i.e., to keep at most `max_features` features per each type),
            or a dictionary mapping each feature type to its maximum number of features.

        Returns
        -------
        A new pharmacophore instance containing only the non-overlapping subset of features.

        Notes
        -----
        The algorithm works as follows:
        For each pharmacophore instance,
        1. All features (of any type) are sorted by their `priority` from best to worst.
        2. The first best feature is selected,
           and all other features within the corresponding `min_distance` are discarded.
        3. The next best remaining feature is selected,
           and all other features within the corresponding `min_distance` are discarded.
           This process is repeated until all features are processed.
           At any point, if the number of selected features for a feature type
           reaches `max_features`, all remaining features of that type
           are discarded for that instance.
        """
        if isinstance(priority, str):
            if priority not in self._features.columns:
                raise ValueError(f"Priority column '{priority}' not found in features DataFrame.")
            priority = self._features[priority]
        args = RemoveOverlapsInput(
            min_distance=min_distance,
            priority=priority,
            highest_priority=highest_priority,
            max_features=max_features,
            n_features=len(self.features),
            feature_types=self.feature_types
        )

        feature_types = list(self.feature_types)
        # Validate feature_types covers all present types
        unique_types = set(self._features["type"])
        missing = unique_types.difference(feature_types)
        if missing:
            raise ValueError(f"Found types not in feature_types: {missing}")

        # Build mapping from type value to index
        type_to_idx = {t: i for i, t in enumerate(feature_types)}

        # Work on a copy to avoid modifying original
        df = self._features.copy()
        df["_priority"] = args.priority

        selected_groups: list[pd.DataFrame] = []

        # Process each instance separately
        for _, group in df.groupby("instance", sort=False):
            # Sort by priority
            sorted_grp = group.sort_values(
                "_priority",
                ascending=args.highest_priority == "lowest",
                kind="stable",
            )
            # Get selected indices
            chosen_indices = scids.functional.dist.ensure_pointcloud_spacing(
                points=np.stack(sorted_grp["center"].to_numpy()),
                point_types=sorted_grp["type"].map(type_to_idx).to_numpy(dtype=int),
                min_spacing=args.min_distance_asarray,
                max_count=args.max_features_asarray,
            )
            # Select corresponding rows
            filtered = sorted_grp.iloc[chosen_indices]
            selected_groups.append(filtered)
        # Concatenate results and drop helper column
        if selected_groups:
            result = pd.concat(selected_groups).drop(columns="_priority")
        else:
            result = self._features.iloc[0:0].copy()
        return Pharmacophore(
            features=result,
            feature_types=self.feature_types,
            inputs=self.inputs + [args.model_dump()],
            name=self.name,
            system=self.system,
            pocket=self.pocket,
            field=self.field,
            extra=self.extra,
        )

    def cluster(
        self,
        function: ClusteringFunction | dict[str, ClusteringFunction],
        min_members: PositiveInt | dict[str, PositiveInt] = 1,
        noise_as_singleton: bool | dict[str, bool] = True,
        weights: pd.Series | np.ndarray | Sequence[float] | None = None,
        center_type: CenterType | dict[str, CenterType] = "average",
        radius_type: RadiusType | dict[str, RadiusType] = "average",
        per_instance: bool = True,
        preserve_members: bool = True,
    ) -> Self:
        """Cluster pharmacophore features using provided clustering functions.

        The clustering is performed on center coordinates of each feature type;
        it can be used in one of two ways:
        - **Per instance**: To cluster features of the same type separately for each instance.
          This reduces the number of features per type in each instance.
        - **Across all instances**: To cluster features of the same type across all instances.
          This reduces all pharmacophore instances into one.
          Note that if the pharmacophore has only one instance,
          this is equivalent to the per-instance case.

        The method also supports different ways
        to compute the center and radius of each cluster,
        allowing flexibility in how clusters are represented.

        Other than the `weights` and `per_instance` arguments,
        all other arguments can be a single value for all feature types,
        or a dictionary mapping each feature type to a value.

        Parameters
        ----------
        function
            Clustering function(s) to use.
            Each function is called with two positional arguments:
            1. A 2D numpy array of shape `(n_features, 3)`
               containing the coordinates of feature centers.
            2. A 1D numpy array of shape `(n_features,)`
               containing the weights for each feature center
               (see the `weights` parameter below).
               Note that while not all clustering algorithms accept/require weights,
               this parameter is always passed to the function.

            The function must return an object with a `labels` attribute,
            which must be a 1D array/sequence of cluster labels as integers
            for each feature center in the input array.
            Negative labels are considered background/noise
            and will not be included in the output pharmacophore,
            unless `noise_as_singleton` is set to `True`.
        min_members
            Minimum number of members required to form a cluster.
            Clusters with fewer members are discarded.
        noise_as_singleton
            If `True`, noise features (i.e., those with negative labels)
            are each assigned a unique label and treated as a separate cluster.
            This only applies if `min_members` is set to 1 for the feature type.
        weights
            Optional weights for each feature center.
            If provided, it must be a 1D array-like object
            with the same length and order as `self.features`.
            If not provided, a default weight of 1.0 is used for all features centers.
            Weights are passed to the clustering function,
            and can also be used to compute center and radius of each cluster.
            For calculating the center and radius,
            if the weights sum to zero for a cluster,
            the mean is used instead of the weighted average.
        center_type
            How to compute the center of each cluster:
            - "function": Use the clustering result's `centers` attribute.
            - "midpoint": Use the midpoint (i.e., bounding box center) of feature centers in the cluster.
            - "mean": Use the mean of the feature centers in the cluster.
            - "average": Use the weighted average of the feature centers in the cluster,
              where weights are taken from the `weights` parameter.
        radius_type
            How to compute the radius of each cluster:
            - "average": Use the weighted average distance from the cluster center
              to the feature centers in the cluster,
              where weights are taken from the `weights` parameter.
            - "mean": Use the mean distance from the cluster center
              to the feature centers in the cluster.
            - "max": Use the maximum distance from the cluster center
              to the feature centers in the cluster.
            - "min": Use the minimum distance from the cluster center
              to the feature centers in the cluster.
        per_instance
            `True`: Cluster features separately for each instance.
            `False`: Cluster all features together regardless of instance.

        Returns
        -------
        A new `Pharmacophore` object with clustered features.
        The `Pharmacophore.features` DataFrame will contain the following columns:
        - `instance`: Instance identifier (0 if `per_instance` is `False`).
        - `type`: Feature type.
        - `label`: Cluster label (taken from the clustering result).
        - `center`: Center coordinates of the cluster according to `center_type`.
        - `radius`: Radius of the cluster according to `radius_type`.
        - `value`: Value associated with the cluster center.
          This is the value of the cluster member with the shortest distance to the cluster center.
        - `n_members`: Number of features in the cluster.
        - `members`: Row indices of the original features that belong to this cluster.
        - `center_<center_type>`: Center coordinates of the cluster
          according to each available `center_type`.
        - `radius_<center_type>_<radius_type>`: Radius of the cluster
          according to each available `center_type` and `radius_type`.
        - `value_<center_type>`: Value associated with the cluster center
          for each available `center_type`.
        """
        args = PharmClusterInput(
            function=function,
            min_members=min_members,
            noise_as_singleton=noise_as_singleton,
            weights=weights,
            center_type=center_type,
            radius_type=radius_type,
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
        has_old_members = "members" in df.columns
        new_features: list[dict] = []
        for group_idx, group in df.groupby(
            ["instance", "type"] if per_instance else ["type"],
            sort=False
        ):
            feature_type = group_idx[-1]
            centers = np.stack(group["center"].to_numpy())
            weights = group["_weights"].to_numpy(dtype=np.float64)
            if centers.shape[0] > 1:
                clustering_result = function[feature_type](centers, weights)
                labels = np.asarray(clustering_result.labels)
            else:
                # If only one center, assign it to a single cluster
                clustering_result = None
                labels = np.array([0], dtype=np.int32)
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
            if args.noise_as_singleton[feature_type]:
                is_noise = labels < 0
                new_label_start = labels.max() + 1
                new_label_end = new_label_start + is_noise.sum()
                labels[is_noise] = np.arange(new_label_start, new_label_end)
            unique_labels_and_noise = np.unique(labels)
            unique_labels = unique_labels_and_noise[unique_labels_and_noise >= 0]
            instance = group_idx[0] if per_instance else 0
            feature_center_type = center_type[feature_type]
            feature_radius_type = radius_type[feature_type]
            for label in unique_labels:
                label_mask = labels == label
                if label_mask.sum() < args.min_members[feature_type]:
                    # Skip clusters with fewer members than required
                    continue
                cluster_points = centers[label_mask]
                cluster_weight = weights[label_mask]
                weights_sum_to_zero = np.sum(cluster_weight) == 0

                # Calculate cluster center
                cluster_centers = {
                    "center_midpoint": (cluster_points.min(axis=0) + cluster_points.max(axis=0)) / 2,
                    "center_mean": np.mean(cluster_points, axis=0),
                }
                cluster_centers["center_average"] = (
                    cluster_centers["center_mean"]
                    if weights_sum_to_zero else
                    np.average(cluster_points, weights=cluster_weight, axis=0)
                )
                if clustering_result is not None and clustering_result.centers is not None:
                    cluster_centers["center_function"] = clustering_result.centers[label]

                # Calculate center values
                values = {}
                if "value" in group.columns:
                    for center_key, center_value in cluster_centers.items():
                        dist_to_points = np.linalg.norm(centers - center_value, axis=1)
                        shortest_dist_idx = np.argmin(dist_to_points)
                        values[f"value_{center_key.removeprefix('center_')}"] = group["value"].iloc[shortest_dist_idx]

                # Calculate cluster radius
                cluster_radii = {}
                member_radii = group["radius"][label_mask].to_numpy(dtype=np.float64)
                for center_key, center_value in cluster_centers.items():
                    center_key = center_key.removeprefix("center_")
                    dist_to_center = np.linalg.norm(cluster_points - center_value, axis=1) + member_radii
                    cluster_radii |= {
                        f"radius_{center_key}_mean": np.mean(dist_to_center),
                        f"radius_{center_key}_max": np.max(dist_to_center),
                        f"radius_{center_key}_min": np.min(dist_to_center)
                    }
                    cluster_radii[f"radius_{center_key}_average"] = (
                        cluster_radii[f"radius_{center_key}_mean"]
                        if weights_sum_to_zero else
                        np.average(dist_to_center, weights=cluster_weight)
                    )
                if has_old_members and preserve_members:
                    member_indices = group["members"][label_mask].explode().to_numpy()
                else:
                    member_indices = group.index[label_mask].to_numpy()
                new_features.append(
                    {
                        "instance": instance,
                        "type": feature_type,
                        "label": label,
                        "center": cluster_centers[f"center_{feature_center_type}"],
                        "radius": cluster_radii[f"radius_{feature_center_type}_{feature_radius_type}"],
                        "value": values.get(f"value_{feature_center_type}"),
                        "n_members": len(member_indices),
                        "members": member_indices,
                        **cluster_centers,
                        **values,
                        **cluster_radii,
                    }
                )
        return Pharmacophore(
            features=pd.DataFrame(
                new_features,
                columns=[
                    "instance",
                    "type",
                    "label",
                    "center",
                    "radius",
                    "value",
                    "n_members",
                    "members",
                    "center_average",
                    "center_mean",
                    "center_midpoint",
                    "radius_average_max",
                    "radius_average_mean",
                    "radius_average_min",
                    "radius_mean_max",
                    "radius_mean_mean",
                    "radius_mean_min",
                    "radius_midpoint_max",
                    "radius_midpoint_mean",
                    "radius_midpoint_min",
                    "value_average",
                    "value_mean",
                    "value_midpoint",
                ]
            ).dropna(axis=1, how="all"),
            feature_types=self.feature_types,
            inputs=self.inputs + [args.model_dump()],
            name=self.name,
            system=self.system,
            pocket=self.pocket,
            field=self.field,
            extra=self.extra,
        )

    def cluster_agg(
        self,
        distance_threshold: PositiveFloat | dict[str, PositiveFloat] | None = None,
        n_clusters: PositiveInt | dict[str, PositiveInt] | None = None,
        linkage: AggLinkageType | dict[str, AggLinkageType] = "complete",
        metric: AggLinkageMetricType | dict[str, AggLinkageMetricType] = "euclidean",
        memory: Any = None,
        # Parameters for `self.cluster()`
        min_members: PositiveInt | dict[str, PositiveInt] = 1,
        noise_as_singleton: bool | dict[str, bool] = True,
        weights: pd.Series | np.ndarray | Sequence[float] | None = None,
        center_type: CenterTypeNoFunction | dict[str, CenterTypeNoFunction] = "average",
        radius_type: RadiusType | dict[str, RadiusType] = "average",
        per_instance: bool = True,
    ) -> Self:
        """Cluster pharmacophore features using a [hierarchical agglomerative clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) algorithm.

        This method creates the clustering functions from the agglomerative clustering parameters
        `distance_threshold`, `n_clusters`, `linkage`, `metric`, and `memory`,
        and then calls the `Pharmacophore.cluster` method with these functions,
        passing the other general clustering parameters as well.

        For the agglomerative clustering parameters,
        see the documentation for the underlying clustering routine
        [`sklearn.cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering).
        For other parameters, see the `Pharmacophore.cluster` method documentation.
        """
        args = PharmClusterAggInput(
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
            memory=memory,
            min_members=min_members,
            noise_as_singleton=noise_as_singleton,
            weights=weights,
            center_type=center_type,
            radius_type=radius_type,
            per_instance=per_instance,
            n_features=len(self.features),
            feature_types=self.feature_types,
        )
        return self.cluster(
            function=args.function,
            min_members=args.min_members,
            noise_as_singleton=args.noise_as_singleton,
            weights=args.weights,
            center_type=args.center_type,
            radius_type=args.radius_type,
            per_instance=args.per_instance
        )

    def cluster_cnn(
        self,
        max_distance: PositiveFloat | Sequence[PositiveFloat] | dict[str, PositiveFloat | Sequence[PositiveFloat]],
        min_neighbors: PositiveInt | Sequence[PositiveInt] | dict[str, PositiveInt | Sequence[PositiveInt]],
        min_members: PositiveInt | dict[str, PositiveInt] = 1,
        max_members: PositiveInt | dict[str, PositiveInt] | None = None,
        # Parameters for `self.cluster()`
        noise_as_singleton: bool | dict[str, bool] = True,
        weights: pd.Series | np.ndarray | Sequence[float] | None = None,
        center_type: CenterTypeNoFunction | dict[str, CenterTypeNoFunction] = "average",
        radius_type: RadiusType | dict[str, RadiusType] = "average",
        per_instance: bool = True,
    ) -> Self:
        """Cluster pharmacophore features using the Common Nearest Neighbors (CNN) algorithm.

        This method creates the clustering functions from the CNN parameters
        `max_distance`, `min_neighbors`, `min_members`, and `max_members`,
        and then calls the `Pharmacophore.cluster` method with these functions,
        passing the other general clustering parameters as well.
        The CNN parameters are described below;
        for other parameters, see the `Pharmacophore.cluster` method documentation.

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
        args = PharmClusterCNNInput(
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            min_members=min_members,
            max_members=max_members,
            noise_as_singleton=noise_as_singleton,
            weights=weights,
            center_type=center_type,
            radius_type=radius_type,
            per_instance=per_instance,
            n_features=len(self.features),
            feature_types=self.feature_types,
        )
        return self.cluster(
            function=args.function,
            min_members=args.min_members,
            noise_as_singleton=args.noise_as_singleton,
            weights=args.weights,
            center_type=args.center_type,
            radius_type=args.radius_type,
            per_instance=args.per_instance
        )

    def match(
        self,
        query: Self | DataFrameLike,
        algorithm: Literal["greedy", "linear"] = "linear",
        max_distance: float | Literal["radius_sum"] | None = "radius_sum",
        max_distance_inclusive: bool = True,
        raise_missing_types: bool = False,
    ) -> pd.DataFrame:
        """Match a query pharmacophore against this pharmacophore.

        For each feature (i.e. unique combination of `instance`, `type`, and `label`)
        in the input query pharmacophore, this method finds the best matching feature
        of the same type in each instance of the target pharmacophore (i.e. self), if any.
        Subsequently, for each matched pair, the distance between their centers
        and the sum of their radii are computed.

        Parameters
        ----------
        query
            Query pharmacophore features to match against the target pharmacophore.
            This can be a `Pharmacophore` instance or any object convertible to a DataFrame
            with the same structure as the `features` DataFrame.
        algorithm
            Matching algorithm to use:

            - "greedy": Greedy nearest neighbor matching.

              For each feature in the query pharmacophore,
              this algorithm finds the closest feature of the same type
              in each instance of the target pharmacophore, if any.
              This means that in each query–target instance pair,
              multiple query features may be matched
              to the same target feature.

            - "linear": [Linear sum assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
              (Hungarian algorithm).

              For each feature in the query pharmacophore,
              this algorithm finds the best matching feature of the same type
              in each instance of the target pharmacophore, if any.
              The best match is determined by minimizing the sum of distances
              between the centers of the query and target features.
              This means that in each query–target instance pair,
              each query feature is matched to at most one target feature,
              and vice versa.
        max_distance
            Maximum distance between query and target features to consider a match.
            If specified, a column named "match" is added to the output DataFrame,
            indicating whether the distance between matching features
            falls within the given threshold.
            This can be a fixed value (float) or the string "radius_sum",
            which indicates that the maximum distance is the sum of the radii
            of the matched query and target features.
            If set to `None`, no "match" column is added.
        max_distance_inclusive
            If `True`, the distance is considered a match
            if it is less than or equal to `max_distance`,
            otherwise it must be strictly less than `max_distance`.
        raise_missing_types
            If `True`, raise an error if the query features contain types
            not present in the target pharmacophore's feature types,
            otherwise treat them as missing.

        Returns
        -------
        DataFrame containing the following columns:
        - `instance`: Query feature instance identifier.
        - `type`: Feature type.
        - `label`: Query feature label.
        - `target_instance`: Feature instance in self.
           If no match is found, this will be `NaN`.
        - `target_label`: Label of the matching feature in self.
           If no match is found, this will be `NaN`.
        - `distance`: Distance between the matched query and target feature centers.
           If no match is found, this will be `NaN`.
        - `radius_sum`: Sum of the radii of the matched query and target features.
           If no match is found, this will be `NaN`.
        - `match`: Boolean indicating if distance is less than `max_distance`.
           Only present if `max_distance` is not `None`.

        The DataFrame thus contains N rows for each row in the query,
        where N is the number of instances in the target pharmacophore.
        """
        query_unverified = query.features if isinstance(query, Pharmacophore) else query
        query = PharmFeaturesInput(
            features=query_unverified,
            feature_types=self.feature_types if raise_missing_types else None
        ).features

        # Put query row indices in a column for later reference
        query = query.reset_index().rename(columns={'index': 'query_idx'})

        # if there are no target features at all, just return the queries with NaNs
        if self._features_for_match.empty:
            df = query[['instance', 'type', 'label']].copy()
            df['target_instance'] = np.nan
            df['target_label']    = np.nan
            df['distance']        = np.nan
            df['radius_sum']      = np.nan
            if max_distance is not None:
                df['match'] = False
            return df.reset_index(drop=True)
        if algorithm == "greedy":
            matches = self._match_greedy(query=query)
        elif algorithm == "linear":
            matches = self._match_linear(query=query)
        else:
            raise ValueError(f"Unknown matching algorithm '{algorithm}'. Supported: 'greedy', 'linear'.")
        # Reorder columns
        final_cols = [
            'instance',
            'type',
            'label',
            'target_instance',
            'target_label',
            'radius_sum',
            'distance',
        ]
        if max_distance is not None:
            distance_threshold = matches['radius_sum'] if max_distance == "radius_sum" else max_distance
            matches["match"] = (
                (matches['distance'] <= distance_threshold)
                if max_distance_inclusive else
                (matches['distance'] < distance_threshold)
            )
            final_cols.append('match')
        return matches[final_cols].sort_values(["instance", "type", "label", "target_instance"]).reset_index(drop=True)

    def _match_greedy(self, query: pd.DataFrame) -> pd.DataFrame:
        """Match a query pharmacophore against this pharmacophore using a greedy nearest neighbor approach."""
        # Cross-join ligand × instance
        # This creates a DataFrame with all combinations of query features and target instances
        query['_key'] = 1
        cross = query.merge(self._feature_instances_for_match, on='_key').drop(columns=['_key'])

        # Merge with target features on instance & type
        # This will add target feature data to each query × target-instance pair,
        # repeating each query × target-instance row
        # for each target feature of the same type in that instance.
        merged = cross.merge(
            self._features_for_match,
            on=['target_instance', 'type'],
            how='left'
        ).convert_dtypes()

        # Ensure these cols always exist, even if merged is empty
        merged['radius_sum'] = np.nan
        merged['distance'] = np.nan

        # Compute distances where matching features exist
        mask = merged['target_label'].notna()
        if mask.any():
            query_centers = np.stack(merged.loc[mask, 'center'].values)
            target_centers = np.stack(merged.loc[mask, 'target_center'].values)
            merged.loc[mask, 'distance'] = np.linalg.norm(query_centers - target_centers, axis=1)
            merged.loc[mask, "radius_sum"] = merged.loc[mask, 'radius'] + merged.loc[mask, 'target_radius']

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
        return best

    def _match_linear(self, query: pd.DataFrame) -> pd.DataFrame:
        """Match a query pharmacophore against this pharmacophore using a linear sum assignment approach."""
        rows: list[dict[str, Any]] = []
        # Iterate over each instance–type combination in the query
        for (query_instance, feature_type), subquery in query.groupby(
            ["instance", "type"],
            sort=False,
        ):
            subquery = subquery.reset_index(drop=True)
            # Iterate over each target instance
            for target_instance, target_instance_features in self._features_per_instance:
                # Select features of the same type in the target instance
                subtarget = target_instance_features[target_instance_features['type'] == feature_type].reset_index(drop=True)
                # If there are no features of this type in the target instance,
                # add empty rows for each query feature
                if subtarget.empty:
                    for _, query_feature in subquery.iterrows():
                        rows.append({
                            'instance': query_instance,
                            'type': feature_type,
                            'label': query_feature['label'],
                            'target_instance': target_instance,
                            'target_label': np.nan,
                            'distance': np.nan,
                            'radius_sum': np.nan,
                        })
                    continue
                # Calculate distances between query and target features of the same type in this instance pair
                query_centers = np.stack(subquery['center'].values)
                target_centers = np.stack(subtarget['center'].values)
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
                distances = scipy.spatial.distance.cdist(
                    query_centers,
                    target_centers,
                    metric='euclidean'
                )
                # Solve the linear sum assignment problem
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
                match_idx_query, match_idx_target = scipy.optimize.linear_sum_assignment(distances)
                # Iterate over each query feature in this instance–type combination
                for query_idx, query_feature in subquery.iterrows():
                    mask = match_idx_query == query_idx
                    if not mask.any():
                        # No match found for this query feature
                        rows.append({
                            'instance': query_instance,
                            'type': feature_type,
                            'label': query_feature['label'],
                            'target_instance': target_instance,
                            'target_label': np.nan,
                            'distance': np.nan,
                            'radius_sum': np.nan,
                        })
                        continue
                    target_idx = match_idx_target[mask][0]
                    target_feature = subtarget.iloc[target_idx]
                    rows.append({
                        'instance': query_instance,
                        'type': feature_type,
                        'label': query_feature['label'],
                        'target_instance': target_instance,
                        'target_label': target_feature['label'],
                        'distance': distances[query_idx, target_idx],
                        'radius_sum': query_feature['radius'] + target_feature['radius'],
                    })
        df = pd.DataFrame(rows)
        df['distance'] = df['distance'].astype(float)
        df['radius_sum'] = df['radius_sum'].astype(float)
        return df.convert_dtypes()

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        instances: Sequence[Any] | None = None,
        feature_types: Sequence[str] | None = None,
        system: System | Literal[False] | None = None,
        default_radius: float = 1.5,
        min_radius: float = 1.0,
        show_box: bool = True,
        show_pocket: bool = True,
        show_fields: bool = False,
        field_only_in_pocket: bool = True,
        show_feature_centers: bool = True,
        show_feature_points: bool = False,
        feature_colors: dict[str, tuple[float, float, float] | tuple[int, int, int]] | None = None,
        override_radius: bool = False,
        gui: bool = True,
        directed_features_components: set[Literal["sphere", "arrow"]] = {"arrow"},
        add_residues: bool = True,
    ):
        def feature_color(feature_id: str) -> tuple[float, float, float] | tuple[int, int, int]:
            """Get color for a feature type, defaulting to gray if not set."""
            return feature_colors.get(
                feature_id,
                self._feature_colors.get(feature_id, (0.5, 0.5, 0.5))
            )

        def normalize_name(name: Any) -> str:
            if isinstance(name, tuple | list | np.ndarray):
                return "-".join(map(str, name))
            return str(name)


        nv = nglwidget or scishow.nglview.NGLWidget()
        feature_colors = feature_colors or {}

        # System
        atoms = None
        system_comp = None
        if system is not False:
            if system is not None:
                atoms = system.composition.atoms
                system_comp = nv.add_trajectory(system)
            elif self.system is not None:
                atoms = self.system.composition.atoms
                system_comp = nv.add_trajectory(self.system)

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
            field = self._field_in_pocket if field_only_in_pocket else self._field
            for feature_type in field.batch_instance_labels["feature"]:
                nv.add_volume(
                    data=field(feature=feature_type),
                    basis=field.grid.unit_vectors,
                    origin=field.grid.lower_bounds,
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
        for _, feature in self.features.sort_values(
            ["instance", "type", "value" if "value" in self.features else "label"]
        ).iterrows():
            if instances is not None and feature["instance"] not in instances:
                continue
            if feature_types is not None and feature["type"] not in feature_types:
                continue
            instance = normalize_name(feature["instance"])
            ftype = normalize_name(feature["type"])
            label = normalize_name(feature["label"])
            name = f"{instance}_{ftype}_{label}"
            radius = feature["radius"]
            end = feature.get("end", None)
            feat_has_direction = isinstance(end, np.ndarray)
            if not feat_has_direction or "sphere" in directed_features_components:
                nv.add_spheres(
                    coords=feature["center"],
                    radii=max(radius, min_radius) if not override_radius else default_radius,
                    name=f"{name} Center",
                    colors=feature_color(feature["type"]),
                    representation_params=scishow.nglview.RepresentationParameters(
                        opacity=0.8,
                        visible=show_feature_centers,
                        lazy=True,
                    )
                )
            if feat_has_direction and "arrow" in directed_features_components:
                nv.shape.add_arrow(
                    feature["center"].tolist(),
                    feature["end"].tolist(),
                    feature_color(feature["type"]),
                    0.25,
                    f"{name} Direction",
                )
            if atoms is not None and add_residues and "res_idx" in feature and pd.notna(feature["res_idx"]):
                res = atoms[atoms["res_idx"] == feature["res_idx"]].iloc[0]
                res_sel = f"{res["res_seq"]}^{res["i_code"]}:{res["chain_id"]}"
                system_comp.add_ball_and_stick(res_sel, name=f"{name} Residue")
        if gui:
            nv.display(gui=True)
        return nv

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

    def new(
        self,
        features: DataFrameLike | None = None,
        feature_types: set[str] | None = None,
        inputs: Sequence[dict[str, Any]] | None = None,
        name: str | None = None,
        system: Any | None = None,
        pocket: Pocket | None = None,
        field: Field | None = None,
        extra: dict[str, Any] | None = None,
    ):
        return Pharmacophore(
            features=features if features is not None else self.features,
            feature_types=feature_types if feature_types is not None else self.feature_types,
            inputs=inputs if inputs is not None else self.inputs,
            name=name if name is not None else self.name,
            system=system if system is not None else self.system,
            pocket=pocket if pocket is not None else self.pocket,
            field=field if field is not None else self.field,
            extra=extra if extra is not None else self.extra,
        )

    def _check_instance_consistency_with_pocket(self, pocket: Pocket) -> None:
        """Check if the pharmacophore instances are consistent with the pocket instances."""
        if pocket.batch_ndim != self._batch_shape.size:
            raise ValueError(
                f"Instance dimensions of the pharmacophore ({self._batch_shape}) "
                f"and the pocket ({pocket.batch_ndim}) do not match."
            )
        if self._batch_shape.size == 0:
            return
        if (pocket.batch_shape < self._batch_shape).any():
            raise ValueError(
                f"The pocket has fewer instances ({pocket.batch_shape}) "
                f"than the pharmacophore ({self._batch_shape})."
            )
        return


def merge(pharmacophores: Sequence[Pharmacophore]) -> Pharmacophore:
    """Merge multiple pharmacophores into a single one.

    Parameters
    ----------
    pharmacophores
        Sequence of `Pharmacophore` objects to merge.

    Returns
    -------
    A new `Pharmacophore` object containing the merged features.
    The `features` DataFrame will contain all features from the input pharmacophores,
    with the `instance` column indicating the source pharmacophore.
    """
    def instance_merger(instance_value, instance_prefix):
        if isinstance(instance_value, (tuple, list, np.ndarray)):
            return (instance_prefix, *instance_value)
        return (instance_prefix, instance_value)


    if not pharmacophores:
        raise ValueError("No pharmacophores to merge.")

    dfs = []
    names = _uniquify([pharm.name for pharm in pharmacophores])
    for pharm_name, pharm in zip(names, pharmacophores):
        feats = pharm.features.copy()
        feats["instance"] = (
            pharm_name if feats["instance"].nunique() == 1 else
            feats["instance"].apply(instance_merger, instance_prefix=pharm_name)
        )
        dfs.append(feats)

    merged_features = pd.concat(dfs, ignore_index=True)
    feature_types = set().union(*(ph.feature_types for ph in pharmacophores))

    return Pharmacophore(
        features=merged_features,
        feature_types=feature_types,
    )


def _uniquify(strings: list[str]) -> list[str]:
    """Return list with duplicates suffixed by incrementing counters.

    Each repeated string is suffixed with _{i}, where i starts from 1
    for the first occurrence of that duplicate. Strings that occur only
    once are left unchanged. The output preserves order and length of
    the input list, and ensures all values in the returned list are
    unique.

    Parameters
    ----------
    strings
        List of input strings, possibly with duplicates.

    Returns
    -------
    list[str]
        New list of strings, same length and order as input,
        with duplicates enumerated by suffix while singletons
        remain unchanged.

    Examples
    --------
    >>> uniquify(["a", "b", "a", "c", "b", "a", "d"])
    ['a_1', 'b_1', 'a_2', 'c', 'b_2', 'a_3', 'd']
    """
    total_counts: defaultdict[str, int] = defaultdict(int)
    for s in strings:
        total_counts[s] += 1

    running_counts: defaultdict[str, int] = defaultdict(int)
    result: list[str] = []

    for s in strings:
        if total_counts[s] == 1:
            result.append(s)
        else:
            running_counts[s] += 1
            result.append(f"{s}_{running_counts[s]}")
    return result


def _extrema_under_footprint(
    field: np.ndarray,
    field_indices: np.ndarray,
    footprint: np.ndarray,
    *,
    maximize: bool = False,
) -> np.ndarray:
    """Return argmin/argmax indices in `field` under a 3D boolean footprint centered at given indices.

    Places the center of `footprint` (which must have odd lengths along each axis) on each
    index in `field_indices`, applies natural clipping at borders, and finds the index in
    `field` (same global coordinates) of the **extreme** value (minimum by default;
    maximum if `maximize=True`) among locations where `footprint` is True.
    The footprint operates over the **last three axes** of `field`. All leading axes
    (if any) are taken exactly from each row of `field_indices`.

    Parameters
    ----------
    field
        N-dimensional array (N >= 3). The last 3 axes are the spatial axes affected by
        the footprint. Must be indexable with integer coordinates from `field_indices`.
    field_indices
        2D array of shape (K, N) with integer indices. Each row specifies a global index
        in `field` at which the *center* of `footprint` is placed.
    footprint
        3D boolean array with odd shape along each axis (so it has a unique center).
        This footprint is aligned with the last 3 axes of `field`.

    Returns
    -------
    np.ndarray
        Array of shape (K, N) with the global indices (same order as `field_indices`)
        of the selected extreme (min or max) element in `field` under the footprint
        for each placement.

    Raises
    -------
    ValueError
        If input shapes/dtypes are invalid (e.g., N < 3, footprint not 3D, footprint
        has even length on any axis, or `field_indices` shape mismatch).

    Notes
    ------
    - If the footprint region is partially outside the array, only the in-bounds portion
      is considered.
    - If, after clipping, there are no True cells in the footprint slice (should not happen
      when the footprint center is True and the center is in-bounds), this function falls
      back to returning the original `field_indices` row for that placement.
    """
    if field.ndim < 3:
        raise ValueError(f"`field` must be at least 3D; got {field.ndim}D")
    if footprint.ndim != 3:
        raise ValueError(f"`footprint` must be 3D; got {footprint.ndim}D")
    if any(s % 2 == 0 for s in footprint.shape):
        raise ValueError(f"`footprint` must have odd lengths; got shape {footprint.shape}")
    if field_indices.ndim != 2 or field_indices.shape[1] != field.ndim:
        raise ValueError(
            f"`field_indices` must have shape (K, {field.ndim}); got {field_indices.shape}"
        )
    if not np.issubdtype(field_indices.dtype, np.integer):
        raise ValueError("`field_indices` must be of integer dtype")

    N = field.ndim
    K = field_indices.shape[0]
    out = np.empty((K, N), dtype=np.int64)

    # Radii (half-sizes) of the footprint along its 3 axes
    rad_z, rad_y, rad_x = (d // 2 for d in footprint.shape)

    # Helper to compute slice bounds (field and footprint) for one axis with center c, radius r, and limit L
    def _bounds(c: int, r: int, L: int) -> tuple[slice, slice]:
        # Field slice [f0:f1)
        f0 = max(0, c - r)
        f1 = min(L, c + r + 1)
        # Map back to footprint slice [p0:p1)
        # Position of f0 in footprint coords:
        p0 = r - (c - f0)
        # Length matches field segment:
        p1 = p0 + (f1 - f0)
        return slice(f0, f1), slice(p0, p1)

    # Precompute for speed
    field_shape_last3 = field.shape[-3:]

    # Choose comparator
    extreme = np.argmax if maximize else np.argmin

    for k in range(K):
        idx = field_indices[k]
        # Split index into leading axes (if any) and last-3 axes
        lead_idx = tuple(idx[:-3]) if N > 3 else ()
        zc, yc, xc = (int(idx[-3]), int(idx[-2]), int(idx[-1]))

        # Bounds per last-3 axes
        (fz, pz) = _bounds(zc, rad_z, field_shape_last3[0])
        (fy, py) = _bounds(yc, rad_y, field_shape_last3[1])
        (fx, px) = _bounds(xc, rad_x, field_shape_last3[2])

        # Extract field view and footprint slice
        fview = field[(*lead_idx, fz, fy, fx)]
        pview = footprint[pz, py, px]

        # Guard: ensure pview has any True
        if not pview.any():
            # Fallback to center (should be rare if center of footprint is True)
            out[k] = idx
            continue

        # Mask invalid cells by setting them to +inf/-inf depending on min/max
        # Copy only if needed
        view = fview
        if not pview.all():
            # We must ignore where pview is False
            if maximize:
                masked = np.where(pview, view, -np.inf)
            else:
                masked = np.where(pview, view, np.inf)
            flat_idx = extreme(masked.ravel())
        else:
            flat_idx = extreme(view.ravel())

        # Convert flat idx back to local (z,y,x) offsets
        local_zyx = np.unravel_index(flat_idx, fview.shape)
        # Compose global index
        gz = fz.start + local_zyx[0]
        gy = fy.start + local_zyx[1]
        gx = fx.start + local_zyx[2]

        if N > 3:
            out[k, :-3] = np.array(lead_idx, dtype=np.int64)
        out[k, -3:] = (gz, gy, gx)

    return out
