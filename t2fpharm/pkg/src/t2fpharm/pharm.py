"""Pharmacophore."""

from typing import Sequence, Any, Self, Literal
from pathlib import Path

import numpy as np
import pandas as pd

import scids.functional.dist
import scishow
import caddpy
import scids

from t2fpharm.system import System
from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.input.pharm.cluster import PharmClusterInput, ClusteringFunction, CenterType, CenterTypeNoFunction, RadiusType
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
        inputs: Sequence[dict[str, Any]] | None = None,
        name: str = "Pharmacophore",
        system: Any | None = None,
        pocket: Pocket | None = None,
        field: Field | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self._features = PharmFeaturesInput(features=features, feature_types=feature_types).features

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
    def inputs(self) -> list[dict[str, Any]]:
        """Inputs used to create the pharmacophore."""
        return self._inputs

    @property
    def name(self) -> str:
        """Name of the pharmacophore."""
        return self._name

    @property
    def system(self) -> Any | None:
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
            This must be a 1D array-like object
            with the same length and order as features in `self.features`.
        highest_priority
            How to interpret the `priority` values:
            - "lowest": The lowest value is the highest priority.
            - "highest": The highest value is the highest priority.
        max_features
            Maximum number of features to keep.
            This can be a single value for all feature types
            (i.e., to keep at most `max_features` features per each type),
            or a dictionary mapping each feature type to its maximum number of features.

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
            sorted_grp = group.sort_values("_priority", ascending=args.highest_priority == "lowest")
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
        weights: pd.Series | np.ndarray | Sequence[float] | None = None,
        center_type: CenterType | dict[str, CenterType] = "average",
        radius_type: RadiusType | dict[str, RadiusType] = "average",
        per_instance: bool = True,
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

        Parameters
        ----------
        function
            Either a single clustering function for all feature types,
            or a dictionary mapping each feature type to a clustering function.
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
            and will not be included in the output pharmacophore.
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
        - `center`: Center coordinates of the cluster.
        - `radius`: Radius of the cluster.
        - `members`: Row indices of the original features that belong to this cluster.
        """
        args = PharmClusterInput(
            function=function,
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

        new_features: list[dict] = []
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
            unique_positive_labels = unique_labels[unique_labels > 0]
            instance = group_idx[0] if per_instance else 0
            feature_center_type = center_type[feature_type]
            feature_radius_type = radius_type[feature_type]
            for label in unique_positive_labels:
                label_mask = labels == label
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
                if clustering_result.centers is not None:
                    cluster_centers["center_function"] = clustering_result.centers[label]

                # Calculate center values
                values = {}
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
                member_indices = group.index[label_mask].to_numpy()
                new_features.append(
                    {
                        "instance": instance,
                        "type": feature_type,
                        "label": label,
                        "center": cluster_centers[f"center_{feature_center_type}"],
                        "radius": cluster_radii[f"radius_{feature_center_type}_{feature_radius_type}"],
                        "value": values[f"value_{feature_center_type}"],
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
            ),
            feature_types=self.feature_types,
            inputs=self.inputs + [args.model_dump()],
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
            weights=weights,
            center_type=center_type,
            radius_type=radius_type,
            per_instance=per_instance,
            n_features=len(self.features),
            feature_types=self.feature_types,
        )
        return self.cluster(
            function=args.function,
            weights=args.weights,
            center_type=args.center_type,
            radius_type=args.radius_type,
            per_instance=args.per_instance
        )

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
           If no match is found, this will be `NaN`.
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
        system: Any | Literal[False] | None = None,
        default_radius: float = 1.5,
        show_box: bool = True,
        show_pocket: bool = True,
        show_fields: bool = False,
        show_feature_centers: bool = True,
        show_feature_points: bool = False,
        feature_colors: dict[str, tuple[float, float, float] | tuple[int, int, int]] | None = None,
        overdide_radius: bool = False,
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
        if system is not False:
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
            radius = feature["radius"]
            nv.add_spheres(
                coords=feature["center"],
                radii=radius if radius and not overdide_radius else default_radius,
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
    type_hbond_acceptor: str | None = "OA",
    type_hbond_donor: str | None = "HD",
    type_water_bridge_ligand_acceptor: str | None = "OA",
    type_water_bridge_ligand_donor: str | None = "HD",
    type_water_bridge_water_acceptor: str | None = "OA",
    type_anion: str | None = "e-",
    type_cation: str | None = "e+",
    type_hydrophobic: str | None = "C",
    type_aromatic: str | None = "A",
    pocket: Pocket | None = None,
    receptor: System | None = None,
):
    plip = caddpy.interaction.from_pdb(pdb_files, ligands=ligands)
    out = []
    for _, row in plip.all.iterrows():
        selected: list[tuple[str, str]] = []
        match row["type"]:
            case "hbond":
                if row["r_is_d"]:
                    # Ligand is acceptor
                    if type_hbond_acceptor:
                        selected.append((type_hbond_acceptor, "l_position"))
                else:
                    # Ligand is donor
                    if type_hbond_donor:
                        selected.append((type_hbond_donor, "h_position"))
            case "water_bridge":
                # Plip only detects bridges where ligand and receptor have different roles
                # i.e., ligand is acceptor and receptor is donor, or vice versa.
                if row["r_is_d"]:
                    # Ligand and water are acceptors
                    if type_water_bridge_ligand_acceptor:
                        selected.append((type_water_bridge_ligand_acceptor, "l_position"))
                    if type_water_bridge_water_acceptor:
                        selected.append((type_water_bridge_water_acceptor, "w_position"))
                else:
                    # Ligand and water are donors
                    if type_water_bridge_ligand_donor:
                        selected.append((type_water_bridge_ligand_donor, "l_position"))
                    # Position of water hydrogen is not available in PLIP
            case "salt_bridge":
                if row["r_is_cation"]:
                    # Ligand is anion
                    if type_anion:
                        selected.append((type_anion, "l_position"))
                else:
                    # Ligand is cation
                    if type_cation:
                        selected.append((type_cation, "l_position"))
            case "hydrophobic":
                if type_hydrophobic:
                    selected.append((type_hydrophobic, "l_position"))
            case "pi_stacking":
                if type_aromatic:
                    selected.append((type_aromatic, "l_position"))
            case _:
                continue
        selected = [(feature_type, row[position_col]) for feature_type, position_col in selected]
        # Sometimes a single atom can be involved
        # in multiple interactions of the same type, e.g., hydrophobic interactions with different residues.
        # Therefore, we only add a new feature if not already present
        for feature_type, position in selected:
            for entry in out:
                if entry["type"] == feature_type and np.allclose(entry["center"], position):
                    break
            else:
                out.append({"instance": row.get("instance", 0), "type": feature_type, "center": position})
    if pocket is not None:
        positions = np.stack([feature["center"] for feature in out])
        coverages = pocket.point_coverage(positions)
        out = [feature for feature, coverage in zip(out, coverages) if coverage]
    return Pharmacophore(features=out, extra={"plip": plip}, system=receptor)
