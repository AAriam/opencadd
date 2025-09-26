"""Pharmacophore."""

from typing import Sequence, Any, Self, Literal, Final, Callable
from collections import defaultdict
import functools

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.spatial

import scids.functional.dist
import scishow
import scids

from t2fpharm.dist import DistanceMatrixFunction, linear as distmatrix_linear
from t2fpharm.system import System
from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.input.pharm.cluster import PharmClusterInput, ClusteringFunction, CenterType, CenterTypeNoFunction, RadiusType
from t2fpharm.input.pharm.cluster_agg import PharmClusterAggInput, AggLinkageType, AggLinkageMetricType
from t2fpharm.input.pharm.cluster_cnn import PharmClusterCNNInput
from t2fpharm.input.pharm.features import PharmFeaturesInput
from t2fpharm.input.pharm.remove_overlaps import RemoveOverlapsInput
from t2fpharm.typing import DataFrameLike, PositiveFloat, PositiveInt, ArrayLike

import scids.functional


class Pharmacophore:
    """Pharmacophore.

    This class represents a pharmacophore as a collection of features,
    where each feature can be a point, vector,
    or radial (spherical) feature in 3D space,
    with additional associated data and identifiers.
    It provides methods to manipulate, analyze, and visualize the pharmacophore.
    The pharmacophore can also be associated with a chemical system,
    binding pocket, and field, which are used for different operations
    such as filtering and refining features.

    Feature representation types are defined as follows:
    1. **Point features** are defined by a `center` coordinate only.
       The center represents the location of the feature's key interaction point.
    2. **Vector features** are defined by a `center` and an `end` coordinate.
       The end point represents the location of the feature's (expected) interacting partner.
       Vector features also have a `radius`,
       which is simply the length of the vector,
       i.e., the distance between their `center` and `end` points.
    3. **Radial features** are defined by an `end` coordinate and a `radius`.
       The end point represents the center of the sphere,
       and the radius defines its size.
       Radial features do not have a `center` coordinate;
       the center can be anywhere on the surface of the sphere.
       They are useful for representing non-directional features
       such as hydrophobic and ionic interactions.

    Parameters
    ----------
    features
        DataFrame-like object containing pharmacophore feature data.
        It can be a `pandas.DataFrame`, or any object that can be
        converted to a DataFrame using the `pandas.DataFrame()` constructor.
        Each row in the resulting DataFrame must represent
        a pharmacophore feature with the following columns:
        - `instance`: Integer or tuple of integers
          representing the index of the feature instance,
          e.g., for when the pharmacophore is derived
          from multiple receptors or ligands.
          If not present, a default value of 0 is added to all features.
        - `type`: String or integer representing the feature type,
           e.g., "hbond_donor", "hbond_acceptor", "hydrophobic", etc.
        - `label`: A hashable identifier for different features of the same type
           within the same instance. That is, for each unique (`instance`, `type`) pair,
           each feature must have a unique label.
           Each feature in the whole pharmacophore can thus be uniquely identified
           by its (`instance`, `type`, `label`) triplet.
           If not present, it will be added with sequential integers starting from 1.
        - `atom_idxs`: Tuple of integers representing the indices of the atoms
          in `system` that contribute to the feature center,
          i.e., the atoms at `end` for vector and radial features.
          If not present, it will be added with `None` values.
        - `repr`: Integer specifying the feature representation:
            - 1: Point feature defined by `center` only.
            - 2: Vector feature defined by `center` and `end`.
            - 3: Radial feature defined by `end` and `radius`.
            If not present, it will be inferred from the presence of
            the `center`, `end`, and `radius` columns.
        - `radius`: A non-negative real number
           representing the radius for radial features,
           or the length of the vector for vector features.
           For point features, this column should be `NaN`.
        - `center`: NumPy array representing the 3D coordinates
          of the feature's center in some reference frame.
          For radial features, this column should be `None`.
        - `end`: NumPy array representing the 3D coordinates
          of the feature's end point in some reference frame.
          For point features, this column should be `None`.
        - `radius_tol`, `center_tol`, `end_tol`: Optional columns representing
          the uncertainty (tolerance) in the corresponding
          `radius`, `center`, and `end` values.
          These are used when matching features to allow for some flexibility
          in the feature positions and sizes.
          When a method increases the uncertainty of a feature,
          the corresponding tolerance value is increased accordingly.
          If not present, they are assumed to be zero.
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
        else:
            self._batch_shape = np.vstack(self._features["instance"]).max(axis=0) + 1

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

    def add_atomic_data(
        self,
        columns: Sequence[str],
        suffix: str = "_from_system",
        collapse: bool = True,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """Add atom-specific data to the features DataFrame.

        This method adds atom-specific information
        from the `self.system.composition.atoms` DataFrame
        as columns to the `self.features` DataFrame,
        based on the `atom_idxs` values of each feature.

        Parameters
        ----------
        columns
            Name of columns in the `self.system.composition.atoms` DataFrame
            to include in the features DataFrame of the pharmacophore.
        suffix
            Suffix to append to the column names from the atoms DataFrame
            to avoid name clashes with existing columns in the features DataFrame.
        collapse
            Whether to collapse the values in each cell of the added columns
            when all values are the same.
            This is only done for columns where all rows satisfy this condition.
            Otherwise (or if False), the values are added as tuples
            with the same order as the `atom_idxs` values.
        inplace
            Whether to also modify the features DataFrame of the pharmacophore in place.

        Returns
        -------
        DataFrame
            Updated features DataFrame with the additional atom-specific columns.
        """
        def _tuple_for_row(atom_tuple: tuple[int, ...], mapping: dict[int, Any]) -> tuple[Any, ...]:
            """Return a tuple of values for the given atom indices, preserving order."""
            return tuple(mapping[int(idx)] for idx in atom_tuple)

        def _all_equal(seq: tuple[Any, ...]) -> bool:
            """True if all elements in seq are equal (or seq has length 0/1)."""
            it = iter(seq)
            try:
                first = next(it)
            except StopIteration:
                return True
            return all(el == first for el in it)

        features = self._features if inplace else self._features.copy()
        if not suffix or any(col.endswith(suffix) for col in features.columns):
            raise ValueError(
                f"Cannot add columns because the specified suffix '{suffix}' "
                "is either empty or already used by existing columns in the features DataFrame."
            )
        atom = self.system.composition.atoms
        missing_columns = [col for col in columns if col not in atom.columns]
        if missing_columns:
            raise ValueError(
                f"Cannot add atom columns because the following columns "
                f"are missing from the system's atoms DataFrame: {missing_columns}."
            )

        # Build fast lookup dicts for each requested column
        atom_indexed = atom.set_index("atom_idx", drop=True)
        index_to_values: dict[str, dict[int, Any]] = {
            col: {int(k): v for k, v in atom_indexed[col].to_dict().items()}
            for col in columns
        }

        # Compute and attach columns
        for col, mapping in index_to_values.items():
            final_name = f"{col}{suffix}" if col in features.columns else col
            # Create tuple values per feature row (preserve atom order)
            tuple_series = features["atom_idxs"].map(lambda t: _tuple_for_row(t, mapping))
            if collapse:
                # Collapse only if *every* row has identical values within its tuple
                if bool(len(tuple_series)) and tuple_series.map(_all_equal).all():
                    # Replace each tuple with its single representative value (or None if empty)
                    collapsed = tuple_series.map(lambda t: (t[0] if len(t) > 0 else None))
                    features[final_name] = collapsed
                else:
                    features[final_name] = tuple_series
            else:
                features[final_name] = tuple_series
        return features

    def filter(
        self,
        mask: pd.Series | np.ndarray | Sequence[bool] | None = None,
        operation_mask: Literal["&", "|"] = "&",
        operation_kwargs: Literal["&", "|"] = "&",
        name: str | None = None,
        **kwargs,
    ) -> Self:
        """Filter pharmacophore features.

        Parameters
        ----------
        mask
            Optional boolean mask to filter features.
            It can be a `pandas.Series`, a numpy array,
            or any sequence of boolean values.
            It must have the same length and order as `self.features`.
            If `None`, only the keyword arguments are used for filtering.
        operation_mask
            Logical operation to combine the `mask` with the mask
            derived from the keyword arguments, if both are provided.
            It can be either `"&"` (logical AND) or `"|"` (logical OR).
        operation_kwargs
            Logical operation to combine multiple keyword argument conditions, if provided.
            It can be either `"&"` (logical AND) or `"|"` (logical OR).
        name
            Optional name for the new pharmacophore.
            If `None`, the name of the current pharmacophore is used.
        **kwargs
            Keyword arguments to filter features based on column values.
            Each keyword argument must correspond to a column in `self.features`.
            The value can be a single value or a list of values.
            If a single value is provided, it is used to select features
            where the column equals that value.
            If a list is provided, it is used to select features
            where the column value is in that list.

            For example, `type="HD"` selects all hydrogen bond donor features,
            while `type=["HD", "OA"]` selects all hydrogen bond donor and acceptor features.

        Returns
        -------
        A new `Pharmacophore` instance containing only the features
        that match the provided mask and keyword argument conditions.
        """
        if mask is not None:
            if len(mask) != len(self._features):
                raise ValueError(
                    f"Mask length {len(mask)} does not match number of features {len(self._features)}."
                )
        reduce_func_map = {
            "&": lambda x, y: x & y,
            "|": lambda x, y: x | y,
        }
        reduce_func = {}
        for param_name, arg in (("operation_mask", operation_mask), ("operation_kwargs", operation_kwargs)):
            try:
                reduce_func[param_name] = reduce_func_map[arg]
            except KeyError:
                raise ValueError(f"Unsupported argument for '{param_name}': '{arg}'. Use '&' or '|'.")

        conditions = []
        for col_name, value in kwargs.items():
            if col_name not in self.features.columns:
                raise KeyError(f"Column '{col_name}' not in features DataFrame.")
            col = self.features[col_name]
            if isinstance(value, str | int | float | bool):
                func = col.eq
            else:
                func = col.isin
            conditions.append(func(value))

        mask_kwargs = None
        if conditions:
            mask_kwargs = functools.reduce(reduce_func["operation_kwargs"], conditions)

        has_mask = mask is not None
        has_mask_kwargs = mask_kwargs is not None
        if has_mask and has_mask_kwargs:
            mask = reduce_func["operation_mask"](mask, mask_kwargs)
        elif has_mask_kwargs:
            mask = mask_kwargs
        elif not has_mask:
            raise ValueError("No keyword conditions or mask provided for filtering.")

        return self.new(
            features=self._features[mask],
            inputs=self.inputs + [{"action": "filter", "params": {"mask": mask}}],
            name=name,
        )

    def select_in_pocket(
        self,
        dist_tol: float | ArrayLike | None = None,
        *,
        center_distance: PositiveFloat | dict[str, PositiveFloat] = 2.0,
        keep_radial: bool = False,
    ) -> Self:
        """Select features that are within the associated pocket.

        This method first converts any radial features to vector features
        by placing their centers at a distance of `center_distance` from each other
        on the surface of the sphere.
        Then, it selects only the features whose centers are within the pocket,
        considering the specified distance tolerance `dist_tol`.
        Finally, if `keep_radial` is `True`,
        those remaining vector features that are generated from radial features
        are converted back to their radial representation.
        Note that this results in a loss of information,
        since parts of the regenerated radial feature may extend outside the pocket.

        Parameters
        ----------
        dist_tol
            Distance tolerance for selecting features within the pocket.
            It can be a single float value applied to all features,
            or an array-like object with the same length as `self.features`,
            specifying a different tolerance for each feature.
            If `None`, the `center_tol` values in `self.features` are used.
        center_distance
            Distance between the centers of vector features
            that are generated on the spherical surface of radial features.
            It can be a single positive float value applied to all radial features,
            or a dictionary mapping feature types to their corresponding distances.
        keep_radial
            Whether to convert vector features that were generated from radial features
            back to their radial representation after filtering.

        Returns
        -------
        A new `Pharmacophore` instance containing only the features
        that are within the associated pocket.
        """
        def _apply_group(group: pd.DataFrame) -> pd.DataFrame:
            is_inside, _, distances = self.pocket.point_coverage(
                np.stack(group["center"]),
                tolerance=group["_distance_tol"].to_numpy(),
                instance=group.name
            )
            return pd.DataFrame(
                {"_is_inside": is_inside, "_distance": distances},
                index=group.index
            )

        if self.pocket is None:
            raise ValueError("Cannot refine by pocket: No pocket associated with the pharmacophore.")
        has_radial = self.has_radial
        if has_radial:
            self.features["_converted"] = self.is_radial
            feats = self.convert_feature_radial_to_vector(distance=center_distance).features
        else:
            feats = self.features.copy()

        if dist_tol is None:
            dist_tol = feats["center_tol"].to_numpy()
        elif np.isscalar(dist_tol):
            dist_tol = np.full(len(feats), dist_tol, dtype=np.float64)
        feats["_distance_tol"] = dist_tol

        if self.pocket.batch_ndim == 0:
            is_inside, _, distances = self.pocket.point_coverage(
                np.stack(feats["center"]),
                tolerance=dist_tol,
            )
            feats["_is_inside"] = is_inside
            feats["_distance"] = distances
        else:
            pocket_results = (
                feats.groupby("instance", sort=False, group_keys=False)
                .apply(_apply_group)
            )
            feats = feats.join(pocket_results, sort=False)

        feats["center_tol"] = np.maximum(feats["center_tol"], feats["_distance"])
        feats = feats[feats["_is_inside"]].drop(columns=["_is_inside", "_distance", "_distance_tol"])
        filtered_pharm = self.new(features=feats)
        if has_radial and keep_radial:
            filtered_pharm = filtered_pharm.convert_feature_vector_to_radial(
                mask=filtered_pharm.features["_converted"],
                merge=True,
            )
        if has_radial:
            filtered_pharm.features.drop(columns=["_converted"], inplace=True)
        return filtered_pharm

    def refine_by_field(
        self,
        search_radius: float = 1.5,
        *,
        extrema_type: Literal["min", "max"] = "min",
        dist_tol: float | ArrayLike | None = None,
        center_distance: PositiveFloat | dict[str, PositiveFloat] = 2.0,
        keep_radial: bool = False,
    ) -> pd.DataFrame:

        if self._field is None:
            raise ValueError("Cannot refine by field: No field associated with the pharmacophore.")

        has_radial = self.has_radial
        if has_radial:
            self.features["_converted"] = self.is_radial
            feats = self.convert_feature_radial_to_vector(distance=center_distance).features
        else:
            feats = self.features.copy()

        if dist_tol is None:
            dist_tol = feats["center_tol"].to_numpy()
        elif np.isscalar(dist_tol):
            dist_tol = np.full(len(feats), dist_tol, dtype=np.float64)

        field = self._field
        old_centers = np.stack(feats["center"])
        center_is_inside_grid, grid_indices, _ = field.grid.point_coverage(old_centers, tolerance=dist_tol)
        feats = feats.loc[center_is_inside_grid]
        grid_indices = grid_indices[center_is_inside_grid]

        instances = np.stack(feats["instance"]).reshape(len(feats), -1)

        field_prefix_indices = feats["type"].map(
            {val: idx for idx, val in enumerate(field.batch_instance_labels["feature"])}
        ).to_numpy().reshape(-1, 1)
        if field.batch_ndim > 1:
            # Merge instance indices with grid indices to get full field indices
            field_prefix_indices = np.concatenate([field_prefix_indices, instances], axis=1)

        field_indices = np.concatenate([field_prefix_indices, grid_indices], axis=1)
        footprint = field.grid.footprint_spherical(search_radius)

        if self.pocket is not None:
            pocket = self.pocket
            if pocket.batch_ndim == 0:
                pocket_indices = grid_indices
            else:
                pocket_indices = np.concatenate([instances, grid_indices], axis=1)
            pocket_stencils = pocket.stencil(
                indices=pocket_indices,
                shape=[0] * pocket.batch_ndim + list((np.array(footprint.shape) - 1) // 2),
                extension_mode="constant",
                extension_constant=False,
            )
            footprint = np.logical_and(footprint, pocket_stencils)

        extremum_indices, in_footprint = _extremum_under_footprint(
            field=field.tensor,
            field_indices=field_indices,
            footprint=footprint,
            maximize=(extrema_type == "max"),
        )
        extrema_coords = field.grid.index_coordinates(extremum_indices[..., -3:])
        extrema_values = field.tensor[tuple(extremum_indices.T)]

        feats["center"] = list(extrema_coords)
        feats["center_tol"] = np.maximum(
            feats["center_tol"],
            np.linalg.norm(extrema_coords - old_centers[center_is_inside_grid], axis=-1)
        )
        feats["value"] = extrema_values
        feats = feats[in_footprint]
        refined_pharm = self.new(features=feats)
        if has_radial and keep_radial:
            refined_pharm = refined_pharm.convert_feature_vector_to_radial(
                mask=refined_pharm.features["_converted"],
                merge=True,
            )
        if has_radial:
            refined_pharm.features.drop(columns=["_converted"], inplace=True)
        return refined_pharm

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

    def convert_feature_radial_to_vector(
        self,
        distance: PositiveFloat | dict[str, PositiveFloat] = 2.0,
        feature_type: str | Sequence[str] | None = None,
        mask: pd.Series | np.ndarray | Sequence[bool] | None = None,
        merge: bool = True,
    ) -> Self | tuple[Self, Self]:
        """Convert radial features to vector features.

        This method transforms features
        represented in a radial format (i.e., with `radius` and `end` without `center`)
        into a vector format (i.e., with `center`, `radius`, and `end`).
        This is done by using the Fibonacci lattice method
        to sample quasi-uniformly distributed points
        on the radial feature's spherical surface,
        `center_distance` apart from each other.
        These points are then used as the `center`
        of new vector features with radius `center_distance / 2`,
        all pointing towards the original radial feature's `end`.

        Parameters
        ----------
        distance
            Desired distance between the centers
            of the new vector features on the spherical surface.
            This can be a single value for all feature types,
            or a dictionary mapping a feature type to its corresponding distance.
            If a dictionary is provided,
            only feature types present in the dictionary
            will be vectorized; other feature types will be ignored.
        feature_type
            Feature type(s) to convert.
            If `None`, all vector feature types are converted.
            If a string, only features of that type are converted.
            If a sequence of strings, only features of those types are converted.
        mask
            Optional boolean mask to select which features
            to consider for conversion, for more control
            alongside the `feature_type` argument.
            It can be a `pandas.Series`, a numpy array,
            or any sequence of boolean values.
            It must have the same length and order as `self.features`.
            If provided, only features where the corresponding value in `mask` is `True`
            will be considered for conversion.
        merge
            - If `True`, return a Pharmacophore instance
              containing all remaining features from `self`
              along with the newly created vector features.
            - If `False`, return a tuple of two Pharmacophore instances:
              the first containing only the newly created vector features,
              and the second containing the remaining features from `self`.

        Returns
        -------
        Either:
        - A new Pharmacophore instance like `self`, but with the selected radial features
          replaced by their vectorized representations (if `merge=True`)
        - A tuple of two Pharmacophore instances:
          the first containing only the vectorized radial features,
          and the second containing the remaining features from `self` (if `merge=False`).
        """
        def vectorize_group(group: pd.DataFrame, radius: float, feature_type: str) -> pd.DataFrame:
            centers = _sample_spherical_surface(  # shape=(N, M, 3)
                center=np.stack(group["end"]),  # shape=(N, 3)
                radius=radius,
                distance=distance[feature_type],
            )
            n_centers_per_feat = centers.shape[1]
            idx = group.index.repeat(n_centers_per_feat)  # repeat each row's index
            out = group.loc[idx].copy()
            out["center"] = list(centers.reshape(-1, 3))
            out["center_tol"] = np.maximum(distance[feature_type] / 2, group["radius_tol"])
            out["radius_tol"] = 0.0

            # Vectorized sub-labels within each (instance, label) group
            sub_labels = out.groupby(["instance", "label"], sort=False).cumcount()
            if isinstance(out["label"].iloc[0], tuple):
                out["label"] = [(*lbl, s) for lbl, s in zip(out["label"], sub_labels)]
            else:
                out["label"] = [(lbl, s) for lbl, s in zip(out["label"], sub_labels)]
            return out

        if isinstance(distance, float):
            distance = {t: distance for t in self.feature_types}

        feats_all = self._features
        feat_mask = self.is_radial & feats_all["type"].isin(distance.keys())
        if feature_type is not None:
            if isinstance(feature_type, str):
                feature_type = [feature_type]
            feat_mask = feat_mask & feats_all["type"].isin(feature_type)
        if mask is not None:
            if len(mask) != len(feats_all):
                raise ValueError("Mask length must match number of features.")
            feat_mask = feat_mask & mask
        feats_selected = feats_all[feat_mask]

        feats_grouped = feats_selected.groupby(["type", "radius"], sort=False, group_keys=False)
        feats_dfs = [
            vectorize_group(
                group=sub_df,
                radius=radius,
                feature_type=feat_type,
            ) for (feat_type, radius), sub_df in feats_grouped
        ]
        feats_vectorized = pd.concat(feats_dfs)
        feats_remaining = feats_all[~feat_mask].copy()

        if merge:
            if feats_remaining.empty:
                feats_out = feats_vectorized
            else:
                feats_remaining["label"] = feats_remaining["label"].apply(lambda x: (x, 0) if not isinstance(x, tuple) else x)
                feats_out = pd.concat([feats_remaining, feats_vectorized])
            return self.new(features=feats_out)

        pharm_vectorized = self.new(features=feats_vectorized)
        pharm_remaining = self.new(features=feats_all[~feat_mask])
        return pharm_vectorized, pharm_remaining

    def convert_feature_vector_to_radial(
        self,
        groupby: Sequence[str] = ("atom_idxs",),
        feature_type: str | Sequence[str] | None = None,
        mask: pd.Series | np.ndarray | Sequence[bool] | None = None,
        merge: bool = True,
    ) -> Self | tuple[Self, Self]:
        """Convert vector features to radial features.

        This method transforms features
        represented in a vector format (i.e., with `center`, `end`, and `radius`)
        into a radial format (i.e., with `radius` and `end` without `center`).
        This is done by grouping vector features that share the same
        `(instance, type, *groupby)` values,
        and replacing them with a single radial feature
        placed at the average of their `end` points,
        with a `radius` equal to the mean length of the vectors.

        Parameters
        ----------
        groupby
            Column names in addition to `instance` and `type` to group vector features by.
            The default is `("atom_idxs",)`, which groups by the `atom_idxs` column.
            This means that vector features that share the same
            `(instance, type, atom_idxs)` values will be grouped together
            and converted into a single radial feature.
            Note that rows with null values in these columns are
            excluded from the grouping and conversion process.
        feature_type
            Feature type(s) to convert.
            If `None`, all vector feature types are converted.
            If a string, only features of that type are converted.
            If a sequence of strings, only features of those types are converted.
        mask
            Optional boolean mask to select which features
            to consider for conversion, for more control
            alongside the `feature_type` argument.
            It can be a `pandas.Series`, a numpy array,
            or any sequence of boolean values.
            It must have the same length and order as `self.features`.
            If provided, only features where the corresponding value in `mask` is `True`
            will be considered for conversion.
        merge
            - If `True`, return a Pharmacophore instance
              containing all remaining features from `self`
              along with the newly created radial features.
            - If `False`, return a tuple of two Pharmacophore instances:
              the first containing only the newly created radial features,
              and the second containing the remaining features from `self`.

        Returns
        -------
        Either:
        - A new Pharmacophore instance like `self`, but with the selected vector features
          replaced by their radial representations (if `merge=True`)
        - A tuple of two Pharmacophore instances:
          the first containing only the radialized features,
          and the second containing the remaining features from `self` (if `merge=False`).
        """
        def radialize_group(group: pd.DataFrame) -> pd.DataFrame:
            centers = np.stack(group["center"])
            ends = np.stack(group["end"])
            lengths = np.linalg.norm(ends - centers, axis=-1)

            row = group.sort_values("label").iloc[0].copy()
            row["center"] = None
            row["center_tol"] = 0.0
            row["end"] = np.mean(ends, axis=0)
            row["radius"] = np.mean(lengths)

            ends_delta = np.linalg.norm(row["end"] - ends, axis=-1)
            idx_largest_delta = np.argmax(ends_delta)
            largest_delta = ends_delta[idx_largest_delta]
            largest_delta_end_tol = group["end_tol"].iloc[idx_largest_delta]
            row["end_tol"] = largest_delta + largest_delta_end_tol

            radii = np.linalg.norm(row["end"] - centers, axis=-1)
            radii_delta = np.abs(radii - row["radius"])
            idx_largest_delta = np.argmax(radii_delta)
            largest_delta = radii_delta[idx_largest_delta]
            largest_delta_center_tol = group["center_tol"].iloc[idx_largest_delta]
            row["radius_tol"] = largest_delta + largest_delta_center_tol

            if isinstance(row["label"], tuple):
                row["label"] = row["label"][:-1]
            return row

        feats_all = self._features
        feat_mask = self.is_vector & feats_all[list(groupby)].notnull().all(axis=1)
        if feature_type is not None:
            if isinstance(feature_type, str):
                feature_type = [feature_type]
            feat_mask = feat_mask & feats_all["type"].isin(feature_type)
        if mask is not None:
            if len(mask) != len(feat_mask):
                raise ValueError("Mask length must match number of features.")
            feat_mask = feat_mask & mask
        feats_selected = feats_all[feat_mask]

        feats_grouped = feats_selected.groupby(["instance", "type", *groupby], sort=False, group_keys=False)
        feats_rows = [radialize_group(group=sub_df) for _, sub_df in feats_grouped]
        feats_radialized = pd.DataFrame(feats_rows).drop(columns=["center"])
        feats_remaining = feats_all[~feat_mask].copy()

        if merge:
            if feats_remaining.empty:
                feats_out = feats_radialized
            else:
                feats_remaining["label"] = feats_remaining["label"].apply(lambda x: x if not isinstance(x, tuple) else x[:-1])
                feats_out = pd.concat([feats_remaining, feats_radialized])
            return self.new(features=feats_out)

        pharm_radialized = self.new(features=feats_radialized)
        pharm_remaining = self.new(features=feats_all[~feat_mask])
        return self.new(features=feats_out)

    def cluster(
        self,
        function: ClusteringFunction | dict[str, ClusteringFunction],
        function_input: Callable | Literal["center", "end"] = distmatrix_linear,
        groupby: str | Sequence[str] = "instance",
        min_members: PositiveInt | dict[str, PositiveInt] = 1,
        noise_as_singleton: bool | dict[str, bool] = True,
        keep_single_groups: bool = True,
        weights: pd.Series | np.ndarray | Sequence[float] | None = None,
        center_type: CenterType | dict[str, CenterType] = "average",
        radius_type: RadiusType | dict[str, RadiusType] = "average",
        preserve_members: bool = True,
        merge: bool = True,
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

        df = self._features.copy()
        df["_weights"] = args.weights
        if mask is not None:
            df_remaining = df[~mask]
            df = df[mask]

        has_old_members = "members" in df.columns
        new_features: list[dict] = []
        for group_idx, group in df.groupby(args.groupby, sort=False):
            feature_type = group_idx[-1]

            # If there is only one feature in the group,
            # either assign it to a single cluster or skip it
            if centers.shape[0] == 1:
                if not keep_single_groups:
                    continue
                clustering_result = None
                labels = np.array([0], dtype=np.int32)

            # If the function input must be a distance matrix
            elif callable(function_input):
                clustering_result = function[feature_type](
                    function_input(np.stack(group[function_input].to_numpy()))[:,:,-1],
                )
                labels = np.asarray(clustering_result.labels)

            # If the function input is either "center" or "end" coordinates
            elif function_input in ("center", "end"):
                # Drop rows with NaN in the specified input column
                group = group.dropna(subset=[function_input])
                clustering_result = function[feature_type](
                    np.stack(group[function_input].to_numpy()),
                    group["_weights"].to_numpy(dtype=np.float64)
                )
                labels = np.asarray(clustering_result.labels)
            else:
                raise ValueError(f"Invalid function_input: {function_input}")

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
        feature_mask: pd.Series | np.ndarray | Sequence[bool] | None = None,
        system: System | Literal[False] | None = None,
        min_radius: float = 0.3,
        show_box: bool = True,
        show_pocket: bool = True,
        show_fields: bool = False,
        field_only_in_pocket: bool = True,
        show_feature_centers: bool = True,
        feature_colors: dict[str, tuple[float, float, float] | tuple[int, int, int]] | None = None,
        override_radius: dict[str, float] | None = None,
        gui: bool = True,
        comp_point_feats: set[Literal["center"]] = {"center"},
        comp_vector_feats: set[Literal["vector", "end", "cone"]] = {"vector", "end", "cone"},
        comp_radial_feats: set[Literal["end", "surface", "surface_min", "surface_max"]] = {"end", "surface", "surface_min", "surface_max"},
        add_residues: bool = True,
        feature_sort_columns: Sequence[str] = ("instance", "type", "label"),
        vector_thickness: float = 0.25,
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
        override_radius = override_radius or {}

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
        features = self.features
        if feature_mask is not None:
            features = features[feature_mask]
        for _, feature in features.sort_values(list(feature_sort_columns)).iterrows():
            repr_type = feature["repr"]
            instance = normalize_name(feature["instance"])
            ftype = normalize_name(feature["type"])
            label = normalize_name(feature["label"])
            name = f"{instance}_{ftype}_{label}"
            color = feature_color(feature["type"])
            feat_is_point = repr_type == 1
            feat_is_vector = repr_type == 2
            feat_is_radial = repr_type == 3
            center = feature["center"].tolist() if feat_is_point or feat_is_vector else None
            end = feature["end"].tolist() if feat_is_vector or feat_is_radial else None
            radius = feature["radius"] if feat_is_vector or feat_is_radial else None

            # Display feature's interacting residue as ball-and-stick
            if atoms is not None and add_residues and pd.notna(feature["atom_idxs"]):
                residues = atoms[
                    atoms["atom_idx"].isin(feature["atom_idxs"])
                ][["res_seq", "i_code", "chain_id"]].drop_duplicates()
                receptor_selection = residues["res_seq"].astype(str) + "^" + residues["i_code"] + ":" + residues["chain_id"]
                system_comp.add_ball_and_stick(" ".join(receptor_selection), name=f"{name} Residue")

            # Display point features
            if feat_is_point:
                if "center" in comp_point_feats:
                    nv.add_spheres(
                        coords=feature["center"],
                        radii=override_radius.get(feature["type"], max(feature["center_tol"], min_radius)),
                        name=name,
                        colors=color,
                        representation_params=scishow.nglview.RepresentationParameters(
                            opacity=0.8,
                            visible=show_feature_centers,
                            lazy=True,
                        )
                    )

            # Display vector features
            elif feat_is_vector:
                shapes = []
                if "vector" in comp_vector_feats:
                    shapes.append(('Arrow', center, end, color, vector_thickness, "Vector"))
                if "end" in comp_vector_feats:
                    shapes.append(('Sphere', end, color, override_radius.get(feature["type"], max(feature["end_tol"], min_radius)), "End"))
                if "cone" in comp_vector_feats:
                    angle_tol = feature["angle_tol"]
                    if angle_tol > 0:
                        cone = nv.add_spherical_conical_shell(
                            apex=end,
                            axis=feature["center"] - feature["end"],
                            angle=angle_tol,
                            r_min=radius - feature["radius_tol"],
                            r_max=radius + feature["radius_tol"],
                            color=color,
                            name="Cone",
                            representation_params=scishow.nglview.RepresentationParameters(
                                opacity=0.6,
                                visible=True,
                                lazy=True,
                            ),
                            add=False,
                        )
                        shapes.append(cone)
                if shapes:
                    nv.add_shape(
                        shapes,
                        name=name,
                        representation_params=scishow.nglview.RepresentationParameters(
                            opacity=0.6,
                            visible=True,
                            lazy=True,
                        )
                    )

            # Display surface of radial feature
            elif feat_is_radial:
                shapes = []
                if "end" in comp_radial_feats:
                    shapes.append(('Sphere', end, color, override_radius.get(feature["type"], max(feature["end_tol"], min_radius)), "End"))
                if "surface_min" in comp_radial_feats:
                    shapes.append(('Sphere', end, color, max(radius - feature["radius_tol"], min_radius), "Surface min"))
                if "surface" in comp_radial_feats:
                    shapes.append(('Sphere', end, color, radius, "Surface"))
                if "surface_max" in comp_radial_feats:
                    shapes.append(('Sphere', end, color, radius + feature["radius_tol"], "Surface max"))
                if shapes:
                    nv.add_shape(
                        shapes,
                        name=name,
                        representation_params=scishow.nglview.RepresentationParameters(
                            opacity=0.6,
                            visible=True,
                            lazy=True,
                        )
                    )
            else:
                raise ValueError(f"Unknown feature representation type: {repr_type}")
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
        if pocket.batch_ndim == 0:
            return
        if pocket.batch_ndim != self._batch_shape.size:
            raise ValueError(
                f"Instance dimensions of the pharmacophore ({self._batch_shape}) "
                f"and the pocket ({pocket.batch_ndim}) do not match."
            )
        if (pocket.batch_shape < self._batch_shape).any():
            raise ValueError(
                f"The pocket has fewer instances ({pocket.batch_shape}) "
                f"than the pharmacophore ({self._batch_shape})."
            )
        return

    @property
    def has_point(self) -> bool:
        """Whether the pharmacophore has any point-like features."""
        return self.is_point.any()

    @property
    def has_vector(self) -> bool:
        """Whether the pharmacophore has any vector-like features."""
        return self.is_vector.any()

    @property
    def has_radial(self) -> bool:
        """Whether the pharmacophore has any radial features."""
        return self.is_radial.any()

    @property
    def is_point(self) -> pd.Series:
        """Boolean Series indicating which features are point-like."""
        return self.features['repr'] == 1

    @property
    def is_vector(self) -> pd.Series:
        """Boolean Series indicating which features are vector-like."""
        return self.features['repr'] == 2

    @property
    def is_radial(self) -> pd.Series:
        """Boolean Series indicating which features are radial."""
        return self.features['repr'] == 3

def merge(
    pharmacophores: Sequence[Pharmacophore],
    name: str = "Merged Pharmacophore",
    system: System | int | None = 0,
    pocket: Pocket | int | None = 0,
    field: Field | int | None = 0,
) -> Pharmacophore:
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
    for idx, pharm in enumerate(pharmacophores):
        feats = pharm.features.copy()
        feats["instance"] = (
            feats["instance"].apply(instance_merger, instance_prefix=idx)
        )
        feats["merge_origin"] = pharm.name
        dfs.append(feats)

    merged_features = pd.concat(dfs)
    feature_types = set().union(*(ph.feature_types for ph in pharmacophores))

    return Pharmacophore(
        features=merged_features,
        feature_types=feature_types,
        system=system if not isinstance(system, int) else pharmacophores[system].system,
        pocket=pocket if not isinstance(pocket, int) else pharmacophores[pocket].pocket,
        field=field if not isinstance(field, int) else pharmacophores[field].field,
        name=name,
    )


def _extremum_under_footprint(
    field: np.ndarray,
    field_indices: np.ndarray,
    footprint: np.ndarray,
    *,
    maximize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
        This can also be a 4D array, where the first axis contains one 3D footprint per
        given field index.


    Returns
    -------
    extremum_indices
        Array of shape (K, N) with the global indices (same order as `field_indices`)
        of the selected extreme (min or max) element in `field` under the footprint
        for each placement.
    in_footprint
        Boolean array of shape (K,) indicating the points for which no element
        under the footprint was found (i.e., footprint was completely out of bounds or all False).
        For these points, the corresponding row in `extremum_indices` is simply a copy
        of the input `field_indices` row.

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
    if footprint.ndim not in (3, 4):
        raise ValueError(f"`footprint` must be 3D or 4D; got {footprint.ndim}D")
    if footprint.ndim == 4 and footprint.shape[0] != field_indices.shape[0]:
        raise ValueError(
            f"If `footprint` is 4D, its first dimension must match the number of rows in `field_indices` "
            f"({field_indices.shape[0]}); got {footprint.shape[0]}"
        )
    if any(s % 2 == 0 for s in footprint.shape[-3:]):
        raise ValueError(f"`footprint` must have odd lengths on last three axes; got shape {footprint.shape}")
    if field_indices.ndim != 2 or field_indices.shape[1] != field.ndim:
        raise ValueError(
            f"`field_indices` must have shape (K, {field.ndim}); got {field_indices.shape}"
        )
    if not np.issubdtype(field_indices.dtype, np.integer):
        raise ValueError("`field_indices` must be of integer dtype")

    N = field.ndim
    K = field_indices.shape[0]
    extremum_indices = np.empty((K, N), dtype=np.int64)
    in_footprint = np.ones((K,), dtype=bool)

    # Radii (half-sizes) of the footprint along its 3 axes
    rad_z, rad_y, rad_x = (d // 2 for d in footprint.shape[-3:])

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

        footprint_k = footprint[k] if footprint.ndim == 4 else footprint
        pview = footprint_k[pz, py, px]

        # Ensure pview has any True
        if not pview.any():
            # Fallback to center (should be rare if center of footprint is True)
            extremum_indices[k] = idx
            in_footprint[k] = False
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
            extremum_indices[k, :-3] = np.array(lead_idx, dtype=np.int64)
        extremum_indices[k, -3:] = (gz, gy, gx)

    return extremum_indices


def _sample_spherical_surface(
    center: np.ndarray,
    radius: float,
    distance: float,
) -> np.ndarray:
    """Sample quasi-uniform points on the surface of a sphere.

    This function uses the Fibonacci lattice method
    to generate coordinates for nearly uniform points on the
    surface of a sphere with given `center` and `radius`.
    The number of points is chosen based on the surface area
    of the sphere and the requested point spacing `distance`.

    Parameters
    ----------
    center
        An array of shape (..., 3) giving the coordinates of the center point(s).
        Broadcasting is supported: multiple centers will produce one set of points per center.
    radius
        The radius of the sphere.
    distance
        Approximate geodesic spacing between sampled points on
        the sphere surface.

    Returns
    -------
    An array of shape (..., M, 3) giving the coordinates of the sampled points,
    where M is the number of points sampled on the sphere's surface.

    Notes
    -----
    - This uses a quasi-uniform Fibonacci sphere distribution,
      which avoids clustering at the poles compared to spherical
      grids.
    - The actual spacing may not be exactly `distance`, but is
      close on average across the sphere.
    """
    center = np.asarray(center, dtype=float)

    if center.shape[-1] != 3:
        raise ValueError("`center` must have shape (..., 3)")

    # --- Estimate number of points ---
    # Surface area of sphere: 4πr²
    surface_area = 4.0 * np.pi * radius**2
    # Area per point ~ distance², so N ~ surface_area / distance²
    n_points = max(1, int(np.round(surface_area / (distance**2))))

    # --- Fibonacci sphere algorithm ---
    # Golden angle increment
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.4

    # Indices
    idx = np.arange(n_points)

    # y-coordinates uniformly spaced in [-1, 1]
    y = 1.0 - 2.0 * (idx + 0.5) / n_points
    r_xy = np.sqrt(1.0 - y**2)

    theta = golden_angle * idx
    x = r_xy * np.cos(theta)
    z = r_xy * np.sin(theta)

    # Unit vectors on sphere surface
    unit_points = np.stack((x, y, z), axis=-1)

    # Scale to radius
    sphere_points = radius * unit_points  # (M, 3)

    # Broadcast to each center: (..., M, 3)
    # Add newaxis at -2 to match sphere_points
    result = center[..., None, :] + sphere_points
    return result


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
