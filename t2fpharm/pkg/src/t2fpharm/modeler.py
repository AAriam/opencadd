"""Pharmacophore modeler for target-focused pharmacophore perception."""

from typing import Sequence, Literal, Callable, Any

import pandas as pd
import numpy as np
from pydantic import BaseModel

import caddpy

from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.pharm import Pharmacophore
from t2fpharm.system import System
from t2fpharm.modeler_structure import StructureBasedModeler
from t2fpharm.input.modeler import ModelerLargestPeaksInput, ModelerSimpleInput
from t2fpharm.cluster import ClusterAggLinkageType, ClusterAggLinkageMetricType
from t2fpharm.typing import PositiveInt, PositiveFloat


__all__ = [
    "Modeler",
]

FilterExtensionMode = Literal["constant", "nearest", "mirror", "reflect", "wrap"]
FilterFunction = Literal["gaussian", "mean", "percentile"] | Callable


class Modeler:
    """Target-focused pharmacophore modeler.

    This class provides methods to perceive pharmacophore features
    from a field tensor, optionally using a binding pocket mask.

    Parameters
    ----------
    field
        The field tensor from which to perceive pharmacophore features.
    pocket
        An optional binding pocket mask to restrict the perception to a specific region.
    system
        Optional chemical system associated with the pharmacophore.
        This is not used by the modeler itself.
        If provided, it is only used by the `display()` method
        of the generated Pharmacophore to visualize the pharmacophore
        in the context of the chemical structure.
        This can be any object that can be visualized by NGLView
        using its `add_trajectory()` method.
    """
    def __init__(
        self,
        system: System | None = None,
        pocket: Pocket | None = None,
        field: Field | None = None,
    ):
        self._field: Field | None = None
        self._pocket: Pocket | None = None

        self.field = field
        self.pocket = pocket
        if system:
            self.system = system
        elif pocket is not None:
            self.system = pocket.receptor

        self._structure_modeler = None
        return

    @property
    def field(self) -> Field:
        return self._field

    @field.setter
    def field(self, value: Field | None):
        if value is None:
            self._field = None
            return
        if not isinstance(value, Field):
            raise TypeError(f"Expected Field object, got {type(value).__name__}.")
        self._field = value
        if self.pocket is not None:
            self._verify_field_pocket_compatible()
        return

    @property
    def pocket(self) -> Pocket | None:
        return self._pocket

    @pocket.setter
    def pocket(self, pocket: Pocket | None):
        if pocket is None:
            self._pocket = None
            return
        if not isinstance(pocket, Pocket):
            raise TypeError(f"Expected Pocket object, got {type(pocket).__name__}.")
        self._pocket = pocket
        if self.field is not None:
            self._verify_field_pocket_compatible()
        return

    @property
    def system(self) -> System | None:
        return self._system

    @system.setter
    def system(self, value: System | None):
        if value is None:
            self._system = None
            return
        if not isinstance(value, System):
            raise TypeError(f"Expected System object, got {type(value).__name__}.")
        self._system = value
        return

    def from_field_agg(
        self,
        *,
        distance_threshold: PositiveFloat | dict[str, PositiveFloat] | None = None,
        n_clusters: PositiveInt | dict[str, PositiveInt] | None = None,
        linkage: ClusterAggLinkageType | dict[str, ClusterAggLinkageType] = "complete",
        metric: ClusterAggLinkageMetricType | dict[str, ClusterAggLinkageMetricType] = "euclidean",
        memory: Any = None,
        min_members: PositiveInt | dict[str, PositiveInt] = 1,
        noise_as_singleton: bool | dict[str, bool] = True,
        center_type: Literal["function", "midpoint", "mean", "average"] | dict[str, Literal["function", "midpoint", "mean", "average"]] = "average",
        radius_type: Literal["average", "mean", "max", "min"] = "max",
        # Parameters for `self.from_field`
        filter_function: FilterFunction | dict[str, FilterFunction] | None = None,
        filter_radius: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_extension_mode: FilterExtensionMode | dict[str, FilterExtensionMode] = "constant",
        filter_extension_constant_value: float | dict[str, float] = 0,
        filter_gaussian_sigma: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_percentile: float | dict[str, float] = 50,
        peak_type: Literal["min", "max"] | dict[str, Literal["min", "max"]] = "min",
        best_per_point: bool | dict[str, bool] = False,
        threshold_value: float | dict[str, float] | None = None,
        threshold_percentile: float | dict[str, float] | None = None,
        threshold_include_equal: bool | dict[str, bool] = False,
    ) -> Pharmacophore:
        """Perceive pharmacophore features using a hierarchical agglomerative clustering algorithm.

        This method is equivalent to calling `Modeler.simple()`
        to create an initial pharmacophore,
        followed by calling `Pharmacophore.cluster_agg()`
        to cluster the feature centers.
        For more information on the algorithm and parameters,
        see the documentation of those methods.
        """
        self._verify_field_available("from_field_agg")
        pharm = self.from_field(
            filter_function=filter_function,
            filter_radius=filter_radius,
            filter_extension_mode=filter_extension_mode,
            filter_extension_constant_value=filter_extension_constant_value,
            filter_gaussian_sigma=filter_gaussian_sigma,
            filter_percentile=filter_percentile,
            peak_type=peak_type,
            best_per_point=best_per_point,
            threshold_value=threshold_value,
            threshold_percentile=threshold_percentile,
            threshold_include_equal=threshold_include_equal,
        )

        weights = pharm.features["value"].copy()
        # Make sure all weights are either >= 0 or <= 0
        for feature_type in pharm.features["type"].unique():
            type_mask = pharm.features["type"] == feature_type
            feature_weights = weights[type_mask]
            ge0 = feature_weights.ge(0)
            le0 = feature_weights.le(0)
            if not (ge0.all() or le0.all()):
                feature_peak_type = pharm.inputs[0]["peak_type"][feature_type]
                indices = ge0[ge0].index if feature_peak_type == "min" else le0[le0].index
                weights.loc[indices] = 0
        return pharm.cluster_agg(
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
            per_instance=True,
        )

    def from_field_cnn(
        self,
        *,
        max_distance: PositiveFloat | Sequence[PositiveFloat] | dict[str, PositiveFloat | Sequence[PositiveFloat]],
        min_neighbors: PositiveInt | Sequence[PositiveInt] | dict[str, PositiveInt | Sequence[PositiveInt]],
        min_members: PositiveInt | dict[str, PositiveInt] = 1,
        max_members: PositiveInt | dict[str, PositiveInt] | None = None,
        noise_as_singleton: bool | dict[str, bool] = True,
        center_type: Literal["function", "midpoint", "mean", "average"] | dict[str, Literal["function", "midpoint", "mean", "average"]] = "average",
        radius_type: Literal["average", "mean", "max", "min"] = "max",
        # Parameters for `self.from_field`
        filter_function: FilterFunction | dict[str, FilterFunction] | None = None,
        filter_radius: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_extension_mode: FilterExtensionMode | dict[str, FilterExtensionMode] = "constant",
        filter_extension_constant_value: float | dict[str, float] = 0,
        filter_gaussian_sigma: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_percentile: float | dict[str, float] = 50,
        peak_type: Literal["min", "max"] | dict[str, Literal["min", "max"]] = "min",
        best_per_point: bool | dict[str, bool] = False,
        threshold_value: float | dict[str, float] | None = None,
        threshold_percentile: float | dict[str, float] | None = None,
        threshold_include_equal: bool | dict[str, bool] = False,
    ) -> Pharmacophore:
        """Perceive pharmacophore features using the Common Nearest Neighbors (CNN) clustering algorithm.

        This method is equivalent to calling `Modeler.simple()`
        to create an initial pharmacophore,
        followed by calling `Pharmacophore.cluster_cnn()`
        to cluster the feature centers.
        For more information on the algorithm and parameters,
        see the documentation of those methods.
        """
        self._verify_field_available("from_field_cnn")
        pharm = self.from_field(
            filter_function=filter_function,
            filter_radius=filter_radius,
            filter_extension_mode=filter_extension_mode,
            filter_extension_constant_value=filter_extension_constant_value,
            filter_gaussian_sigma=filter_gaussian_sigma,
            filter_percentile=filter_percentile,
            peak_type=peak_type,
            best_per_point=best_per_point,
            threshold_value=threshold_value,
            threshold_percentile=threshold_percentile,
            threshold_include_equal=threshold_include_equal,
        )

        # if max_distance is None:
        #     # As default, include all 26 neighbors in a 3D grid
        #     # plus orthogonal second neighbors (i.e., 26 + 6 = 32 neighbors)
        #     max_distance = self.field.grid.spacings[0] * 2.1
        # if min_members is None:
        #     hydrogen_radius = 1.2
        #     hydrogen_volume = (4/3) * np.pi * hydrogen_radius**3
        #     half_hydrogen_volume = hydrogen_volume / 2
        #     voxel_volume = self.field.grid.point_volume
        #     min_members = int(np.ceil(half_hydrogen_volume / voxel_volume))
        # if max_members is None:
        #     max_members = min_members * 5 if isinstance(min_members, int) else [
        #         min_member * 5 for min_member in min_members
        #     ]

        weights = pharm.features["value"].copy()
        # Make sure all weights are either >= 0 or <= 0
        for feature_type in pharm.features["type"].unique():
            type_mask = pharm.features["type"] == feature_type
            feature_weights = weights[type_mask]
            ge0 = feature_weights.ge(0)
            le0 = feature_weights.le(0)
            if not (ge0.all() or le0.all()):
                feature_peak_type = pharm.inputs[0]["peak_type"][feature_type]
                indices = ge0[ge0].index if feature_peak_type == "min" else le0[le0].index
                weights.loc[indices] = 0
        return pharm.cluster_cnn(
            max_distance=max_distance,
            min_neighbors=min_neighbors,
            min_members=min_members,
            max_members=max_members,
            noise_as_singleton=noise_as_singleton,
            weights=weights,
            center_type=center_type,
            radius_type=radius_type,
            per_instance=True,
        )

    def from_field_extrema(
        self,
        *,
        min_distance: PositiveFloat | dict[tuple[str, str], PositiveFloat],
        priority_value_filter: bool = True,
        priority_factor: dict[str, float] | None = None,
        max_features: PositiveInt | dict[str, PositiveInt] | None = None,
        # Parameters for `self.from_field`
        filter_function: FilterFunction | dict[str, FilterFunction] | None = None,
        filter_radius: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_extension_mode: FilterExtensionMode | dict[str, FilterExtensionMode] = "constant",
        filter_extension_constant_value: float | dict[str, float] = 0,
        filter_gaussian_sigma: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_percentile: float | dict[str, float] = 50,
        peak_type: Literal["min", "max"] | dict[str, Literal["min", "max"]] = "min",
        best_per_point: bool | dict[str, bool] = False,
        threshold_value: float | dict[str, float] | None = None,
        threshold_percentile: float | dict[str, float] | None = None,
        threshold_include_equal: bool | dict[str, bool] = False,
    ) -> Pharmacophore:
        """Perceive pharmacophore features as largest extrema in the fields.

        This method is equivalent to calling `Modeler.simple()`
        to create an initial pharmacophore,
        followed by calling `Pharmacophore.remove_overlaps()`
        to filter the features based on their distance.
        For more information on the algorithm and parameters,
        see the documentation of those methods.

        Parameters
        ----------
        priority_value_filter
            - `True`: Use the filtered field values for the priority calculation.
            - `False`: Use the original field values before applying the filter function.
        priority_factor
            Optional dictionary mapping feature types to their priority factors.
            If provided, the field values (as defined by `priority_value_filter`)
            of each feature type are multiplied by the corresponding factor
            to create the `priority` parameter of `Pharmacophore.remove_overlaps()`.
            The factors must be chosen such that the highest priority is the lowest value.
            This is useful when the field values of different feature types
            do not all have the same scale/sign, or when you want to add bias
            to the feature selection process to favor certain types over others.
        """
        self._verify_field_available("from_field_extrema")
        pharm = self.from_field(
            filter_function=filter_function,
            filter_radius=filter_radius,
            filter_extension_mode=filter_extension_mode,
            filter_extension_constant_value=filter_extension_constant_value,
            filter_gaussian_sigma=filter_gaussian_sigma,
            filter_percentile=filter_percentile,
            peak_type=peak_type,
            best_per_point=best_per_point,
            threshold_value=threshold_value,
            threshold_percentile=threshold_percentile,
            threshold_include_equal=threshold_include_equal,
        )

        priority_factor = ModelerLargestPeaksInput(
            priority_factor=priority_factor,
            feature_types=self.field.batch_instance_labels["feature"],
        ).priority_factor
        priority = pharm.features["value_filter" if priority_value_filter else "value"].copy()
        for feature_type, factor in priority_factor.items():
            if factor is not None:
                priority.loc[pharm.features["type"] == feature_type] *= factor
        return pharm.remove_overlaps(
            min_distance=min_distance,
            priority=priority,
            highest_priority="lowest",
            max_features=max_features,
        )

    def from_field(
        self,
        *,
        filter_function: FilterFunction | dict[str, FilterFunction] | None = None,
        filter_radius: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_extension_mode: FilterExtensionMode | dict[str, FilterExtensionMode] = "constant",
        filter_extension_constant_value: float | dict[str, float] = 0,
        filter_gaussian_sigma: PositiveFloat | dict[str, PositiveFloat] | None = None,
        filter_percentile: float | dict[str, float] = 50,
        peak_type: Literal["min", "max"] | dict[str, Literal["min", "max"]] = "min",
        best_per_point: bool | dict[str, bool] = False,
        threshold_value: float | dict[str, float] | None = None,
        threshold_percentile: float | dict[str, float] | None = None,
        threshold_include_equal: bool | dict[str, bool] = True,
    ) -> Pharmacophore:
        """Perceive pharmacophore features from the field tensor.

        All parameters can be specified as a single value for all feature types,
        or as a dictionary mapping feature types to their respective values.

        Parameters
        ----------
        filter_function
            Optional filter function to apply to the field values
            before perceiving pharmacophore features.
            The function can be either a generic callable object
            or the name of one of the predefined filter functions.
            If a callable is provided, it must accept a single argument,
            which is a 1D array of field values
            within a sphere of radius `filter_radius` centered at each grid point.
            The function must then return a single value
            as the replacement value for that grid point.
            For example, passing `numpy.mean` would have the same effect
            as the predefined "mean" filter function described below.
            The predefined filter functions are:
            - "gaussian": Apply a [Gaussian filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)
              with `filter_gaussian_sigma` as the standard deviation
              and `filter_radius` as the radius of the Gaussian kernel.
              This performs a spherical smoothing of the field values,
              where the value at each grid point is replaced by the weighted average
              of its neighbors within the specified radius, with weights
              determined by the Gaussian function.
            - "mean": Apply a mean filter with a spherical footprint of radius `filter_radius`.
              This is similar to a Gaussian filter but uses a uniform weight for all neighbors.
            - "percentile": Apply a [percentile filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html)
              with `filter_percentile` as the percentile to compute.
              This replaces each grid point value with the specified percentile
              of the values within a sphere of radius `filter_radius`.
        filter_radius
            Radius of the spherical footprint/kernel
            for the filter function.
            This must be defined for each feature type
            that has a `filter_function` defined.
        filter_extension_mode
            Mode for extending the field tensor
            for when the filter overlaps a border.
            Available modes are:
            - "constant": Fields are extended by filling all values beyond the borders
              with the same constant value defined by the parameter
              `filter_extension_constant_value` (k k k k | a b c d | k k k k).
            - "nearest": Fields are extended by repeating the nearest border value (a a a a | a b c d | d d d d).
            - "mirror": Fields are extended by mirroring the values at the borders,
               with the mirror plane placed at the border value (d c b | a b c d | c b a).
            - "reflect": Fields are extended by mirroring the values at the borders,
              with the mirror plane placed after the border value (d c b a | a b c d | d c b a)
            - "wrap": Fields are extended by wrapping the values around the opposite border (a b c d | a b c d | a b c d).
        filter_extension_constant_value
            Constant value to use when `filter_extension_mode` is "constant".
            This value is used to fill the extended borders of the field tensor.
        filter_gaussian_sigma
            Standard deviation of the Gaussian kernel to use
            when `filter_function` is "gaussian".
        filter_percentile
            Percentile to compute when `filter_function` is "percentile".
            This value must be between 0 and 100.
        peak_type
            Type of peaks to search for in the field tensor.
            - "min": Best values are minima.
            - "max": Best values are maxima.
        best_per_point
            If `True`, discard grid points
            where the field value is not the best value
            (i.e., lowest for "min" `peak_type` or highest for "max" `peak_type`)
            among all feature types at that grid point.
        threshold_value
            Threshold value for the peaks.
            - When `peak_type` is "min", only minima with values
              less than (or equal to) this threshold are considered.
            - When `peak_type` is "max", only maxima with values
              greater than (or equal to) this threshold are considered.
            - If a value is `None`, no thresholding is applied.
        threshold_percentile
            If provided, only the best `threshold_percentile` percent
            of the remaining grid points after applying the `filter_function`,
            `best_per_point`, `threshold_value`, and pocket mask are kept.
        threshold_include_equal
            Whether to include peaks with values equal to the threshold.

        Notes
        -----
        The algorithm works as follows:
        For each feature type in the field,
        1. If `filter_function` is provided,
           for each grid point the field value is replaced
           by the value of the `filter_function`
           applied to the field values within a sphere
           of radius `filter_radius` centered at the grid point.
           This can be used to smooth the field
           (e.g., using a Gaussian or percentile filter),
           or to apply a custom transformation to the field values.
        2. If `best_per_point` is `True`,
           grid points whose field value (after applying the filter function, if any)
           is not the best (i.e. lowest for "min" peaks or highest for "max" peaks)
           among all feature type field values at that grid point are discarded.
           This prevents selecting multiple feature types at the exact same grid point.
        3. If `threshold_value` is provided,
           grid points whose field values
           (after applying the filter function, if any)
           do not meet the threshold are discarded.
           This is useful to filter out noise
           and reduce the number of features.
        4. If the modeler has a pocket defined,
           grid points outside the pocket are discarded.
           This ensures that only features
           within the pocket are considered.
        5. If `threshold_percentile` is provided,
           only the best `threshold_percentile` percent
           of the remaining grid points
           are kept for each feature type.
        6. The remaining grid points are taken as pharmacophore feature centers
           and used to create a `Pharmacophore` object.
        """
        self._verify_field_available("from_field")
        kwargs = locals()
        del kwargs["self"]
        args = ModelerSimpleInput(
            **kwargs,
            feature_types=self.field.batch_instance_labels["feature"],
            grid=self.field.grid,
        )
        tensor = self._field_filter(args.filter_function)
        mask = self._field_mask(
            tensor=tensor,
            peak_type=args.peak_type,
            best_per_point=args.best_per_point,
            threshold_value=args.threshold_value,
            threshold_percentile=args.threshold_percentile,
            threshold_include_equal=args.threshold_include_equal,
        )
        feature_radius = {
            feature_type: radius or self.field.grid.spacings[0] / 2
            for feature_type, radius in args.filter_radius.items()
        }

        indices = np.argwhere(mask)
        grid_indices = indices[:, -3:]
        coordinates = self.field.grid.index_coordinates(grid_indices)
        instances_ndim = indices.shape[1] - 4
        if instances_ndim == 0:
            instance_indices = np.zeros(indices.shape[0], dtype=int)
        elif instances_ndim == 1:
            instance_indices = indices[:, 1]
        else:
            instance_indices = list(map(tuple, indices[:, 1:-3].tolist()))
        types = [self._feature_type(idx) for idx in indices[:, 0]]
        radii = [feature_radius[feature_type] for feature_type in types]
        features = pd.DataFrame(
            {
                "instance": instance_indices,
                "type": types,
                "label":list(map(tuple, grid_indices.tolist())),
                "center": list(coordinates),
                "center_tol": radii,
                "value": self.field.tensor[mask],
                "value_filter": tensor[mask],
            }
        )
        return self._pharmacophore(
            features=features,
            feature_types=set(self.field.batch_instance_labels["feature"]),
            inputs=[args.model_dump()],
            extra={"mask": mask},
        )

    def from_interactions(
        self,
        *,
        type_hbond_acceptor: str | None = "OA",
        type_hbond_donor: str | None = "HD",
        type_anionic: str | None = "e-",
        type_cationic: str | None = "e+",
        type_hydrophobic: str | None = "C",
        type_aromatic: str | None = "A",
        type_halogen_acceptor: str | None = "XA",
        type_halogen_donor: str | None = "XD",
        atom_mask: pd.Series | np.ndarray | Sequence[bool] | None = None,
        name: str | None = None,
    ):
        """Create a pharmacophore from a receptor–ligand complex.

        This method uses the PLIP library to analyze the interactions
        between the receptor and ligand(s) in the provided complex.
        For each detected interaction,
        two features are created: one for each interaction partner.
        Note that if any of the `type_*` parameters are set to `None`,
        the corresponding feature type will not be included in the pharmacophore.

        Parameters
        ----------
        type_hbond_acceptor
            Feature type ID for hydrogen bond acceptor features.
        type_hbond_donor
            Feature type ID for hydrogen bond donor features.
        type_anionic
            Feature type ID for anionic features.
        type_cationic
            Feature type ID for cationic features.
        type_hydrophobic
            Feature type ID for hydrophobic features.
        type_aromatic
            Feature type ID for aromatic features.
        type_halogen_acceptor
            Feature type ID for halogen bond acceptor features.
        type_halogen_donor
            Feature type ID for halogen bond donor features.
        atom_mask
            Optional boolean mask to select a subset of atoms
            from the system for pharmacophore feature perception.
            If provided, only atoms in `self.system.composition.atoms`
            where the mask is `True` will be considered when perceiving features.
        name
            Optional name for the pharmacophore.
        """
        self._verify_system_available("from_interactions")
        plip = caddpy.interaction.from_chemsys(self.system, add_polar_hydrogens=False)
        feats = plip.all.rename(columns={"type": "interaction"})

        r_is_donor = feats["r_is_d"].fillna(False).astype(bool) if "r_is_d" in feats else pd.Series(False, index=feats.index)
        r_is_cation = feats["r_is_cation"].fillna(False).astype(bool) if "r_is_cation" in feats else pd.Series(False, index=feats.index)
        r_is_acceptor = ~r_is_donor
        r_is_anion = ~r_is_cation

        interaction_df = {
            name: feats[(feats["interaction"] == interaction_type) & condition]
            for name, interaction_type, condition in (
                ("hbond_acceptor",    "hbond", r_is_acceptor),
                ("hbond_donor",       "hbond", r_is_donor),
                ("wbridge_acceptor",  "water_bridge", r_is_acceptor),
                ("wbridge_donor",     "water_bridge", r_is_donor),
                ("saltbridge_anion",  "salt_bridge", r_is_anion),
                ("saltbridge_cation", "salt_bridge", r_is_cation),
                ("hydrophobic",       "hydrophobic", True),
                ("aromatic",          "pi_stacking", True),
                ("pication_aromatic", "pi_cation", r_is_anion),
                ("pication_cation",   "pi_cation", r_is_cation),
                ("halogen_acceptor",  "halogen", r_is_acceptor),
            )
        }

        atom = self.system.composition.atoms
        if atom_mask is not None:
            atom = atom[atom_mask]
        atom_serial_to_idx = dict(zip(atom["serial"], atom["atom_idx"]))
        output_col_names = ["type", "center", "end", "res_idx", "atom_idxs", "interaction"]
        feat_dfs = []
        for df_name,              center, end,   res,  serial, res_rev, serial_rev, feat,                feat_rev in (
            ("hbond_acceptor",    "h",    "r",   "r",  "r",    "l",     "l",        type_hbond_donor,    type_hbond_acceptor),
            ("hbond_donor",       "l",    "h",   "r",  "r",    "l",     "l",        type_hbond_acceptor, type_hbond_donor),
            ("wbridge_acceptor",  "h",    "w_o", "w",  "w_o",  "l",     "l",        type_hbond_donor,    type_hbond_acceptor),
            ("wbridge_acceptor",  "w_h",  "r",   "r",  "r",    "w",     "w_h",      type_hbond_donor,    type_hbond_acceptor),
            ("wbridge_donor",     "l",    "w_h", "w",  "w_h",  "l",     "l",        type_hbond_acceptor, type_hbond_donor),
            ("wbridge_donor",     "w_o",  "h",   "r",  "r",    "w",     "w_o",      type_hbond_acceptor, type_hbond_donor),
            ("saltbridge_anion",  "l",    "r",   "r",  "r",    "l",     "l",        type_cationic,         type_anionic),
            ("saltbridge_cation", "l",    "r",   "r",  "r",    "l",     "l",        type_anionic,          type_cationic),
            ("hydrophobic",       "l",    "r",   "r",  "r",    "l",     "l",        type_hydrophobic,    type_hydrophobic),
            ("aromatic",          "l",    "r",   "r",  "r",    "l",     "l",        type_aromatic,       type_aromatic),
            ("pication_aromatic", "l",    "r",   "r",  "r",    "l",     "l",        type_cationic,         type_aromatic),
            ("pication_cation",   "l",    "r",   "r",  "r",    "l",     "l",        type_aromatic,       type_cationic),
            ("halogen_acceptor",  "l",    "r",   "r",  "r",    "l",     "l",        type_halogen_donor,  type_halogen_acceptor),
        ):
            df = interaction_df[df_name]
            if df.empty:
                continue
            for c, e, r, s, f in (
                (center, end, res, serial, feat),
                (end, center, res_rev, serial_rev, feat_rev),
            ):
                if f is None:
                    continue
                atom_idxs = df[f"{s}_serials"].apply(
                    lambda arr: tuple(
                        atom_serial_to_idx[s] for s in arr
                        if atom_serial_to_idx.get(s) is not None
                    )
                )
                mask = atom_idxs.apply(lambda arr: len(arr) > 0)
                dff = df[mask].copy()
                if dff.empty:
                    continue
                dff["type"] = f
                dff["center"] = dff[f"{c}_position"]
                dff["end"] = dff[f"{e}_position"]
                dff["res_idx"] = dff[f"{r}_res_idx"]
                dff["atom_idxs"] = atom_idxs[mask]
                dff = dff[output_col_names]
                feat_dfs.append(dff)
        if feat_dfs:
            feats = pd.concat(feat_dfs, ignore_index=True).reset_index(drop=True)
        else:
            feats = pd.DataFrame(columns=output_col_names)
        return self._pharmacophore(
            features=feats,
            feature_types={
                feature_type for feature_type in (
                    type_hbond_acceptor,
                    type_hbond_donor,
                    type_anionic,
                    type_cationic,
                    type_hydrophobic,
                    type_aromatic,
                    type_halogen_acceptor,
                    type_halogen_donor
                ) if feature_type is not None
            },
            inputs=[],
            extra={"plip": plip},
            name=name or f"{self.system.name} Pharmacophore",
        )

    def from_structure(
        self,
        *,
        type_hbond_acceptor: str | None = "OA",
        type_hbond_donor: str | None = "HD",
        type_anionic: str | None = "e-",
        type_cationic: str | None = "e+",
        type_hydrophobic: str | None = "C",
        atom_mask: pd.Series | np.ndarray | Sequence[bool] | None = None,
        len_hbond: float = 2.5,
        len_ionic: float = 3.0,
        len_hydrophobic: float = 4.0,
        name: str | None = None,
    ) -> Pharmacophore:
        """Perceive pharmacophore features from the system structure.

        This method can create vector features
        for hydrogen bond acceptors and donors,
        and radial features for anionic, cationic, and hydrophobic interactions.
        If any of the `type_*` parameters are set to `None`,
        the corresponding feature type will not be generated.

        Parameters
        ----------
        type_hbond_acceptor
            Feature type ID for hydrogen bond acceptor features.
        type_hbond_donor
            Feature type ID for hydrogen bond donor features.
        type_anionic
            Feature type ID for anionic features.
        type_cationic
            Feature type ID for cationic features.
        type_hydrophobic
            Feature type ID for hydrophobic features.
        atom_mask
            Optional boolean mask to select a subset of atoms
            from the system for pharmacophore feature perception.
            If provided, only atoms in `self.system.composition.atoms`
            where the mask is `True` will be considered when perceiving features.
        len_hbond
            Ideal length of a hydrogen bond.
        len_ionic
            Ideal length of an ionic interaction.
        len_hydrophobic
            Ideal length of a hydrophobic interaction.
        name
            Optional name for the pharmacophore.

        Notes
        -----
        The algorithm works as follows:
        1. For hydrogen bond acceptor features,
            all hydrogen bond donor atoms in the (masked) system are identified
            using their AutoDock atom types (i.e., hydrogen atoms with type "HD").
            Then, for each donor hydrogen atom,
            its bonded heavy atom is found.
            A hydrogen bond acceptor feature is then created
            as a vector with its `end` at the position of the donor hydrogen atom,
            and its `center` at a position that is `len_hbond` away
            from the donor hydrogen atom along the H–X bond.
        2. For hydrogen bond donor features,
            all hydrogen bond acceptor atoms in the (masked) system are identified
            using their AutoDock atom types
            (i.e., nitrogen and oxygen atoms with types "NA" and "OA").
            For each acceptor atom, N hydrogen bond donor features are created
            (N being the number of hydrogen bonds the acceptor atom can form,
            i.e., 1 for nitrogen and 2 for oxygen).
            as vectors with their `end` at the position of the acceptor atom,
            and their `center` at positions that are `len_hbond` away
            from the acceptor atom along the directions of the acceptor atom's lone pairs.
            The lone pair directions are approximated
            using the number and positions of the atoms bonded to the acceptor atom
            to fill either a trigonal planar
            or tetrahedral geometry around the acceptor atom as follows:
            - Nitrogen bonded to 2 atoms: one lone pair to fill trigonal planar geometry.
            - Nitrogen bonded to 3 atoms: one lone pair to fill tetrahedral geometry.
            - Oxygen bonded to 1 atom: two lone pairs to fill trigonal planar geometry
              in-plane with the O–X–Y plane (Y being the heaviest atom bonded to X).
              This is the ideal orientation for oxygens with one (mesomeric) double bond,
              and is a good approximation for oxygens with one single bond.
            - Oxygen bonded to 2 atoms: two lone pairs to fill tetrahedral geometry.
        3. For anionic features,
            all cationic atoms in the (masked) system are identified
            using their formal charges.
            For each cationic atom,
            an anionic feature is created as a radial feature
            with its `end` at the position of the cationic atom,
            and its `radius` set to `len_ionic`.
        4. For cationic features,
            all anionic atoms in the (masked) system are identified
            using their formal charges.
            For each anionic atom,
            a cationic feature is created as a radial feature
            with its `end` at the position of the anionic atom,
            and its `radius` set to `len_ionic`.
        5. For hydrophobic features,
            all hydrophobic atoms in the (masked) system are identified
            using their AutoDock atom types and bonded element types
            (i.e., aliphatic carbon atoms with type "C" that are
            only bonded to carbon or hydrogen atoms).
            For each hydrophobic atom,
            a hydrophobic feature is created as a radial feature
            with its `end` at the position of the hydrophobic atom,
            and its `radius` set to `len_hydrophobic`.

        All atom and bonding information is obtained from the Chemical Component Dictionary.
        """
        if self._structure_modeler is None:
            self._verify_system_available("from_structure")
            self._structure_modeler = StructureBasedModeler(self.system)
        feats = self._structure_modeler.model(
            type_hbond_acceptor=type_hbond_acceptor,
            type_hbond_donor=type_hbond_donor,
            type_anionic=type_anionic,
            type_cationic=type_cationic,
            type_hydrophobic=type_hydrophobic,
            atom_mask=atom_mask,
            len_hbond=len_hbond,
            len_ionic=len_ionic,
            len_hydrophobic=len_hydrophobic,
        )
        return self._pharmacophore(
            features=feats,
            feature_types={
                feature_type for feature_type in (
                    type_hbond_donor,
                    type_hbond_acceptor,
                    type_cationic,
                    type_anionic,
                    type_hydrophobic,
                ) if feature_type is not None
            },
            inputs=[],
            extra={},
            name=name or f"{self.system.name} Pharmacophore",
        )

    def _field_filter(self, filter_function: dict[str, FilterFunction | None]) -> np.ndarray:
        """Apply the specified filter functions to the field tensor."""
        tensor = np.empty_like(self.field.tensor)
        for feature_field_idx, feature_field in enumerate(self.field.tensor):
            feature_type = self._feature_type(feature_field_idx)
            function = filter_function[feature_type]
            if function is None:
                tensor[feature_field_idx] = feature_field
            else:
                function(
                    input=feature_field,
                    output=tensor[feature_field_idx],
                    axes=(-3, -2, -1),
                )
        return tensor

    def _field_mask(
        self,
        tensor: np.ndarray,
        peak_type: dict[str, Literal["min", "max"]],
        best_per_point: dict[str, bool],
        threshold_value: dict[str, float | None],
        threshold_percentile: dict[str, float | None],
        threshold_include_equal: dict[str, bool],
    ):
        mask = self._field_mask_best_per_point(
            tensor=tensor,
            peak_type=peak_type,
            best_per_point=best_per_point,
        )
        if self.pocket is not None:
            mask = np.logical_and(mask, self.pocket.tensor)
        mask = self._field_mask_threshold(
            tensor=tensor,
            mask=mask,
            peak_type=peak_type,
            threshold_value=threshold_value,
            threshold_percentile=threshold_percentile,
            threshold_include_equal=threshold_include_equal,
        )
        return mask

    def _field_mask_threshold(
        self,
        tensor: np.ndarray,
        mask: np.ndarray,
        peak_type: dict[str, Literal["min", "max"]],
        threshold_value: dict[str, float | None],
        threshold_percentile: dict[str, float | None],
        threshold_include_equal: dict[str, bool],
    ):
        comparison_function = {
            ("min", True): np.less_equal,
            ("min", False): np.less,
            ("max", True): np.greater_equal,
            ("max", False): np.greater,
        }
        for feature_idx in range(mask.shape[0]):
            feature_type = self._feature_type(feature_idx)
            comp_func = comparison_function[
                    (peak_type[feature_type], threshold_include_equal[feature_type])
                ]
            if threshold_value[feature_type] is not None:
                mask[feature_idx] = np.logical_and(
                    mask[feature_idx],
                    comp_func(tensor[feature_idx], threshold_value[feature_type])
                )
            if threshold_percentile[feature_type] is not None:
                percentile = threshold_percentile[feature_type]
                if peak_type[feature_type] == "max":
                    percentile = 100 - percentile
                threshold = np.percentile(
                    tensor[feature_idx][mask[feature_idx]],
                    percentile
                )
                mask[feature_idx] = np.logical_and(
                    mask[feature_idx],
                    comp_func(tensor[feature_idx], threshold)
                )
        return mask

    def _field_mask_best_per_point(
        self,
        tensor: np.ndarray,
        peak_type: dict[str, Literal["min", "max"]],
        best_per_point: dict[str, bool]
    ) -> np.ndarray:
        """Compute a boolean mask to select best feature per point.

        For each feature `f` in `field`:
        - If `best_per_mode[f]` is `False`, `mask[f]` is all `True` (no filtering).
        - If `best_per_mode[f]` is `True`, `mask[f]` is `True` only where `field[f]`
          is the min (if `peak_type[f]=='min'`) or max (if 'max') across features.

        Parameters
        ----------
        tensor
            Input tensor of shape `(n_features, *n_batches, nx, ny, nz)`.
        peak_type
            Sequence of length `n_features` where each element is 'min' or 'max',
            determining which extremum to filter for when `best_per_point[f]` is True.
        best_per_point
            Boolean sequence of length `n_features`, where each element indicates
            whether to filter for the best value for each feature type.

        Returns
        -------
        Boolean array of same shape as `field` with per-feature filtering.

        Notes
        -----
        - Computes argmin and/or argmax only if at least one feature needs it.
        - Initializes an all-True mask and overrides only the flagged features.
        - Ties broken by first occurrence via argmin/argmax.
        """
        # Determine which extrema to compute
        need_min = any(best and peak == 'min' for best, peak in zip(best_per_point.values(), peak_type.values()))
        need_max = any(best and peak == 'max' for best, peak in zip(best_per_point.values(), peak_type.values()))

        # Compute indices of extrema if needed
        idx_min = np.argmin(tensor, axis=0) if need_min else None
        idx_max = np.argmax(tensor, axis=0) if need_max else None

        # Start with all True mask
        mask = np.ones_like(tensor, dtype=bool)

        # Override flagged features based on their peak type
        for feature_idx in range(mask.shape[0]):
            feature_type = self._feature_type(feature_idx)
            if not best_per_point[feature_type]:
                continue
            peak_type_value = peak_type[feature_type]
            if peak_type_value == 'min':
                mask[feature_idx] = (idx_min == feature_idx)
            elif peak_type_value == 'max':
                mask[feature_idx] = (idx_max == feature_idx)
            else:
                raise ValueError(
                    f"Invalid `peak_type` '{peak_type_value}' for feature '{feature_type}'; must be 'min' or 'max'"
                )
        return mask

    def _pharmacophore(
        self,
        features: pd.DataFrame,
        feature_types: set[str],
        inputs: list[dict[str, Any]],
        extra: dict[str, Any] | None = None,
        name: str = "Pharmacophore",
    ) -> Pharmacophore:
        return Pharmacophore(
            features=features,
            feature_types=feature_types,
            inputs=inputs,
            system=self.system,
            pocket=self.pocket,
            field=self.field,
            extra=extra,
            name=name,
        )

    def _feature_type(self, field_idx: int) -> str:
        """Get the feature type for a given field index."""
        return self.field.batch_instance_labels["feature"][field_idx]

    def _verify_field_pocket_compatible(self):
        if self.pocket.grid != self.field.grid:
            raise ValueError(
                "Pocket and field must have the same grid, "
                f"but got pocket grid {self.pocket.grid} and field grid {self.field.grid}."
            )
        if self.pocket.tensor.shape != self.field.tensor.shape[1:] and self.pocket.tensor.shape != self.field.tensor.shape[-3:]:
            raise ValueError(
                "Pocket and field tensors must have the same shape along their last dimensions, "
                f"but got pocket tensor shape {self.pocket.tensor.shape} "
                f"and field tensor shape {self.field.tensor.shape}."
            )
        return

    def _verify_system_available(self, method_name: str):
        if self.system is None:
            raise ValueError(
                f"Cannot use the `{method_name}` modeling method "
                "because no system was provided. "
                "Please provide a system."
            )
        return

    def _verify_field_available(self, method_name: str):
        if self.field is None:
            raise ValueError(
                f"Cannot use the `{method_name}` modeling method "
                "because no field was provided. "
                "Please provide a field."
            )
        return