"""Pharmacophore."""

from typing import Sequence, Any, Self, Literal
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import scishow
import caddpy

from t2fpharm.system import System
from t2fpharm.pocket import Pocket
from t2fpharm.field import Field
from t2fpharm.typing import DataFrameLike


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
        self._feature_types = feature_types
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
