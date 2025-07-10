"""Pharmacophore."""

from typing import Sequence, Any, Self
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
        - `type`: A string representing the feature type.
        - `center`: A sequence of three real numbers representing
           the 3D coordinates of the feature in some reference frame.
        - `radius`: A non-negative real number representing the feature radius.
           If not present, it will be added with a default value of 0.
    feature_types

    extra
        Optional dictionary to bundle additional information
        related to the pharmacophore, such as metadata or processing results.
        This is not used by this class, but can be useful for downstream processing.
    system
        Optional chemical system associated with the pharmacophore.
        If provided, it is used by the `display()` method
        to visualize the pharmacophore in the context of the chemical structure.
    """

    def __init__(
        self,
        features: DataFrameLike,
        feature_types: set[str],
        inputs: dict[str, Any] | None = None,
        name: str = "Pharmacophore",
        system: System | None = None,
        pocket: Pocket | None = None,
        field: Field | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self._features = _PharmacophoreInput(features=features).features
        self._feature_types = feature_types
        self._inputs = inputs or {}
        self._name = name
        self._system = system
        self._pocket = pocket
        self._field = field
        self._extra = extra or {}

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
    def feature_types(self) -> set[str]:
        """Set of feature types."""
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
        """Pocket mask associated with the pharmacophore."""
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
        ligand: Self | DataFrameLike,
        max_distance: float | None = None
    ) -> pd.DataFrame:
        if isinstance(ligand, Pharmacophore):
            ligand = ligand.features
        elif not isinstance(ligand, pd.DataFrame):
            try:
                ligand = pd.DataFrame(ligand)
            except Exception as e:
                raise ValueError(
                    "Invalid ligand input. The input cannot be converted to a DataFrame. "
                    "Expected a DataFrame-like input or LigandPharmacophore instance."
                ) from e
        ligand_features = set(ligand['type'])
        if (invalid_features := ligand_features - self.feature_types):
            raise ValueError(
                f"Invalid feature values: {sorted(invalid_features)}. "
                f"Allowed: {sorted(self.feature_types)}"
            )
        ligand = ligand.reset_index().rename(columns={'index': 'ligand_idx'})

        # Get all unique instances
        instances = pd.DataFrame({'instance': self.features['instance'].unique()})
        # Cross-join ligand × instance
        ligand['_key'] = 1
        instances['_key'] = 1
        cross = ligand.merge(instances, on='_key').drop(columns=['_key'])
        # Merge with features on instance & type
        feat = self.features.rename(columns={'radius': 'radius_feat'})
        merged = cross.merge(
            feat[['instance', 'type', 'label', 'center', 'radius_feat']],
            on=['instance', 'type'],
            how='left'
        ).rename(columns={'radius': 'radius_lig'})
        merged['match'] = False
        merged['distance'] = np.nan
        merged['max_distance'] = np.nan
        # Compute distances where feature exists
        mask = merged['center'].notna()
        if mask.any():
            pos_arr = np.stack(merged.loc[mask, 'position'].values)
            cen_arr = np.stack(merged.loc[mask, 'center'].values)
            merged.loc[mask, 'distance'] = np.linalg.norm(pos_arr - cen_arr, axis=1)
            merged.loc[mask, 'max_distance'] = (
                merged.loc[mask, 'radius_lig'] + merged.loc[mask, 'radius_feat']
            ) if max_distance is None else max_distance
            merged.loc[mask, 'match'] = merged.loc[mask, 'distance'] < merged.loc[mask, 'max_distance']
        # Defaults for missing-feature cases
        merged['distance'] = merged['distance'].astype(float)
        merged['max_distance'] = merged['max_distance'].astype(float)
        # Pick minimum-distance feature per ligand_idx×instance
        # Treat NaN distances as +inf so real distances sort first
        merged['dist_sort'] = merged['distance'].fillna(np.inf)
        best = (
            merged
            .sort_values(['ligand_idx', 'instance', 'dist_sort'])
            .drop_duplicates(['ligand_idx', 'instance'], keep='first')
            .drop(columns=['dist_sort', 'position', 'center', 'radius_lig', 'radius_feat'])
        )
        # Reorder & return
        final_cols = ['ligand_idx', 'instance', 'label', 'type', 'match', 'distance', 'max_distance']
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
            if feature not in self.feature_types:
                raise ValueError(f"Invalid feature type: {feature}. Allowed: {self.feature_types}")
            if isinstance(color, Sequence) and len(color) == 3:
                if all(isinstance(c, (int, float)) for c in color):
                    self._feature_colors[feature] = tuple(color)
                else:
                    raise ValueError(f"Invalid color format for feature '{feature}': {color}")
            else:
                raise ValueError(f"Color must be a tuple of three values for feature '{feature}'")
        return


def from_plip(
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



class _PharmacophoreInput(BaseModel):
    """Model to validate and normalize `Pharmacophore` inputs.

    This model accepts any input convertible to a pandas DataFrame and ensures:
    - Columns 'type' and 'center' are present.
    - 'type' values are strings.
    - 'center' entries become 1D numpy arrays of three floats.
    - A non-negative 'radius' column is present (added with zeros if missing).

    Attributes
    ----------
    features
        Normalized DataFrame with columns ['type', 'center', 'radius'].
    """
    features: pd.DataFrame

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

        # Validate 'feature'
        if not pd.api.types.is_string_dtype(df['type']):
            raise ValueError("Feature column 'type' must be strings")

        # Validate and normalize 'center' column
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

        self.features = df.convert_dtypes()
        return self
