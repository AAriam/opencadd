from typing import Any, Sequence, Self
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

import scishow
import caddpy
from t2fpharm.pharmacophore import Pharmacophore
from t2fpharm.receptor import Receptor
from t2fpharm.pocket import Pocket


class LigandPharmacophore(Pharmacophore):
    def __init__(
        self,
        features: Any,
        extra: dict[str, Any] | None = None,
        receptor: Receptor | None = None,
    ):
        self.features = _LigandInput(features=features).features
        self.extra = extra or {}
        self.receptor = receptor
        super().__init__()
        return

    def display(
        self,
        nglwidget: scishow.nglview.NGLWidget | None = None,
        default_radius: float = 1.5,
        feature_colors: dict[str, tuple[float, float, float] | tuple[int, int, int]] | None = None,
    ):
        nv = nglwidget or scishow.nglview.NGLWidget()
        if self.receptor:
            nv.add_trajectory(self.receptor)
        feature_colors = feature_colors or {}
        for feature_idx, feature in self.features.iterrows():
            nv.add_spheres(
                coords=feature["position"],
                radii=feature["radius"] or default_radius,
                name=f"{feature['type'].upper()}_{feature_idx}",
                colors=feature_colors.get(
                    feature["type"],
                    self._feature_colors.get(feature["type"], (0.5, 0.5, 0.5))
                ),
                representation_params=scishow.nglview.RepresentationParameters(
                    opacity=0.8,
                    visible=True,
                    lazy=True,
                )
            )
        return nv.display(gui=True)


class _LigandInput(BaseModel):
    """
    Pydantic v2 model to validate and normalize a ligand DataFrame.

    This model accepts any input convertible to a pandas DataFrame and ensures:
    - Columns 'type' and 'position' are present.
    - 'type' values are strings drawn from `allowed_features`.
    - 'position' entries become 1D numpy arrays of three floats.
    - A non-negative 'radius' column is present (added with zeros if missing).

    Attributes
    ----------
    features
        Normalized DataFrame with columns ['type', 'position', 'radius'].
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
            raise ValueError(f"Cannot convert input to DataFrame: {e}")

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
        required_cols = {'type', 'position'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate 'feature'
        if not pd.api.types.is_string_dtype(df['type']):
            raise ValueError("Feature column 'type' must be strings")

        # Validate and normalize 'position'
        df['position'] = df['position'].apply(to_array)

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


def from_plip(
    pdb_files: str | bytes | Path | Sequence,
    ligands: Sequence[tuple[str, int | str, int]] | None = None,
    type_hbond_donor: str = "HD",
    type_hbond_acceptor: str = "OA",
    type_anion: str = "e-",
    type_cation: str = "e+",
    type_hydrophobic: str = "C",
    pocket: Pocket | None = None,
    receptor: Receptor | None = None,
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
            if entry["type"] == feature_type and np.allclose(entry["position"], position):
                break
        else:
            out.append({"type": feature_type, "position": position})
    if pocket is not None:
        positions = np.stack([feature["position"] for feature in out])
        coverages = pocket.point_coverage(positions)
        out = [feature for feature, coverage in zip(out, coverages) if coverage]
    return LigandPharmacophore(features=out, extra={"plip": plip})
