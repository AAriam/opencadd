
from typing import Any, Sequence, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

import numpy as np
import pandas as pd


class Pharmacophore:
    def __init__(
        self,
        features: pd.DataFrame,
        field,
    ):
        self.features = features
        self.field = field
        return

    def match_spherical(self, ligand: Any) -> pd.DataFrame:
        ligand = _LigandInput(
            ligand=ligand,
            allowed_features=self.field.ids
        ).ligand.reset_index().rename(columns={'index': 'ligand_idx'})
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
        # Compute distances where feature exists
        mask = merged['center'].notna()
        if mask.any():
            pos_arr = np.stack(merged.loc[mask, 'position'].values)
            cen_arr = np.stack(merged.loc[mask, 'center'].values)
            merged.loc[mask, 'distance'] = np.linalg.norm(pos_arr - cen_arr, axis=1)
            merged.loc[mask, 'max_distance'] = (
                merged.loc[mask, 'radius_lig'] + merged.loc[mask, 'radius_feat']
            )
            merged.loc[mask, 'match'] = merged.loc[mask, 'distance'] < merged.loc[mask, 'max_distance']
        # Defaults for missing-feature cases
        merged['distance'] = merged['distance'].astype(float)
        merged['max_distance'] = merged['max_distance'].astype(float)
        merged['match'] = merged['match'].fillna(False)
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
        final_cols = ['ligand_idx', 'instance', 'match', 'label', 'distance', 'max_distance']
        return best[final_cols].reset_index(drop=True)


class _LigandInput(BaseModel):
    """
    Pydantic v2 model to validate and normalize a ligand DataFrame.

    This model accepts any input convertible to a pandas DataFrame and ensures:
    - Columns 'feature' and 'position' are present.
    - 'feature' values are strings drawn from `allowed_features`.
    - 'position' entries become 1D numpy arrays of three floats.
    - A non-negative 'radius' column is present (added with zeros if missing).

    Attributes
    ----------
    ligand
        Normalized DataFrame with columns ['feature', 'position', 'radius'].
    """
    ligand: pd.DataFrame
    allowed_features: Sequence[str]

    # Allow arbitrary types like pandas DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('ligand', mode='before')
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
        """Validate and normalize the ligand DataFrame."""
        def to_array(val: Any) -> np.ndarray:
            """Convert position to a 1D numpy array of three floats."""
            arr = np.asarray(val, dtype=float)
            if arr.shape != (3,):
                raise ValueError(
                    f"Position must be sequence of 3 numbers, got shape {arr.shape}"
                )
            return arr

        df = self.ligand

        # Check required columns
        required_cols = {'feature', 'position'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate 'feature'
        if not pd.api.types.is_string_dtype(df['feature']):
            df['feature'] = df['feature'].astype(str)
        invalid_feats = set(df['feature']) - self.allowed_features
        if invalid_feats:
            raise ValueError(
                f"Invalid feature values: {sorted(invalid_feats)}. "
                f"Allowed: {sorted(self.allowed_features)}"
            )

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

        self.ligand = df
        return self
