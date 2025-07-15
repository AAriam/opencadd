from typing import Self, Any, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PharmFeaturesInput(BaseModel):
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
